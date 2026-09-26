"""Windows of G1 tracking rollouts for the text-driven policy (docs/04 line A).

Source: `data/g1_rollouts/*.pkl` — the TextOp tracker following BABEL-annotated AMASS retargeted to the robot.
Each rollout carries `proprio (T,67)`, `action (T,29)`, `frame_ann` (BABEL frame-level labels in seconds),
`success`, `fps=50`.  The BABEL labels ride along on every rollout, so no join against the BABEL release is
needed (unlike the SMPL side, where `scripts/hml_phys/09_babel_boundaries.py` had to map them in).

Three things are deliberately different from the SMPL pipeline, and all three are simplifications:

1. **No canonicalisation.**  `root_lin_vel_b`, `root_ang_vel_b` and `projected_gravity_b` are already in the
   BODY frame, and projected gravity encodes attitude without any heading ambiguity.  The whole "which frame
   is the origin" problem that cost us a 29%-reconstruction detour on the SMPL side simply does not exist here.
2. **Horizons are converted by DURATION, not by frame count.**  The SMPL policy ran at 30 fps with 16 history
   frames (0.53 s) and 4 generated action frames (0.133 s).  G1 runs at 50 fps, so the same durations are 27
   and 7 frames.  Copying the frame counts across would silently change the time scale by 1.67x.
3. **Text is a BABEL frame label**, chosen by overlap with the GENERATED span, not a whole-clip caption.

The intent streams use L_INTENT = 28 frames (0.56 s at 50 fps, matching the SMPL side's 16/30 = 0.53 s) and 28
is divisible by the intent VAE's 4x temporal downsampling.

__getitem__ -> dict(
    x        [H+F, 96]  normalised token, the future F rows' ACTION channels are what the policy generates
    mask     [H+F]      1 on observed history rows
    text     str        the BABEL label for this window (may be "")
    progress, total_len scalars
    hist/fut/holi [28, 67]  normalised proprio sequences for the intent VAE (only when intent=True)
)
"""
import os
from bisect import bisect_right

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

G1_DIR = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/g1_rollouts"
STATS = os.path.join(G1_DIR, "g1_token_stats.npz")
PROPRIO_DIM, ACTION_DIM = 67, 29
TOKEN_DIM = PROPRIO_DIM + ACTION_DIM
FPS = 50.0
L_INTENT = 28                       # 0.56 s at 50 fps; a multiple of the VAE's 4x downsampling

# duration-matched to the SMPL policy (16 and 4 frames at 30 fps)
H_DEFAULT = 27
F_DEFAULT = 7

TRAIN_FILES = ["train_all_x1.pkl", "dagger4_trainall_v4_mix05_randprompt.pkl"]
VAL_FILES = ["val_all_x1.pkl", "dagger1_valall_v1_mix05.pkl",
             "dagger2_valall_v3_mix05.pkl", "dagger3_valall_v3_mix05_randprompt.pkl"]


def load_stats(path=STATS):
    z = np.load(path, allow_pickle=True)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


class G1WindowDataset(Dataset):
    """Windows over G1 rollouts.  `split` is 'train' or 'val' -- this dataset has no test split, so val is the
    evaluation split (project CLAUDE.md §1)."""

    def __init__(self, split="train", H=H_DEFAULT, F=F_DEFAULT, stride=1, files=None, success_only=True,
                 stats_path=STATS, intent=False, max_rollouts=0, seed=0, normalise=True):
        assert split in ("train", "val"), f"G1 has only train/val, got {split!r}"
        self.split, self.H, self.F, self.stride = split, int(H), int(F), int(stride)
        self.T = self.H + self.F
        self.intent = bool(intent)
        self.normalise = bool(normalise)
        self.mean, self.std = load_stats(stats_path)
        self.rng = np.random.RandomState(seed)

        files = files or (TRAIN_FILES if split == "train" else VAL_FILES)
        self.pro, self.act, self.ann, self.meta = [], [], [], []
        need = max(self.T, L_INTENT + self.F, L_INTENT)
        for fn in files:
            p = os.path.join(G1_DIR, fn)
            if not os.path.exists(p):
                continue
            obj = joblib.load(p)
            rolls = obj["rollouts"] if isinstance(obj, dict) else obj
            for r in rolls:
                if success_only and not bool(r.get("success", False)):
                    continue
                pro, act = r["proprio"], r["action"]
                if pro.shape[0] < need or pro.shape[0] != act.shape[0]:
                    continue
                if not (np.isfinite(pro).all() and np.isfinite(act).all()):
                    continue
                self.pro.append(np.ascontiguousarray(pro, np.float32))
                self.act.append(np.ascontiguousarray(act, np.float32))
                # BABEL labels, sorted by start time, kept in seconds; `transition` is BABEL's own label
                self.ann.append(sorted([(float(a), float(b), str(lab)) for a, b, lab, *_ in r.get("frame_ann", [])]))
                self.meta.append(dict(file=fn, motion=str(r.get("motion", "")), n=int(pro.shape[0])))
                if max_rollouts and len(self.pro) >= max_rollouts:
                    break
            del obj, rolls
            if max_rollouts and len(self.pro) >= max_rollouts:
                break
        assert self.pro, f"no usable rollouts for split {split}"

        # window index: s = index of the first HISTORY frame; the generated span is [s+H, s+H+F)
        idx = []
        for i, a in enumerate(self.pro):
            n = a.shape[0]
            lo = max(0, L_INTENT - self.H)          # leave room for the intent history when it is needed
            for s in range(lo, n - self.T + 1, self.stride):
                idx.append((i, s))
        self.windows = np.asarray(idx, np.int64).reshape(-1, 2)
        self.n_rollouts = len(self.pro)

    def __len__(self):
        return len(self.windows)

    def label_for(self, i, a, b):
        """the BABEL label covering the most of the frame span [a, b) of rollout i; '' if nothing covers it."""
        t0, t1 = a / FPS, b / FPS
        best, best_ov = "", 0.0
        for (sa, sb, lab) in self.ann[i]:
            ov = min(sb, t1) - max(sa, t0)
            if ov > best_ov:
                best, best_ov = lab, ov
        return best

    def norm(self, x):
        return (x - self.mean) / self.std if self.normalise else x

    def _state_seq(self, i, a, b):
        """normalised proprio rows [a, b) -- the 67-d state the intent VAE consumes."""
        s = self.pro[i][a:b]
        return ((s - self.mean[:PROPRIO_DIM]) / self.std[:PROPRIO_DIM]).astype(np.float32) if self.normalise else s

    def __getitem__(self, k):
        i, s = map(int, self.windows[k])
        n = self.pro[i].shape[0]
        j = s + self.H                                    # first generated frame
        tok = np.concatenate([self.pro[i][s:s + self.T], self.act[i][s:s + self.T]], -1)
        x = self.norm(tok).astype(np.float32)
        mask = np.zeros(self.T, np.float32); mask[:self.H] = 1.0
        out = dict(x=torch.from_numpy(x), mask=torch.from_numpy(mask),
                   text=self.label_for(i, j, j + self.F),
                   progress=float(j) / max(1.0, float(n)), total_len=float(n) / FPS)
        if self.intent:
            out["hist"] = torch.from_numpy(self._state_seq(i, j - L_INTENT, j))
            out["fut"] = torch.from_numpy(self._state_seq(i, j, min(n, j + L_INTENT)))
            if out["fut"].shape[0] < L_INTENT:            # pad the tail by repeating the last frame
                pad = out["fut"][-1:].repeat(L_INTENT - out["fut"].shape[0], 1)
                out["fut"] = torch.cat([out["fut"], pad], 0)
            rows = np.round(np.linspace(0, n - 1, L_INTENT)).astype(np.int64)   # holistic = whole rollout, 28 rows
            out["holi"] = torch.from_numpy(self._state_seq(i, 0, n)[rows])
        return out


def collate_g1(batch, clip_encoder=None, cache=None):
    """Stack a batch.  Text is returned as the raw label list; the trainer encodes it (CLIP is on the GPU)."""
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "mask") if k in batch[0]}
    for k in ("hist", "fut", "holi"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    out["text"] = [b["text"] for b in batch]
    out["progress"] = torch.tensor([b["progress"] for b in batch], dtype=torch.float32)
    out["total_len"] = torch.tensor([b["total_len"] for b in batch], dtype=torch.float32)
    return out
