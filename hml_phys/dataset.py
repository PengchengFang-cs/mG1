"""Sliding-window dataset for the physics policy (docs/07 §1, §1.5, §2, §11).

Frame convention (from the PHC recorder): row t = (state AFTER step t, action applied AT step t), i.e. the action that
produced this row's state; the closed-loop buffer (mc_rollout.HistoryBuffer) pushes exactly the same pairing.
v2 (whole_sequence=True): the future runs to the clip end (capped at F frames, variable length, `valid` mask);
v1: fixed F-frame future.
Each sample = window of T = H + F frames from one tracked clip:
  root  [T,15], body [T,420]  (normalised tokens, window-canonical frame; see hml_phys.tokens)
  observed_mask [T]  1 for the H history frames, 0 for the F future frames
  text: index into the CLIP cache (-1 = empty / unconditional), progress in [0,1], total_len (seconds)
Augmentations (all training-time only):
  * start-rest (p_rest): window at clip start, history = 16 copies of frame 0 with zero velocities and the
    hold action of frame 0; future = clip frames [0:F).
  * neutral-pose (p_neutral, only clips that start near rest): history = 16 copies of the evaluation
    neutral standing state (from env_constants) with hold action; future = clip frames [0:F).
  * history noise (sigma_hist > 0): gaussian noise on the normalised history tokens (off by default).
Text rule: captions with time tags are candidates only for windows whose future overlaps the tag span
(on the clip's original timeline, physics frame i <-> i / fps_eff seconds); untagged captions are always
candidates; no candidate -> empty text.
"""
import json, os
import numpy as np, joblib, torch
from torch.utils.data import Dataset
from hml_phys import tokens as tk

ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/humanml3d_phys"


def norm_caption(c):
    return " ".join(c.strip().split())


def _hip_yaw(bp):
    """yaw (rad) of the hip-across direction R_hip - L_hip (MuJoCo order 5, 1) in the xy plane."""
    d = bp[tk.R_HIP, :2] - bp[tk.L_HIP, :2]
    return float(np.arctan2(d[1], d[0]))


def _rotate_z(bp, rs, ang, center):
    """rotate body_pos [T,24,3] and root_state [T,13] about the vertical axis through `center` (xy) by `ang`."""
    c, s_ = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    ctr = np.array([center[0], center[1], 0.0])
    bp = ((bp - ctr) @ Rz.T + ctr).astype(bp.dtype)
    rs = rs.copy()
    rs[:, :3] = (rs[:, :3] - ctr) @ Rz.T + ctr
    from scipy.spatial.transform import Rotation as R
    rs[:, 3:7] = (R.from_matrix(Rz) * R.from_quat(rs[:, 3:7])).as_quat()
    rs[:, 7:10] = rs[:, 7:10] @ Rz.T; rs[:, 10:13] = rs[:, 10:13] @ Rz.T
    return bp, rs.astype(rs.dtype)


class TextCache:
    def __init__(self, path=os.path.join(ROOT, "text_cache_clipL14")):
        self.captions = json.load(open(os.path.join(path, "captions.json")))
        self.index = {c: i for i, c in enumerate(self.captions)}
        self.tokens = np.load(os.path.join(path, "tokens.npy"))  # fully in RAM (2.5 GB fp16); shared with forked workers
        self.pooled = np.load(os.path.join(path, "pooled.npy"))
        self.lengths = np.load(os.path.join(path, "lengths.npy"))
        self.dim = self.pooled.shape[1]; self.max_tokens = self.tokens.shape[1]

    def get(self, idx):
        """-> tokens [L,D] float32, pooled [D], length int (idx<0 -> zeros, length 0)."""
        if idx < 0:
            return np.zeros((self.max_tokens, self.tokens.shape[2]), np.float32), np.zeros(self.dim, np.float32), 0
        return np.asarray(self.tokens[idx], dtype=np.float32), self.pooled[idx], int(self.lengths[idx])


class TokenStats:
    """per-dim mean/std for root and body tokens (MotionCraft-style std = sqrt(var + eps))."""
    def __init__(self, path):
        z = np.load(path)
        self.root_mean, self.root_std = z["root_mean"], z["root_std"]
        self.body_mean, self.body_std = z["body_mean"], z["body_std"]

    def norm(self, root, body):
        return (root - self.root_mean) / self.root_std, (body - self.body_mean) / self.body_std

    def denorm(self, root, body):
        return root * self.root_std + self.root_mean, body * self.body_std + self.body_mean

    @staticmethod
    def fit(root_windows, body_windows, eps=1e-5):
        r = np.concatenate(root_windows, 0).astype(np.float64); b = np.concatenate(body_windows, 0).astype(np.float64)
        return dict(root_mean=r.mean(0).astype(np.float32), root_std=np.sqrt(r.var(0) + eps).astype(np.float32),
                    body_mean=b.mean(0).astype(np.float32), body_std=np.sqrt(b.var(0) + eps).astype(np.float32))


class PhysWindowDataset(Dataset):
    def __init__(self, split, H=16, F=32, stats_path=None, text_cache=None, env_constants=None,
                 p_rest=0.1, p_neutral=0.05, sigma_hist=0.0, stride=1, seed=0, train=True, max_clips=0,
                 whole_sequence=False, F_min=8):
        """whole_sequence=True (v2): the future runs from the history end to the clip end, capped at F frames
        (variable length; padded frames are marked invalid). whole_sequence=False (v1): fixed F-frame future."""
        self.H, self.F, self.T = H, F, H + F
        self.whole_sequence, self.F_min = whole_sequence, F_min
        self.train = train
        self.p_rest, self.p_neutral, self.sigma_hist = (p_rest, p_neutral, sigma_hist) if train else (0.0, 0.0, 0.0)
        d = joblib.load(os.path.join(ROOT, f"hml_phys_{split}.pkl"))
        if max_clips:
            d = {k: v[:max_clips] for k, v in d.items()}
        self.clips = d
        self.text_cache = text_cache
        self.stats = TokenStats(stats_path) if stats_path else None
        self.env = env_constants  # dict(neutral_body_pos [24,3], neutral_dof_pos [69], neutral_root_state [13], pd_offset [69], pd_scale [69])
        self.rng = np.random.RandomState(seed)
        # window index: (clip, start)
        idx = []
        self.rest_start = []
        for i, T in enumerate(d["n_frames"]):
            bp = d["body_pos"][i]
            v = np.linalg.norm(np.diff(bp[:min(16, len(bp))], axis=0), axis=-1).mean() * 30 if len(bp) > 1 else 0.0
            self.rest_start.append(bool(v < 0.15 and bp[0, 0, 2] > 0.8))  # near rest AND upright
            if self.whole_sequence:
                if T >= self.H + self.F_min:
                    idx += [(i, s) for s in range(0, T - self.H - self.F_min + 1, stride)]
            elif T >= self.T:
                idx += [(i, s) for s in range(0, T - self.T + 1, stride)]
        self.windows = np.array(idx, dtype=np.int64)
        self.n_clips_short = int(sum(1 for T in d["n_frames"] if T < (self.H + self.F_min if self.whole_sequence else self.T)))
        # text candidates per clip: list of (cache_idx, f_sec, t_sec)  (f=t=0 -> untagged)
        self.cands = []
        for texts in d["texts"]:
            c = []
            for t in texts:
                ci = self.text_cache.index.get(norm_caption(t["caption"]), -1) if self.text_cache else -1
                c.append((ci, float(t["f_tag"]), float(t["to_tag"])))
            self.cands.append(c)

    def __len__(self):
        return len(self.windows)

    def future_len(self, clip, s):
        """number of generated frames for window (clip, s)."""
        if not self.whole_sequence:
            return self.F
        return int(min(self.F, int(self.clips["n_frames"][clip]) - (s + self.H)))

    # ---- text
    def pick_text(self, clip, s, rng):
        fps = float(self.clips["fps_eff"][clip])
        f0, f1 = (s + self.H) / fps, (s + self.H + self.future_len(clip, max(s, 0))) / fps  # future span in original seconds
        cand = [ci for ci, a, b in self.cands[clip] if (a == 0.0 and b == 0.0) or (a < f1 and b > f0)]
        cand = [ci for ci in cand if ci >= 0]
        return int(cand[rng.randint(len(cand))]) if cand else -1

    # ---- raw window with augmentations
    def raw_window(self, clip, s, mode):
        """-> raw arrays of length H + F_used (F_used = future_len; variable in whole-sequence mode)."""
        c = self.clips
        Fu = self.future_len(clip, s) if mode == "normal" else min(self.F, int(c["n_frames"][clip]))
        Tw = self.H + Fu
        bp, ds, rs, ac = (c["body_pos"][clip][s:s + Tw].copy(), c["dof_state"][clip][s:s + Tw].copy(),
                          c["root_state"][clip][s:s + Tw].copy(), c["action"][clip][s:s + Tw].copy())
        if mode in ("rest", "neutral"):
            # history = 16 static frames, future = clip frames [0:Fu)  (s == 0 by construction)
            fut = slice(0, Fu)
            bp_f, ds_f, rs_f, ac_f = c["body_pos"][clip][fut], c["dof_state"][clip][fut], c["root_state"][clip][fut], c["action"][clip][fut]
            if mode == "rest":
                bp0, dof0, rs0 = bp_f[0], ds_f[0, :, 0], rs_f[0].copy()
            else:
                # neutral standing history: rotate the clip's future about z so that its frame-0 heading (hip-across
                # direction) matches the neutral pose, and place the neutral pose at the clip's start xy
                bp0, dof0, rs0 = self.env["neutral_body_pos"].copy(), self.env["neutral_dof_pos"], self.env["neutral_root_state"].copy()
                yaw_n = _hip_yaw(bp0); yaw_c = _hip_yaw(bp_f[0])
                bp_f, rs_f = _rotate_z(bp_f, rs_f, yaw_n - yaw_c, center=bp_f[0, 0, :2])
                shift = bp_f[0, 0, :2] - bp0[0, :2]
                bp0[:, :2] += shift; rs0[:2] += shift
            rs0[7:13] = 0.0
            hold = tk.hold_action(dof0, self.env["pd_offset"], self.env["pd_scale"]).astype(np.float32)
            bp = np.concatenate([np.repeat(bp0[None], self.H, 0), bp_f], 0)
            ds_h = np.zeros((self.H,) + ds_f.shape[1:], ds_f.dtype); ds_h[:, :, 0] = dof0
            ds = np.concatenate([ds_h, ds_f], 0)
            rs = np.concatenate([np.repeat(rs0[None], self.H, 0), rs_f], 0)
            ac = np.concatenate([np.repeat(hold[None], self.H, 0), ac_f], 0)
        return bp, ds, rs, ac

    def __getitem__(self, i):
        clip, s = map(int, self.windows[i])
        rng = np.random.RandomState((self.rng.randint(1 << 30) + i) % (1 << 31)) if self.train else np.random.RandomState(i)
        mode = "normal"
        if self.train and self.env is not None:
            u = rng.rand()
            if u < self.p_neutral:
                if self.rest_start[clip]:
                    mode, s = "neutral", 0
            elif u < self.p_neutral + self.p_rest:
                mode, s = "rest", 0
        bp, ds, rs, ac = self.raw_window(clip, s, mode)
        root, body = tk.window_tokens(bp, ds, rs, ac)
        if self.stats is not None:
            root, body = self.stats.norm(root, body)
        if self.sigma_hist > 0:
            root[:self.H] += rng.randn(self.H, root.shape[1]).astype(np.float32) * self.sigma_hist
            body[:self.H] += rng.randn(self.H, body.shape[1]).astype(np.float32) * self.sigma_hist
        Tw = root.shape[0]
        valid = np.zeros(self.T, np.float32); valid[:Tw] = 1.0
        if Tw < self.T:  # pad (whole-sequence mode); padded frames are invalid
            root = np.concatenate([root, np.zeros((self.T - Tw, root.shape[1]), np.float32)], 0)
            body = np.concatenate([body, np.zeros((self.T - Tw, body.shape[1]), np.float32)], 0)
        T_clip = int(self.clips["n_frames"][clip]); fps = float(self.clips["fps_eff"][clip])
        now = (0 if mode != "normal" else s) + (0 if mode != "normal" else self.H)  # frames of the clip already executed
        progress = now / max(1, T_clip)
        total_len = T_clip / fps
        ti = self.pick_text(clip, s if mode == "normal" else -self.H, rng)  # augmented: future = clip frames [0:F)
        mask = np.zeros(self.T, np.float32); mask[:self.H] = 1.0
        return dict(root=torch.from_numpy(root.astype(np.float32)), body=torch.from_numpy(body.astype(np.float32)),
                    observed_mask=torch.from_numpy(mask), valid=torch.from_numpy(valid), text_idx=ti, progress=np.float32(progress),
                    total_len=np.float32(total_len), clip=clip, start=s, mode=mode, n_frames=Tw)


def collate(batch, text_cache):
    out = {k: torch.stack([b[k] for b in batch]) for k in ["root", "body", "observed_mask", "valid"]}
    Tmax = int(max(b["n_frames"] for b in batch))  # trim padding to the longest window in the batch
    for k in ["root", "body", "observed_mask", "valid"]:
        out[k] = out[k][:, :Tmax].contiguous()
    empty = text_cache.index.get("", -1)  # windows without caption use the CLIP("") features (same as dropout / CFG)
    toks, pooled, lens = zip(*[text_cache.get(b["text_idx"] if b["text_idx"] >= 0 else empty) for b in batch])
    out["text_tokens"] = torch.from_numpy(np.stack(toks)); out["text_pooled"] = torch.from_numpy(np.stack(pooled))
    out["text_len"] = torch.tensor(lens); out["text_dropped"] = torch.tensor([b["text_idx"] < 0 for b in batch])
    out["progress"] = torch.tensor([b["progress"] for b in batch]); out["total_len"] = torch.tensor([b["total_len"] for b in batch])
    out["mode"] = [b["mode"] for b in batch]
    return out


def load_env_constants(path=os.path.join(ROOT, "env_constants.npz")):
    z = np.load(path)
    return {k: z[k] for k in z.files}
