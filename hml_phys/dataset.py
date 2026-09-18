"""Sliding-window dataset for the physics policy (docs/07 §1, §1.5, §2, §11).

Frame convention (from the PHC recorder): row t = (state AFTER step t, action applied AT step t), i.e. the action that
produced this row's state; the closed-loop buffer (mc_rollout.HistoryBuffer) pushes exactly the same pairing.
v3 (docs/07 §15): the window is [sparse distant history | dense recent history | future]:
  * dense history  = the H_dense most recent frames, kept as they are;
  * sparse history = up to H_sparse frames drawn from the preceding L_max - H_dense frames with SCRIPT's
    exponential bias towards the recent end (tokens.sample_sparse_history); unused slots are marked invalid;
  * future         = F frames (or, with whole_sequence, to the clip end, capped at F).
Tokens are always computed on the CONTIGUOUS span first and the selected rows are gathered afterwards, so
every row keeps its true instantaneous velocities. Each row carries a signed `frame_index` (0 = first
generated frame, history negative, true frame offsets) which drives the positional encoding and the gaps
used by the local-root bridge. The window is canonicalised on the newest history frame (改动 3b).
Each sample = window of T = H_sparse + H_dense + F rows from one tracked clip:
  root  [T,15], body [T,420]  (normalised tokens, window-canonical frame; see hml_phys.tokens)
  observed_mask [T]  1 for history rows, 0 for future rows
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
    """per-dim mean/std for root, body and local-root features (MotionCraft-style std = sqrt(var + eps))."""
    def __init__(self, path):
        z = np.load(path)
        self.root_mean, self.root_std = z["root_mean"], z["root_std"]
        self.body_mean, self.body_std = z["body_mean"], z["body_std"]
        self.local_root_mean = z["local_root_mean"] if "local_root_mean" in z else np.zeros(4, np.float32)
        self.local_root_std = z["local_root_std"] if "local_root_std" in z else np.ones(4, np.float32)

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
                 whole_sequence=False, F_min=8, H_sparse=16, L_max=154, alpha=3.0, randomize_history=True,
                 p_no_sparse=0.15, alpha_range=(1.0, 5.0)):
        """H        : dense recent history frames (kept verbatim)
        H_sparse    : slots for the sparse distant history (0 disables the long history entirely)
        L_max       : total history span in frames that the sparse part may reach back over
        alpha       : SCRIPT's exponential bias (0 = uniform, larger = more recent-biased)
        randomize_history: during training draw H_sparse and alpha per sample so that history length is a
                    test-time knob; p_no_sparse is the probability of drawing no sparse history at all.
        whole_sequence=True (v2, kept for reference): future runs to the clip end, capped at F frames."""
        self.H, self.F = H, F
        self.H_sparse, self.L_max, self.alpha = H_sparse, L_max, alpha
        self.randomize_history = bool(randomize_history) and train
        self.p_no_sparse, self.alpha_range = p_no_sparse, alpha_range
        self.T = H_sparse + H + F
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
            need = self.H + (self.F_min if self.whole_sequence else self.F)
            if T >= need:
                # s = index of the first DENSE history frame; the sparse part reaches further back when it can
                idx += [(i, s) for s in range(0, T - need + 1, stride)]
        self.windows = np.array(idx, dtype=np.int64)
        self.n_clips_short = int(sum(1 for T in d["n_frames"] if T < (self.H + (self.F_min if self.whole_sequence else self.F))))
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

    def draw_history_cfg(self, rng):
        """per-sample (n_sparse, alpha); constant outside training."""
        if not self.randomize_history:
            return self.H_sparse, self.alpha
        if rng.rand() < self.p_no_sparse:
            return 0, self.alpha
        n = int(rng.randint(1, self.H_sparse + 1)) if self.H_sparse > 0 else 0
        a = float(rng.uniform(*self.alpha_range))
        return n, a

    # ---- text
    def pick_text(self, clip, s, rng):
        fps = float(self.clips["fps_eff"][clip])
        f0, f1 = (s + self.H) / fps, (s + self.H + self.future_len(clip, max(s, 0))) / fps  # future span in original seconds
        cand = [ci for ci, a, b in self.cands[clip] if (a == 0.0 and b == 0.0) or (a < f1 and b > f0)]
        cand = [ci for ci in cand if ci >= 0]
        return int(cand[rng.randint(len(cand))]) if cand else -1

    # ---- raw window with augmentations
    def raw_window(self, clip, s, mode, n_sparse, alpha, rng):
        """Assemble one window.

        Returns (bp, ds, rs, ac, frame_index, n_hist, n_used) where the arrays are the CONTIGUOUS span that
        the tokens must be computed on, `frame_index` are the signed offsets of the rows to gather out of
        that span (0 = first generated frame), `n_hist` the number of history rows and `n_used` the number
        of valid rows. Rows are gathered by the caller after tokenisation, so every row keeps its true
        instantaneous velocities.

        s = index of the first DENSE history frame. The sparse part reaches back over the preceding
        min(L_max - H, s) frames; in the rest/neutral modes the history is synthetic and there is no
        sparse part (the clip has not started yet).
        """
        c = self.clips
        if mode in ("rest", "neutral"):
            Fu = min(self.F, int(c["n_frames"][clip]))
            fut = slice(0, Fu)
            bp_f, ds_f, rs_f, ac_f = (c["body_pos"][clip][fut], c["dof_state"][clip][fut],
                                      c["root_state"][clip][fut], c["action"][clip][fut])
            if mode == "rest":
                bp0, dof0, rs0 = bp_f[0], ds_f[0, :, 0], rs_f[0].copy()
            else:
                # neutral standing history: rotate the clip's future about z so that its frame-0 heading (hip-across
                # direction) matches the neutral pose, and place the neutral pose at the clip's start xy
                bp0, dof0, rs0 = self.env["neutral_body_pos"].copy(), self.env["neutral_dof_pos"], self.env["neutral_root_state"].copy()
                yaw_n = _hip_yaw(bp0); yaw_c = _hip_yaw(bp_f[0])
                bp_f, rs_f = _rotate_z(bp_f, rs_f, yaw_n - yaw_c, center=bp_f[0, 0, :2])
                shift = bp_f[0, 0, :2] - bp0[0, :2]
                bp0 = bp0.copy(); bp0[:, :2] += shift; rs0[:2] += shift
            rs0 = rs0.copy(); rs0[7:13] = 0.0
            hold = tk.hold_action(dof0, self.env["pd_offset"], self.env["pd_scale"]).astype(np.float32)
            bp = np.concatenate([np.repeat(bp0[None], self.H, 0), bp_f], 0)
            ds_h = np.zeros((self.H,) + ds_f.shape[1:], ds_f.dtype); ds_h[:, :, 0] = dof0
            ds = np.concatenate([ds_h, ds_f], 0)
            rs = np.concatenate([np.repeat(rs0[None], self.H, 0), rs_f], 0)
            ac = np.concatenate([np.repeat(hold[None], self.H, 0), ac_f], 0)
            rows = np.arange(self.H + Fu)
            frame_index = rows - self.H
            return bp, ds, rs, ac, rows, frame_index, self.H, self.H + Fu

        Fu = self.future_len(clip, s)
        l_distant = int(min(self.L_max - self.H, s))          # frames available before the dense history
        span0 = s - l_distant                                  # first frame of the contiguous span
        span1 = s + self.H + Fu                                # one past the last
        bp = c["body_pos"][clip][span0:span1]
        ds = c["dof_state"][clip][span0:span1]
        rs = c["root_state"][clip][span0:span1]
        ac = c["action"][clip][span0:span1]
        sparse = tk.sample_sparse_history(l_distant, n_sparse, alpha, rng)   # indices into [span0, s)
        dense = np.arange(l_distant, l_distant + self.H)
        future = np.arange(l_distant + self.H, l_distant + self.H + Fu)
        rows = np.concatenate([sparse, dense, future]).astype(np.int64)
        frame_index = rows - (l_distant + self.H)              # 0 = first generated frame, history negative
        return bp, ds, rs, ac, rows, frame_index, len(sparse) + self.H, len(rows)

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
        n_sparse, alpha = self.draw_history_cfg(rng)
        bp, ds, rs, ac, rows, frame_index, n_hist, n_used = self.raw_window(clip, s, mode, n_sparse, alpha, rng)
        # tokens on the contiguous span (true instantaneous velocities), canonicalised on the newest history
        # frame (改动 3b), then gather the selected rows
        origin = int(rows[n_hist - 1])
        root_full, body_full = tk.window_tokens(bp, ds, rs, ac, origin=origin)
        root, body = root_full[rows], body_full[rows]
        if self.stats is not None:
            root, body = self.stats.norm(root, body)
        if self.sigma_hist > 0:
            root[:n_hist] += rng.randn(n_hist, root.shape[1]).astype(np.float32) * self.sigma_hist
            body[:n_hist] += rng.randn(n_hist, body.shape[1]).astype(np.float32) * self.sigma_hist
        # pad to the fixed layout [H_sparse | H | F]; unused sparse slots are invalid and masked out
        pad = self.T - n_used
        fidx = np.asarray(frame_index, np.int64)
        valid = np.ones(n_used, np.float32)
        mask = np.zeros(n_used, np.float32); mask[:n_hist] = 1.0
        if pad > 0:
            z = np.zeros((pad, root.shape[1]), np.float32); root = np.concatenate([z, root], 0)
            z = np.zeros((pad, body.shape[1]), np.float32); body = np.concatenate([z, body], 0)
            valid = np.concatenate([np.zeros(pad, np.float32), valid], 0)
            mask = np.concatenate([np.ones(pad, np.float32), mask], 0)      # padded rows count as observed
            fidx = np.concatenate([np.full(pad, fidx[0], np.int64), fidx], 0)
        T_clip = int(self.clips["n_frames"][clip]); fps = float(self.clips["fps_eff"][clip])
        now = 0 if mode != "normal" else s + self.H     # clip frames already executed at the replan point
        progress = now / max(1, T_clip)
        total_len = T_clip / fps
        ti = self.pick_text(clip, s if mode == "normal" else -self.H, rng)  # augmented: future = clip frames [0:F)
        return dict(root=torch.from_numpy(root.astype(np.float32)), body=torch.from_numpy(body.astype(np.float32)),
                    observed_mask=torch.from_numpy(mask), valid=torch.from_numpy(valid),
                    frame_index=torch.from_numpy(fidx), text_idx=ti, progress=np.float32(progress),
                    total_len=np.float32(total_len), clip=clip, start=s, mode=mode, n_frames=n_used,
                    n_hist=pad + n_hist,                       # index of the first future row in the PADDED layout
                    n_sparse=int(n_hist - self.H))             # real sparse rows (excludes the padded slots)


def collate(batch, text_cache):
    keys = ["root", "body", "observed_mask", "valid", "frame_index"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    Tmax = int(max(b["n_frames"] for b in batch))  # padding sits at the FRONT, so trim from the left
    T = out["root"].shape[1]
    if Tmax < T:
        for k in keys:
            out[k] = out[k][:, T - Tmax:].contiguous()
    empty = text_cache.index.get("", -1)  # windows without caption use the CLIP("") features (same as dropout / CFG)
    toks, pooled, lens = zip(*[text_cache.get(b["text_idx"] if b["text_idx"] >= 0 else empty) for b in batch])
    out["text_tokens"] = torch.from_numpy(np.stack(toks)); out["text_pooled"] = torch.from_numpy(np.stack(pooled))
    out["text_len"] = torch.tensor(lens); out["text_dropped"] = torch.tensor([b["text_idx"] < 0 for b in batch])
    out["progress"] = torch.tensor([b["progress"] for b in batch]); out["total_len"] = torch.tensor([b["total_len"] for b in batch])
    out["mode"] = [b["mode"] for b in batch]
    out["n_sparse"] = torch.tensor([b["n_sparse"] for b in batch])
    out["n_hist"] = torch.tensor([b["n_hist"] for b in batch]) - (T - Tmax)  # adjust for the left trim
    return out


def load_env_constants(path=os.path.join(ROOT, "env_constants.npz")):
    z = np.load(path)
    return {k: z[k] for k in z.files}
