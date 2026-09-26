"""Training / test sequences for the intent VAE (MIND §4.2-4.3; docs/07 §20).

The VAE must encode the three kinds of 16-frame state sequences MIND feeds its encoder, in exactly the
canonical frame and normalisation our policy uses (so the latents can later be computed from policy windows):
  hist : history  S_h = frames t-15 .. t      (MIND: h = 16)
  fut  : immediate S_I = frames t+1 .. t+16   (MIND: L = 16)
         both canonicalised on frame t, the newest history frame, as in PhysWindowDataset (改动 3b);
         in training, the policy's start-rest (p 0.1) / neutral-pose (p 0.05) augmentation is reused so that
         the standing history at the start of every closed-loop rollout is in-distribution.
  holi : holistic S_H = the whole clip uniformly downsampled to 16 frames (MIND appendix A), canonicalised on
         the clip's first frame; tokens are computed on the full clip first, so velocities stay instantaneous.
One item = (hist, fut, holi) of one window; holi belongs to that window's clip.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from hml_phys import tokens as tk
from hml_phys.dataset import PhysWindowDataset, TokenStats
from hml_phys.intent_vae import VAE_STATE_DIM


def state_from_tokens(root_n, body_n):
    """normalised root [T,15] + body [T,420] -> VAE state [T,366] (drop the 69 action channels)."""
    return np.concatenate([root_n, body_n[:, :tk.STATE_DIM]], -1).astype(np.float32)


def holistic_rows(n_frames, L=16):
    return np.round(np.linspace(0, n_frames - 1, L)).astype(np.int64)


def holistic_crop_rows(n_frames, L, rng, p_exact=0.25, max_trim=0.25):
    """VAE v2 augmentation of the holistic sequence (docs/07 §20.1). With prob p_exact the exact test-time sampling
    (whole clip, linspace). Otherwise a random sub-span [a, b) that trims at most max_trim of the clip at each end
    (so it covers >= 50%), sampled at 16 evenly spaced positions with a random phase of +-half a spacing.
    Returns absolute, non-decreasing frame indices."""
    if rng.rand() < p_exact or n_frames < 2 * L:
        return holistic_rows(n_frames, L)
    a = rng.randint(0, int(n_frames * max_trim) + 1)
    b = n_frames - rng.randint(0, int(n_frames * max_trim) + 1)
    s = (b - 1 - a) / (L - 1)
    rows = np.round(a + np.arange(L) * s + (rng.rand() - 0.5) * s)
    return np.maximum.accumulate(np.clip(rows, a, b - 1)).astype(np.int64)


class IntentSeqDataset(Dataset):
    def __init__(self, split, stats_path, env_constants, train, L=16, max_clips=0, seed=0, p_rest=0.1, p_neutral=0.05,
                 holi_aug=False, p_holi_exact=0.25):
        self.L, self.train = L, train
        self.holi_aug, self.p_holi_exact = bool(holi_aug) and train, p_holi_exact
        # the policy dataset with a [16 dense | 16 future] layout and no sparse history provides the window index,
        # the rest/neutral augmentation and the canonicalisation, unchanged
        self.base = PhysWindowDataset(split, H=L, F=L, stats_path=None, text_cache=None, env_constants=env_constants,
                                      p_rest=p_rest, p_neutral=p_neutral, train=train, max_clips=max_clips, seed=seed,
                                      H_sparse=0, L_max=L, randomize_history=False)
        self.stats = TokenStats(stats_path)
        c = self.base.clips
        self.holi = np.zeros((len(c["n_frames"]), L, VAE_STATE_DIM), np.float32)
        for i, n in enumerate(c["n_frames"]):
            root, body = tk.window_tokens(c["body_pos"][i], c["dof_state"][i], c["root_state"][i], c["action"][i], origin=0)
            rows = holistic_rows(int(n), L)
            self.holi[i] = state_from_tokens(*self.stats.norm(root[rows], body[rows]))

    def __len__(self):
        return len(self.base.windows)

    def __getitem__(self, i):
        b = self.base
        clip, s = map(int, b.windows[i])
        rng = np.random.RandomState((b.rng.randint(1 << 30) + i) % (1 << 31)) if self.train else np.random.RandomState(i)
        mode = "normal"
        if self.train and b.env is not None:          # same draw as PhysWindowDataset.__getitem__
            u = rng.rand()
            if u < b.p_neutral:
                if b.rest_start[clip]:
                    mode, s = "neutral", 0
            elif u < b.p_neutral + b.p_rest:
                mode, s = "rest", 0
        bp, ds, rs, ac, rows, _, n_hist, n_used = b.raw_window(clip, s, mode, 0, 0.0, rng)
        assert n_hist == self.L and n_used == 2 * self.L, (n_hist, n_used)
        origin = int(rows[n_hist - 1])
        root, body = tk.window_tokens(bp, ds, rs, ac, origin=origin)
        x = state_from_tokens(*self.stats.norm(root[rows], body[rows]))
        if self.holi_aug:
            # tokens on the contiguous span from the first sampled frame (canonical origin, as for the whole clip)
            c = b.clips
            hr = holistic_crop_rows(int(c["n_frames"][clip]), self.L, rng, self.p_holi_exact)
            sp = slice(int(hr[0]), int(hr[-1]) + 1)
            r_, b_ = tk.window_tokens(c["body_pos"][clip][sp], c["dof_state"][clip][sp], c["root_state"][clip][sp],
                                      c["action"][clip][sp], origin=0)
            holi = state_from_tokens(*self.stats.norm(r_[hr - hr[0]], b_[hr - hr[0]]))
        else:
            holi = self.holi[clip]
        return dict(hist=torch.from_numpy(x[:self.L]), fut=torch.from_numpy(x[self.L:]),
                    holi=torch.from_numpy(holi), clip=clip, mode=mode)


LOCAL_VEL = slice(tk.ROOT_DIM + tk.BODY_SLICES["local_vel"][0], tk.ROOT_DIM + tk.BODY_SLICES["local_vel"][1])   # 87:159


def vae_history_input(hist):
    """The VAE was trained on history sequences whose token span STARTS at the first history frame, so that frame's
    local velocity is the tokeniser's edge copy of frame 1's. Policy windows / closed-loop buffers usually have earlier
    frames and hence a true velocity there. Reproduce the training input exactly before encoding (docs/07 §21.7 实施记录).
    Works on numpy or torch, shape [..., 16, 366]."""
    out = hist.clone() if hasattr(hist, "clone") else hist.copy()
    out[..., 0, LOCAL_VEL] = out[..., 1, LOCAL_VEL]
    return out
