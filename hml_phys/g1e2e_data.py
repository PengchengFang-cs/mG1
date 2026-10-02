"""Dataset for the end-to-end G1 line: text + proprio history -> the tracker's action.

Reads what `scripts/g1e2e_record_rollouts.py` wrote: per clip, 50 Hz (proprio 51, action 21) from
FRoM-W1's G1 student tracking a HumanML3D reference, plus that clip's HumanML3D captions.

    proprio 51 = base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | dof_pos(21) | dof_vel(21)
    action  21   target = default_dof_pos + 0.25 * action
    token   72 = proprio | action

The generation rate is a parameter, not a property of the data. Recording is at the control rate, 50 Hz,
and `gen_hz` subsamples by an INTEGER stride, so only the divisors of 50 are reachable: 50, 25, 10, 5, 2, 1.
20 Hz is not among them (50/20 = 2.5); reaching exactly 20 would mean interpolating the action stream, a
different operation that is deliberately not done here. The default is 25 Hz, the closest divisor: MIND's
16 history frames then cover 0.64 s instead of 0.32 s at 50 Hz, which is the difference between seeing a
twitch and seeing a motion. The actions are PD setpoints, so a policy that decides at 25 Hz and
interpolates up to the robot's 50 is sound. What a lower decision rate may cost is the stabilising
high-frequency part of the tracker's own feedback, which is why the rate is a parameter to measure.

Normalisation is per channel over the training split, computed once and cached: the channels are in wildly
different units (gravity is order 1, joint velocity order 30), so an unnormalised token would let dof_vel
dominate every loss. Statistics come from TRAIN ONLY and are reused for test, never recomputed per split.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

PROPRIO_DIM = 51
ACTION_DIM = 21
TOKEN_DIM = PROPRIO_DIM + ACTION_DIM
CONTROL_HZ = 50
# channel slices inside proprio, for per-group diagnostics
SLICES = dict(lin_vel=slice(0, 3), ang_vel=slice(3, 6), gravity=slice(6, 9),
              dof_pos=slice(9, 30), dof_vel=slice(30, 51))


def compute_stats(rollouts, gen_stride):
    """Per-channel mean/std of the 72-d token over the given clips, at the generation stride."""
    acc_n, acc_s, acc_ss = 0, np.zeros(TOKEN_DIM, np.float64), np.zeros(TOKEN_DIM, np.float64)
    for v in rollouts.values():
        tok = np.concatenate([v["proprio"], v["action"]], -1)[::gen_stride].astype(np.float64)
        acc_n += tok.shape[0]
        acc_s += tok.sum(0)
        acc_ss += (tok ** 2).sum(0)
    mean = acc_s / acc_n
    var = np.maximum(acc_ss / acc_n - mean ** 2, 0.0)
    std = np.sqrt(var)
    # A dead channel would otherwise blow up on division; 1e-3 is well below any live channel's spread.
    std = np.where(std < 1e-3, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), int(acc_n)


class G1E2EWindows(Dataset):
    """Fixed-length windows of [history | future] tokens with the clip's caption embedding.

    A window's future rows carry the actions to predict. Their proprio channels are NOT handed to the
    model: in the closed loop those states do not exist yet, and because a token holds state as well as
    action, leaving them in lets the policy read off the answer -- measured at 80x on the loss in an
    earlier G1 attempt. `mask_future_proprio` is the only place that is allowed to decide what the future
    rows show, and it defaults to showing just the first one, the state the loop can actually observe.
    """

    def __init__(self, rollouts_path, text_emb_path, H, F, gen_hz=25, stride=1, stats=None,
                 obs_future="first", max_clips=0, seed=0):
        assert CONTROL_HZ % gen_hz == 0, (
            f"gen_hz must divide the {CONTROL_HZ} Hz recording rate by an integer stride; {gen_hz} does "
            f"not. Available: {[h for h in range(1, CONTROL_HZ + 1) if CONTROL_HZ % h == 0]}. "
            f"20 Hz in particular is NOT reachable this way (50/20 = 2.5) -- it would need interpolation "
            f"of the action stream rather than subsampling, which is a different thing and not done here.")
        self.gen_stride = CONTROL_HZ // gen_hz
        self.H, self.F, self.gen_hz = H, F, gen_hz
        self.obs_future = obs_future

        roll = joblib.load(rollouts_path)
        if max_clips:
            keys = sorted(roll)[:max_clips]
            roll = {k: roll[k] for k in keys}
        self.emb = joblib.load(text_emb_path)       # {clip_id: (n_caps, L, D) or (n_caps, D) float32}

        self.clips, self.windows = [], []
        T = H + F
        skipped = 0
        for key, v in sorted(roll.items()):
            if key not in self.emb:
                skipped += 1
                continue
            tok = np.concatenate([v["proprio"], v["action"]], -1).astype(np.float32)[::self.gen_stride]
            if tok.shape[0] < T:
                skipped += 1
                continue
            ci = len(self.clips)
            self.clips.append(dict(key=key, tok=tok, n=tok.shape[0]))
            for s in range(0, tok.shape[0] - T + 1, stride):
                self.windows.append((ci, s))
        self.n_skipped = skipped

        if stats is None:
            mean, std, _ = compute_stats(roll, self.gen_stride)
        else:
            mean, std = stats
        self.mean, self.std = mean, std
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ci, s = self.windows[i]
        c = self.clips[ci]
        x = (c["tok"][s:s + self.H + self.F] - self.mean) / self.std
        x = self.mask_future_proprio(x.copy())
        caps = self.emb[c["key"]]
        cap = caps[self.rng.integers(len(caps))] if len(caps) > 1 else caps[0]
        progress = (s + self.H) / max(1, c["n"])
        return dict(
            x=torch.from_numpy(x),
            holi=torch.from_numpy(self.holistic(ci)),
            text=torch.from_numpy(np.asarray(cap, dtype=np.float32)),
            scal=torch.tensor([progress, c["n"] / float(self.gen_hz) / 10.0], dtype=torch.float32),
            key=c["key"],
        )

    def holistic(self, ci):
        """The WHOLE clip's proprio, resampled to H rows -- MIND's holistic-intent target.

        "Holistic" means the clip the caption describes, not the current window: MIND's own
        `holistic_rows` is `linspace(0, n_frames - 1, L)` over the entire clip. Fixing the row count at H
        also keeps the encoding at the latent length the intent DiT's positional embedding is built for;
        encoding the H + F window instead gives one latent frame too many.
        """
        c = self.clips[ci]
        rows = np.round(np.linspace(0, c["n"] - 1, self.H)).astype(np.int64)
        h = (c["tok"][rows, :PROPRIO_DIM] - self.mean[:PROPRIO_DIM]) / self.std[:PROPRIO_DIM]
        return h.astype(np.float32)

    def mask_future_proprio(self, x):
        """Zero the proprio channels of the future rows the closed loop cannot observe."""
        if self.obs_future == "all":
            return x
        keep = self.H + (1 if self.obs_future == "first" else 0)
        x[keep:, :PROPRIO_DIM] = 0.0
        return x


def describe(ds):
    frames = sum(c["n"] for c in ds.clips)
    return dict(clips=len(ds.clips), windows=len(ds.windows), skipped=ds.n_skipped,
                frames=frames, minutes=frames / ds.gen_hz / 60.0, gen_hz=ds.gen_hz,
                H=ds.H, F=ds.F, obs_future=ds.obs_future)


class G1E2EStateSeq(Dataset):
    """Fixed-length proprio-only sequences, for the intent VAE.

    MIND's intent VAE encodes STATE sequences with no actions in them, so this yields the 51 proprio
    channels only. Length must be divisible by the VAE's temporal downsampling (4 with the default
    down_t=2, stride_t=2) or the decoder cannot return the sequence it was given.
    """

    def __init__(self, rollouts_path, L, gen_hz=25, stride=None, stats=None, max_clips=0, down=4):
        assert CONTROL_HZ % gen_hz == 0, (
            f"gen_hz must divide {CONTROL_HZ} by an integer stride; see G1E2EWindows for the list")
        assert L % down == 0, f"L={L} must be divisible by the VAE downsampling {down}"
        self.gen_stride = CONTROL_HZ // gen_hz
        self.L, self.gen_hz = L, gen_hz
        stride = stride or max(1, L // 2)

        roll = joblib.load(rollouts_path)
        if max_clips:
            roll = {k: roll[k] for k in sorted(roll)[:max_clips]}

        if stats is None:
            mean, std, _ = compute_stats(roll, self.gen_stride)
        else:
            mean, std = stats
        # The VAE sees proprio only, so it normalises with the proprio half -- but the FULL token
        # statistics are kept and saved with the checkpoint, because stage 2 must use exactly these
        # numbers. Letting each stage recompute them only agrees when both see identical data, which is
        # a guarantee that quietly breaks the moment one run subsets its clips.
        self.mean_full, self.std_full = mean, std
        self.mean, self.std = mean[:PROPRIO_DIM], std[:PROPRIO_DIM]

        self.clips, self.windows = [], []
        for key, v in sorted(roll.items()):
            p = v["proprio"].astype(np.float32)[::self.gen_stride]
            if p.shape[0] < L:
                continue
            ci = len(self.clips)
            self.clips.append(dict(key=key, p=p, n=p.shape[0]))
            for s in range(0, p.shape[0] - L + 1, stride):
                self.windows.append((ci, s))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ci, s = self.windows[i]
        c = self.clips[ci]
        x = (c["p"][s:s + self.L] - self.mean) / self.std
        return dict(x=torch.from_numpy(x.astype(np.float32)), key=c["key"])
