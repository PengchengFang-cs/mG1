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
    # A dead channel would otherwise blow up on division. The threshold has to sit ABOVE the smallest
    # genuinely-dead channel: projected_gravity[2] measures std 0.00423 on the recorded data (the robot is
    # upright almost always, so that component is pinned near -1), and a 1e-3 guard let it through to be
    # amplified ~236x into the network input and weighted equally in the VAE's reconstruction objective.
    std = np.where(std < 1e-2, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), int(acc_n)


class G1E2EWindows(Dataset):
    """Fixed-length windows of [history | future] tokens with the clip's caption embedding.

    A window's future rows carry the actions to predict. Their proprio channels are NOT handed to the
    model: in the closed loop those states do not exist yet, and because a token holds state as well as
    action, leaving them in lets the policy read off the answer -- measured at 80x on the loss in an
    earlier G1 attempt. `mask_future_proprio` is the only place that is allowed to decide what the future
    rows show, and it defaults to showing just the first one, the state the loop can actually observe.
    """

    def __init__(self, rollouts_path, text_cache, H, F, gen_hz=25, stride=1, stats=None,
                 obs_future="first", max_clips=0, seed=0, deterministic_caption=False):
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
        # Tokens, their per-caption LENGTHS and the CLIP pooled vectors all travel together: a caption's
        # mask is only correct against its own length, and the pooled vector is the sentence embedding
        # the policy's cross-attention was designed around (text_mode="sentence_xattn" makes it the only
        # sentence-level text signal). A mean over the 50 token slots is not that vector.
        tc = Path(text_cache)
        self.emb = joblib.load(tc / "tokens.pkl")
        self.lens = joblib.load(tc / "lengths.pkl")
        self.pooled = joblib.load(tc / "pooled.pkl")
        self.deterministic_caption = deterministic_caption

        # The window index is built with H future rows, not F. MIND's immediate intent is the encoding of
        # the NEXT H state rows (intent_policy_data.py:8, `fut [16,366] future t+1..t+16 -> VAE -> I_I`),
        # so a window must always have H future rows available even though the policy only generates F of
        # them -- exactly what the validated version does with `kw["F"] = L_INTENT` and then trims the
        # policy rows back to F_act. Without this the IIP has no target distinct from its own prefix and
        # degenerates into an identity map.
        self.clips, self.windows = [], []
        self.F_fut = max(F, H)
        T = H + self.F_fut
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

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ci, s = self.windows[i]
        c = self.clips[ci]
        full = (c["tok"][s:s + self.H + self.F_fut] - self.mean) / self.std
        # The immediate-intent target: the next H proprio rows, UNMASKED. It reaches the frozen VAE only,
        # never the policy -- the policy sees `x` below, where the future proprio is zeroed.
        fut = full[self.H:self.H + self.H, :PROPRIO_DIM].copy()
        x = self.mask_future_proprio(full[:self.H + self.F].copy())
        caps = self.emb[c["key"]]
        # Deterministic per-window choice, not a shared RNG: a generator built in __init__ is inherited
        # identically by every forked worker, so all of them drew the same caption sequence and the
        # paraphrase augmentation was a quarter as diverse as intended. Deriving the index from the
        # window also makes the eval split reproducible, which checkpoint selection depends on.
        j = 0 if self.deterministic_caption else (hash((c["key"], s)) % len(caps))
        progress = (s + self.H) / max(1, c["n"])
        return dict(
            x=torch.from_numpy(x),
            fut=torch.from_numpy(fut.astype(np.float32)),
            holi=torch.from_numpy(self.holistic(ci)),
            text=torch.from_numpy(np.asarray(caps[j], dtype=np.float32)),
            text_len=torch.tensor(int(self.lens[c["key"]][j]), dtype=torch.long),
            text_pooled=torch.from_numpy(np.asarray(self.pooled[c["key"]][j], dtype=np.float32)),
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
                H=ds.H, F=ds.F, F_fut=ds.F_fut, obs_future=ds.obs_future)


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

        # Three KINDS of sequence, as the reference VAE trains on (train_intent_vae.py:73,126 uses
        # KINDS = ("hist", "fut", "holi")): contiguous slices AND the whole clip resampled to L rows.
        # Stage 2 asks this encoder for the holistic latent, whose rows are ~13 apart (about 0.5 s per
        # step) -- far outside the distribution of a contiguous slice. Without the holistic kind here,
        # HIP is trained to regress a latent the encoder was never fit to produce.
        self.clips, self.windows = [], []
        for key, v in sorted(roll.items()):
            p = v["proprio"].astype(np.float32)[::self.gen_stride]
            if p.shape[0] < L:
                continue
            ci = len(self.clips)
            self.clips.append(dict(key=key, p=p, n=p.shape[0]))
            for s in range(0, p.shape[0] - L + 1, stride):
                self.windows.append((ci, s))
            self.windows.append((ci, -1))          # -1 = the holistic resampling of this clip

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ci, s = self.windows[i]
        c = self.clips[ci]
        if s < 0:
            rows = np.round(np.linspace(0, c["n"] - 1, self.L)).astype(np.int64)
            x = (c["p"][rows] - self.mean) / self.std
        else:
            x = (c["p"][s:s + self.L] - self.mean) / self.std
        return dict(x=torch.from_numpy(x.astype(np.float32)), key=c["key"])
