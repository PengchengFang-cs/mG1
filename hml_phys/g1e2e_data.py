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

import json
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


def load_rollouts(path):
    """One rollout file, or several comma-separated ones merged into a single dict.

    The full build is recorded in halves on the two cards (shards 0-9 and 10-19), 10.5 GB each. Merging
    them on disk would cost another 21 GB and a transient doubling while writing, for no benefit: the
    clip keys are disjoint by construction, so loading both and updating is equivalent and free.
    Measured: 72800 clips load in 15 s at 10.1 GB RSS, so both halves sit at ~21 GB -- well inside the
    job's 200 GB, and the DataLoader's forked workers share the arrays copy-on-write.
    """
    parts = [p for p in str(path).split(",") if p]
    roll = {}
    for p in parts:
        d = joblib.load(p)
        dup = set(d) & set(roll)
        assert not dup, f"{p} overlaps {len(dup)} clip keys with an earlier file, e.g. {sorted(dup)[:3]}"
        roll.update(d)
        del d
    if len(parts) > 1:
        print(f"rollouts: {len(parts)} files, {len(roll)} clips total")
    return roll


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
    # PROPRIO channels only. A dead proprio channel would otherwise blow up on division, and the
    # threshold has to sit ABOVE the smallest genuinely-dead one: projected_gravity[2] measures std
    # 0.00423 on the recorded data (the robot is upright almost always, so that component is pinned near
    # -1), and a 1e-3 guard let it through to be amplified ~236x into the network input.
    # The 21 ACTION channels are deliberately excluded: they are the loss target, and forcing an
    # under-0.01 std to 1.0 would down-weight that joint's squared error by up to 1e4 while making the
    # shadow NMSE denominator ~0 for it, so neither the loss nor the metric could see it. A genuinely
    # constant action channel should fail loudly instead.
    std[:PROPRIO_DIM] = np.where(std[:PROPRIO_DIM] < 1e-2, 1.0, std[:PROPRIO_DIM])
    bad = np.nonzero(std[PROPRIO_DIM:] < 1e-4)[0]
    assert bad.size == 0, (
        f"action channels {bad.tolist()} have std < 1e-4 over the recorded rollouts; a constant action "
        f"channel means the tracker never moved that joint, which is a data problem, not something to "
        f"paper over with a normalisation floor")
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
                 obs_future="first", max_clips=0, seed=0, deterministic_caption=False,
                 p_rest=0.1, holi_aug=True, p_holi_exact=0.25, max_mpjpe=0.0):
        assert CONTROL_HZ % gen_hz == 0, (
            f"gen_hz must divide the {CONTROL_HZ} Hz recording rate by an integer stride; {gen_hz} does "
            f"not. Available: {[h for h in range(1, CONTROL_HZ + 1) if CONTROL_HZ % h == 0]}. "
            f"20 Hz in particular is NOT reachable this way (50/20 = 2.5) -- it would need interpolation "
            f"of the action stream rather than subsampling, which is a different thing and not done here.")
        self.gen_stride = CONTROL_HZ // gen_hz
        self.H, self.F, self.gen_hz = H, F, gen_hz
        self.obs_future = obs_future
        self.p_rest = float(p_rest)
        self.holi_aug = bool(holi_aug)
        self.p_holi_exact = float(p_holi_exact)
        self.seed = int(seed)

        roll = load_rollouts(rollouts_path)
        # The hold action -- the action whose PD target is the robot's CURRENT joint angles -- needs the
        # env's default_dof_pos, which only the recorder has. `tokens.hold_action` is the reference's
        # version of the same quantity.
        # With several rollout files, take the meta of the FIRST: default_dof_pos and action_scale come
        # from the env and are identical across parts recorded by the same tracker under the same config.
        meta_p = Path(str(rollouts_path).split(",")[0].replace(".pkl", ".meta.json"))
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        self.default_dof = (np.asarray(meta["default_dof_pos"], dtype=np.float32)
                            if "default_dof_pos" in meta else None)
        self.action_scale = float(meta.get("action_scale", 0.25))
        if self.p_rest > 0 and self.default_dof is None:
            raise SystemExit(
                f"--p-rest needs default_dof_pos in {meta_p.name}, which this rollout file predates. "
                f"Re-record (scripts/g1e2e_record_rollouts.py now stores it) or pass --p-rest 0.")
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
        n_mpjpe_drop = 0
        for key, v in sorted(roll.items()):
            # With --repeats > 1 the recorder keys a clip's pass r > 0 as "<clip>#r", so the captions and
            # the reference library are keyed by `base_key`, not by the dict key. Falling back to the
            # dict key keeps rollout files recorded before repeats existed working unchanged.
            base = v.get("base_key", key)
            if base not in self.emb:
                skipped += 1
                continue
            # The recorder's termination filter only removed the clips where the tracker FELL (0.5 m MEAN
            # body error, measured: 140/2000 = 7.0%, the tracker's own fall rate). A clip the robot
            # followed at 0.49 m average still carries its caption as a label for whatever the robot
            # actually did. `mpjpe_mean_m` was stored so this could be tightened and had no consumer.
            if max_mpjpe > 0 and float(v.get("mpjpe_mean_m", 0.0)) > max_mpjpe:
                n_mpjpe_drop += 1
                continue
            tok = np.concatenate([v["proprio"], v["action"]], -1).astype(np.float32)[::self.gen_stride]
            if tok.shape[0] < T:
                skipped += 1
                continue
            ci = len(self.clips)
            # `key` is the per-pass id (unique, used for reporting); `base` is the caption/library id.
            self.clips.append(dict(key=key, base=base, tok=tok, n=tok.shape[0]))
            # `s` runs to n - (H + F): the tail is reachable because _window_rows clamps. The IIP target
            # then repeats the clip's last state row, which is what the reference does past its end.
            for s in range(0, tok.shape[0] - (H + F) + 1, stride):
                self.windows.append((ci, s))
            if self.p_rest > 0:
                # One start-rest window per clip, marked by s = -1. The closed loop hands over from a
                # standing robot with a hold action, a history the recorded data never contains (every
                # recorded window has 16 distinct moving frames), so the very first plan of every episode
                # was off-distribution. The reference builds these deliberately (dataset.py:19-21).
                for _ in range(max(1, int(round(self.p_rest * len(self.windows) / max(1, len(self.clips)))))):
                    self.windows.append((ci, -1))
        self.n_skipped, self.n_mpjpe_drop = skipped, n_mpjpe_drop

        if stats is None:
            mean, std, _ = compute_stats(roll, self.gen_stride)
        else:
            mean, std = stats
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.windows)

    def _window_rows(self, c, s):
        """The H + F_fut token rows of a window, clamping the tail to the clip's last row.

        Without the clamp a window had to fit H + F_fut = 32 rows, so `s <= n - 32` and the last 12
        generation rows of every clip (0.48 s at 25 Hz) were never a target -- while `progress` only ever
        spanned [H/n, (n-H)/n] in training and the closed loop feeds it 0.0 and 1.0. Repeating the last
        row is the right clamp: it is what the motion library itself does past the end of a reference.
        """
        idx = np.minimum(np.arange(s, s + self.H + self.F_fut), c["n"] - 1)
        return c["tok"][idx]

    def __getitem__(self, i):
        ci, s = self.windows[i]
        c = self.clips[ci]
        if s < 0:                      # start-rest augmentation, see _rest_window
            return self._rest_window(ci)
        full = (self._window_rows(c, s) - self.mean) / self.std
        # The immediate-intent target: the next H proprio rows, UNMASKED. It reaches the frozen VAE only,
        # never the policy -- the policy sees `x` below, where the future proprio is zeroed.
        fut = full[self.H:self.H + self.H, :PROPRIO_DIM].copy()
        x = self.mask_future_proprio(full[:self.H + self.F].copy())
        caps = self.emb[c["base"]]
        # Deterministic per-window choice, not a shared RNG: a generator built in __init__ is inherited
        # identically by every forked worker, so all of them drew the same caption sequence and the
        # paraphrase augmentation was a quarter as diverse as intended. Deriving the index from the
        # window also makes the eval split reproducible, which checkpoint selection depends on.
        j = 0 if self.deterministic_caption else (hash((c["base"], s)) % len(caps))
        progress = (s + self.H) / max(1, c["n"])
        return dict(
            x=torch.from_numpy(x),
            fut=torch.from_numpy(fut.astype(np.float32)),
            holi=torch.from_numpy(self.holistic(ci, s)),
            text=torch.from_numpy(np.asarray(caps[j], dtype=np.float32)),
            text_len=torch.tensor(int(self.lens[c["base"]][j]), dtype=torch.long),
            text_pooled=torch.from_numpy(np.asarray(self.pooled[c["base"]][j], dtype=np.float32)),
            scal=torch.tensor([progress, c["n"] / float(self.gen_hz) / 10.0], dtype=torch.float32),
            key=c["key"],
        )

    def _rest_window(self, ci):
        """A window whose history is the clip's first frame held still, with the hold action.

        The closed loop starts from a standing robot: 16 identical state rows, zero velocities, and the
        action whose PD target is the pose it is already in. The recorded data contains no such window --
        every one of them has 16 distinct moving frames -- so the first plan of every episode was made
        off-distribution. The future rows are the clip's real opening, so the label is "start this motion
        from rest", which is exactly the transition the loop has to make.
        """
        c = self.clips[ci]
        row = c["tok"][0].copy()
        row[SLICES["lin_vel"]] = 0.0
        row[SLICES["ang_vel"]] = 0.0
        row[SLICES["dof_vel"]] = 0.0
        row[PROPRIO_DIM:] = (row[SLICES["dof_pos"]] - self.default_dof) / self.action_scale
        full = np.concatenate([np.repeat(row[None], self.H, 0),
                               self._window_rows(c, 0)[:self.F_fut]], 0)
        full = (full - self.mean) / self.std
        fut = full[self.H:self.H + self.H, :PROPRIO_DIM].copy()
        x = self.mask_future_proprio(full[:self.H + self.F].copy())
        caps = self.emb[c["base"]]
        j = 0 if self.deterministic_caption else (hash((c["base"], -1)) % len(caps))
        return dict(
            x=torch.from_numpy(x.astype(np.float32)),
            fut=torch.from_numpy(fut.astype(np.float32)),
            holi=torch.from_numpy(self.holistic(ci, -1)),
            text=torch.from_numpy(np.asarray(caps[j], dtype=np.float32)),
            text_len=torch.tensor(int(self.lens[c["base"]][j]), dtype=torch.long),
            text_pooled=torch.from_numpy(np.asarray(self.pooled[c["base"]][j], dtype=np.float32)),
            scal=torch.tensor([0.0, c["n"] / float(self.gen_hz) / 10.0], dtype=torch.float32),
            key=c["key"],
        )

    def holistic(self, ci, s=0):
        """The WHOLE clip's proprio, resampled to H rows -- MIND's holistic-intent target.

        "Holistic" means the clip the caption describes, not the current window: MIND's own
        `holistic_rows` is `linspace(0, n_frames - 1, L)` over the entire clip. Fixing the row count at H
        also keeps the encoding at the latent length the intent DiT's positional embedding is built for;
        encoding the H + F window instead gives one latent frame too many.

        With `holi_aug` the span is augmented exactly as the reference does
        (`hml_phys/intent_data.holistic_crop_rows`): with probability `p_holi_exact` the exact test-time
        sampling, otherwise a random sub-span covering at least half the clip, sampled at H evenly spaced
        positions with a random phase. Without it there is exactly ONE holistic target per clip, shared by
        every window and every caption of that clip -- 1816 training examples for HIP, which is why HIP
        went to a test loss of 5.26 against unit-variance targets while its train loss was 0.34.
        """
        c = self.clips[ci]
        if self.holi_aug:
            from hml_phys.intent_data import holistic_crop_rows
            # Seeded from the window, not from a shared RandomState: a generator built in __init__ is
            # inherited identically by every forked DataLoader worker, so all of them would draw the
            # same span sequence -- the same mistake the caption choice above had to fix.
            rng = np.random.RandomState((hash((ci, s)) ^ self.seed) & 0x7FFFFFFF)
            rows = holistic_crop_rows(c["n"], self.H, rng, p_exact=self.p_holi_exact)
        else:
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
    return dict(clips=len(ds.clips), windows=len(ds.windows), skipped=ds.n_skipped, mpjpe_dropped=ds.n_mpjpe_drop, rest_windows=sum(1 for _, s in ds.windows if s < 0),
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

        roll = load_rollouts(rollouts_path)
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
            # ONE holistic item per clip, i.e. ~1/22 of stage 1. The reference emits (hist, fut, holi)
            # per item and concatenates all three into every batch (train_intent_vae.py:73,126), a 1:1:1
            # mix, and a code review on 2026-10-02 noted this as a deviation -- but in the "impact
            # unestablished" bucket, not as a defect: this VAE was healthy at 1/22 (test MSE 0.04137,
            # 32/32 latent dims active).
            # Raising it to 1:1 was tried on 2026-10-03 and must not be read as a regression OR an
            # improvement: holistic sequences span the whole clip at ~0.5 s per row and are far harder to
            # reconstruct than a contiguous 0.64 s window, so putting them at ~50% of BOTH splits changed
            # what the metric measures (best test MSE 0.24660 against 0.04137 on 78308/40591 windows).
            # The two numbers are not comparable, and deciding between the mixtures would need a
            # controlled comparison that nothing here calls for -- the defects the reviews actually
            # located are all in stage 2 and inference. Left at the known-good value.
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
