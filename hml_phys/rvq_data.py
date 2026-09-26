"""Windows of normalised physics tokens for the RVQ tokeniser (docs/08); MoMask-original whole-body by default.

One item = W consecutive frames of one tracked clip, taken with stride 1 from every clip of the split
(clips shorter than W are skipped), as NORMALISED tokens restricted to one of three channel variants:

    "action" ->  69 channels : the PD actions only                 (tokens.action_channels_in_token())
    "token"  -> 435 channels : the whole physics token             (root 15 | body 420)
    "state"  -> 366 channels : everything except the actions       (root 15 | body[:351])

CANONICAL FRAME -- the window's FIRST frame.  Tokens are computed as
    tokens.window_tokens(body_pos[s:s+W], dof_state[s:s+W], root_state[s:s+W], action[s:s+W], origin=0)
i.e. on exactly the span the window covers and with the window's own frame 0 as the origin (frame 0's root xy
goes to the world origin, its hip-across direction onto +x, so the character faces +y).  A window is therefore
SELF-CONTAINED: nothing outside [s, s+W) is read, and the tokeniser sees the same thing whether the window
comes from this dataset, from a closed-loop rollout buffer or from a generated sequence.

Difference from the POLICY windows (hml_phys/dataset.py: PhysWindowDataset, docs/07 §15 改动 3b):
  * the policy canonicalises on the NEWEST HISTORY frame (origin = the last observed row) so that the frames
    it predicts sit closest to the origin; here the origin is the OLDEST frame of the window;
  * the policy computes the tokens on a CONTIGUOUS SPAN that is longer than the rows it keeps (the sparse long
    history gathers rows out of that span), so every kept row keeps a true instantaneous velocity; here the
    span IS the window;
  * a policy window is a [sparse history | dense history | future] layout of length H_sparse + H + F with
    observed / valid masks; here every row is equal and the length is a plain W.
  Consequence of tokenising the window span alone: `local_vel` of row 0 is the tokeniser's edge copy of row 1's
  (get_repr pads the finite difference at the front), exactly like the intent VAE's history input
  (intent_data.vae_history_input).  Rows 1.. carry true instantaneous velocities.  `root_trans_vel`,
  `root_rot_vel` and `dof_vel` are RECORDED velocities rather than finite differences, so they are exact on
  every row, row 0 included.

Normalisation uses the same per-dim statistics as the policy (data/humanml3d_phys/token_stats_v3.npz,
hml_phys/dataset.py: TokenStats), restricted to the variant's channels; `mean` / `std` are exposed so the
evaluation can go back to physical units.

AUGMENTATION: the policy's start-rest / neutral-pose augmentation is deliberately NOT reused.  It is not
trivial to port: it synthesises a fixed-length HISTORY prefix (H frozen copies of a rest or neutral state plus
the corresponding hold action) in front of the clip's first F frames, which only makes sense for the policy's
[history | future] layout and additionally needs env_constants.npz.  There is no history/future split here, so
RVQ windows are plain clip windows with NO augmentation at all -- `train` and `env_constants` are accepted for
signature compatibility with the policy datasets and are unused.

Split policy: "train" and "test" only -- the val split is banned project-wide (project CLAUDE.md §1).
"""
import os

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from hml_phys import tokens as tk
from hml_phys.dataset import ROOT, TokenStats

TOKEN_DIM = tk.ROOT_DIM + tk.BODY_DIM                      # 435
DEFAULT_STATS = os.path.join(ROOT, "token_stats_v3.npz")
DOWNSAMPLE = 4                                             # RVQ temporal downsampling; W must be a multiple
DEFAULT_WINDOW = 64
VARIANTS = ("action", "token", "state")
EXPLICIT_FEATURE = "local_positions"                       # MoMask's "explicit" geometric channels


def _token_feature_slices():
    """name -> (a, b) half-open channel range inside the 435-d token, in token order."""
    out = {}
    for name, (a, b) in tk.ROOT_SLICES.items():
        out[name] = (a, b)
    for name, (a, b) in tk.BODY_SLICES.items():
        out[name] = (tk.ROOT_DIM + a, tk.ROOT_DIM + b)
    return out


TOKEN_FEATURE_SLICES = _token_feature_slices()
# per-feature (physical unit, channels per entity).  "m/frame" is a per-frame displacement at 30 fps;
# "rot6d" is the first two columns of a rotation matrix; "pd" is a raw PD action (pd_tar = offset + scale * a).
FEATURE_UNITS = {
    "root_trans": ("m", 3), "root_rot_6d": ("rot6d", 6), "root_trans_vel": ("m/s", 3), "root_rot_vel": ("rad/s", 3),
    "local_positions": ("m", 3), "local_vel": ("m/frame", 3), "dof_pose_6d": ("rot6d", 6), "dof_vel": ("rad/s", 3),
    "action": ("pd", 3),
}
ACTION_CH0 = TOKEN_FEATURE_SLICES["action"][0]             # 366; token channel -> dof index = c - ACTION_CH0

_PART_CACHE = None


def _part_channels():
    global _PART_CACHE
    if _PART_CACHE is None:
        _PART_CACHE = tk.part_channels()
    return _PART_CACHE


def variant_channels(variant):
    """-> int64 indices into the 435-d token that the variant keeps (sorted)."""
    if variant == "token":
        return np.arange(TOKEN_DIM, dtype=np.int64)
    if variant == "action":
        return tk.action_channels_in_token()
    if variant == "state":
        keep = np.ones(TOKEN_DIM, bool)
        keep[tk.action_channels_in_token()] = False
        return np.flatnonzero(keep).astype(np.int64)
    raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")


def variant_dim(variant):
    return len(variant_channels(variant))


def _inverse_index(channels):
    inv = np.full(TOKEN_DIM, -1, np.int64)
    inv[channels] = np.arange(len(channels), dtype=np.int64)
    return inv


def variant_local_channels(host, sub):
    """-> int64 LOCAL indices of the `sub` variant's channels inside the `host` variant's vector.

    `variant_local_channels("token", "action")` gives the 69 action channels as offsets into the 435-d token
    vector, which is what the two-tokenizer ("码本对照") CodeFlow needs: the window is loaded ONCE as the full
    "token" variant (the observation tokenizer's input) and the generation tokenizer's channels are sliced out
    of that same tensor, so each tokenizer sees exactly the channels it was trained on and no window is read
    from disk twice.  Raises if `sub` is not a subset of `host`.
    """
    loc = _inverse_index(variant_channels(host))[variant_channels(sub)]
    assert (loc >= 0).all(), f"the {sub!r} variant is not a subset of the {host!r} variant"
    return loc.astype(np.int64)


def part_groups(variant):
    """variant -> (names, groups): the per-part channel groups as LOCAL indices into the variant's vector.

    Parts that own no channel in the variant are DROPPED, so "action" yields 5 parts (the root part owns no
    action channel -- the pelvis is not actuated) and "token" / "state" yield all 6.  The groups partition the
    variant's channels exactly, so np.concatenate(groups) is a permutation of range(variant_dim(variant))
    (what PartRVQVAE expects for its part_index / inverse_index buffers).
    """
    ch = variant_channels(variant)
    inv = _inverse_index(ch)
    names, groups = [], []
    for name, idx in zip(tk.PART_NAMES, _part_channels()):
        loc = inv[idx]
        loc = loc[loc >= 0]
        if len(loc) == 0:
            continue
        names.append(name)
        groups.append(np.sort(loc).astype(np.int64))
    total = sum(len(g) for g in groups)
    assert total == len(ch), f"part groups must partition the {variant} variant, got {total} / {len(ch)}"
    return names, groups


def part_names(variant):
    return part_groups(variant)[0]


def part_dims(variant):
    return [len(g) for g in part_groups(variant)[1]]


def feature_groups(variant):
    """variant -> {feature name: LOCAL indices into the variant's vector}, token order, empty groups dropped."""
    ch = variant_channels(variant)
    inv = _inverse_index(ch)
    out = {}
    for name, (a, b) in TOKEN_FEATURE_SLICES.items():
        loc = inv[np.arange(a, b)]
        loc = loc[loc >= 0]
        if len(loc):
            out[name] = loc.astype(np.int64)
    return out


def variant_parts(variant):
    """The trainer-facing helper (scripts/hml_phys/train_rvq.py).

    -> (parts, explicit_idx)
       parts        : list of int64 arrays, the LOCAL channel indices of each part inside the variant's vector,
                      parts that own no channel dropped -> exactly what PartRVQVAE(part_channels=...) wants.
       explicit_idx : local indices of MoMask's "explicit" geometric channels (our `local_positions`, the 24
                      local joint positions), or None for the "action" variant, which has none.
    """
    _, groups = part_groups(variant)
    expl = feature_groups(variant).get(EXPLICIT_FEATURE)
    return groups, (expl if expl is not None and len(expl) else None)


def token_mean_std(stats):
    """TokenStats -> (mean [435], std [435]) float32 over the full token."""
    mean = np.concatenate([stats.root_mean, stats.body_mean]).astype(np.float32)
    std = np.concatenate([stats.root_std, stats.body_std]).astype(np.float32)
    assert mean.shape == (TOKEN_DIM,) and std.shape == (TOKEN_DIM,)
    return mean, std


class RVQWindowDataset(Dataset):
    """Windows of normalised physics tokens, one channel variant, canonicalised on the window's first frame.

    __getitem__ -> FloatTensor [window, n_channels]  (a bare tensor; use meta(i) for the (clip, start) of item i)

    Attributes
      channels        [C] int64   token channels the variant keeps
      n_channels/dim  int         C
      mean, std       [C] float32 normalisation statistics restricted to the variant
      part_names      list[str]   parts that own channels in this variant
      part_groups     list[[..]]  local channel indices per part (a partition of range(C))
      feature_groups  dict        feature name -> local channel indices
      explicit_idx    [..]|None   local_positions channels (MoMask's explicit loss term)
      windows         [N,2] int64 (clip index, start frame)
      clip_lengths    [n_clips]   frames per clip
      n_clips_short   int         clips dropped for being shorter than the window
    """

    def __init__(self, split, variant="action", window=DEFAULT_WINDOW, stride=1, downsample=DOWNSAMPLE,
                 stats_path=DEFAULT_STATS, normalise=True, max_clips=0, max_windows=0, seed=0,
                 env_constants=None, train=False):
        """env_constants / train: accepted for signature compatibility with the policy datasets and UNUSED --
        there is no augmentation here (see the module docstring)."""
        assert split != "val", "the val split is banned in this project (CLAUDE.md §1)"
        assert variant in VARIANTS, f"unknown variant {variant!r}, expected one of {VARIANTS}"
        assert window > 1 and window % downsample == 0, \
            f"window ({window}) must be a positive multiple of the downsampling rate ({downsample})"
        assert stride >= 1
        self.split, self.variant, self.window, self.stride, self.downsample = split, variant, window, stride, downsample
        self.normalise = bool(normalise)
        self.train = bool(train)                            # unused: no augmentation

        d = joblib.load(os.path.join(ROOT, f"hml_phys_{split}.pkl"))
        if max_clips:
            d = {k: v[:max_clips] for k, v in d.items()}
        self.clips = d
        self.clip_lengths = np.asarray(d["n_frames"], dtype=np.int64)

        self.channels = variant_channels(variant)
        self.dim = self.n_channels = int(len(self.channels))
        self.part_names, self.part_groups = part_groups(variant)
        self.feature_groups = feature_groups(variant)
        self.explicit_idx = variant_parts(variant)[1]
        self.stats_path = stats_path
        self.stats = TokenStats(stats_path) if stats_path else None
        if self.stats is not None:
            self._tmean, self._tstd = token_mean_std(self.stats)
        else:
            assert not self.normalise, "normalise=True needs a stats_path"
            self._tmean, self._tstd = np.zeros(TOKEN_DIM, np.float32), np.ones(TOKEN_DIM, np.float32)
        self.mean, self.std = self._tmean[self.channels].copy(), self._tstd[self.channels].copy()

        idx = [(i, s) for i, T in enumerate(self.clip_lengths) if T >= window
               for s in range(0, int(T) - window + 1, stride)]
        self.windows = np.asarray(idx, dtype=np.int64).reshape(-1, 2)
        self.n_clips_short = int((self.clip_lengths < window).sum())
        self.n_windows_all = len(self.windows)
        if max_windows and len(self.windows) > max_windows:     # deterministic subset, for smoke runs
            sel = np.sort(np.random.RandomState(seed).choice(len(self.windows), max_windows, replace=False))
            self.windows = self.windows[sel]

    def __len__(self):
        return len(self.windows)

    def meta(self, i):
        """-> (clip index, start frame) of item i."""
        clip, s = map(int, self.windows[i])
        return clip, s

    def raw_token_window(self, i):
        """-> un-normalised full token [W, 435] float32 of window i (root 15 | body 420)."""
        clip, s = self.meta(i)
        c, e = self.clips, s + self.window
        # the canonical frame is degenerate by construction (its hip-across direction is exactly +x, so
        # qbetween divides by a zero norm before tokens.heading_quat overrides that row with the exact
        # limit); the resulting numpy FP warning is expected and would otherwise fire once per window.
        with np.errstate(invalid="ignore", divide="ignore"):
            root, body = tk.window_tokens(c["body_pos"][clip][s:e], c["dof_state"][clip][s:e],
                                          c["root_state"][clip][s:e], c["action"][clip][s:e], origin=0)
        return np.concatenate([root, body], -1).astype(np.float32)

    def __getitem__(self, i):
        x = self.raw_token_window(i)
        if self.normalise:
            x = (x - self._tmean) / self._tstd
        return torch.from_numpy(np.ascontiguousarray(x[:, self.channels], dtype=np.float32))

    # ---- helpers for consumers (training / evaluation)
    def denorm(self, x):
        """normalised [..., C] (numpy or torch) -> physical units."""
        if torch.is_tensor(x):
            m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * s + m
        return x * self.std + self.mean

    def dof_index(self):
        """for the action channels of this variant: the dof (joint-axis) index of each action channel."""
        loc = self.feature_groups.get("action")
        if loc is None:
            return np.zeros(0, np.int64)
        return (self.channels[loc] - ACTION_CH0).astype(np.int64)

    def describe(self):
        return (f"RVQWindowDataset(split={self.split}, variant={self.variant}, window={self.window}, "
                f"stride={self.stride}) -> {len(self)} windows of {self.window}x{self.n_channels} "
                f"({self.n_windows_all} before subsetting), {len(self.clip_lengths)} clips "
                f"({self.n_clips_short} shorter than the window), parts "
                + ", ".join(f"{n}:{len(g)}" for n, g in zip(self.part_names, self.part_groups)))
