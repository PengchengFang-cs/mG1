"""Windows for the CodeFlow policy (docs/08 §10).

A CodeFlow window is CONTIGUOUS -- the tokenizer is a temporal convolution, so it cannot be fed the policy's
sparse distant history. One window is `window` frames split as [n_hist observed | window - n_hist generated],
canonicalised on the window's own frame 0, which is exactly the distribution `RVQWindowDataset` (and therefore
the frozen tokenizer) was trained on. That is the "native" convention the user settled on 2026-09-21: every
generated chunk carries its own frame and the policy side composes the transform (docs/08 §9.4).

Losing the sparse long history is the deliberate difference between version 1 and the current best policy;
version 2 gets the long horizon back through the holistic intent (HIP), which is unchanged.

__getitem__ -> dict(x [W, C] normalised tokens of the variant, text_idx, progress, total_len)

TWO-TOKENIZER ("码本对照") MODE -- where the channel split happens
-----------------------------------------------------------------
When the policy observes the full token and generates only the actions (`CodeFlowPolicy(rvq_obs_ckpt=...)`),
the dataset is built with `variant="token"` and returns the 435-channel window UNCHANGED; the trainer slices
the 69 action channels out of the future half on the GPU with `CodeFlowPolicy.gen_channels_of` (built from
`rvq_data.variant_local_channels`).  The split is deliberately NOT done here: the two halves would otherwise
be collated and copied twice through the dataloader workers for no gain, and keeping one tensor per item makes
it impossible for the history and the future to come from different windows.
"""
import numpy as np
import torch

from hml_phys import tokens as tk
from hml_phys.dataset import TextCache, norm_caption
from hml_phys.intent_data import holistic_rows, state_from_tokens
from hml_phys.rvq_data import DEFAULT_STATS, DEFAULT_WINDOW, DOWNSAMPLE, RVQWindowDataset

L_INTENT = 16                      # MIND's sequence length for all three intent streams


class CodeFlowDataset(RVQWindowDataset):
    def __init__(self, split, variant="action", window=DEFAULT_WINDOW, n_hist=32, text_cache=None,
                 stride=1, downsample=DOWNSAMPLE, stats_path=DEFAULT_STATS, max_clips=0, max_windows=0,
                 seed=0, train=True, env_constants=None, intent=False):
        super().__init__(split, variant=variant, window=window, stride=stride, downsample=downsample,
                         stats_path=stats_path, max_clips=max_clips, max_windows=max_windows, seed=seed,
                         env_constants=env_constants, train=train)
        assert 0 < n_hist < window and n_hist % downsample == 0, \
            f"n_hist ({n_hist}) must be a positive multiple of {downsample} and shorter than the window"
        self.n_hist = int(n_hist)
        self.n_lat = window // downsample
        self.n_lat_hist = n_hist // downsample
        self.text_cache = text_cache
        self.rng = np.random.RandomState(seed)
        # text candidates per clip: (cache index, f_tag seconds, to_tag seconds); f=t=0 means untagged
        self.intent = bool(intent)
        if self.intent:
            assert n_hist >= L_INTENT and window - n_hist >= L_INTENT, \
                f"the intent streams need {L_INTENT} frames on each side of the boundary"
            # holistic intent: the whole clip downsampled to 16 frames, canonicalised on its first frame,
            # exactly as hml_phys/intent_data.IntentSeqDataset builds it (docs/07 §20)
            c = self.clips
            self.holi = np.zeros((len(c["n_frames"]), L_INTENT, 366), np.float32)
            for i, n in enumerate(c["n_frames"]):
                with np.errstate(invalid="ignore", divide="ignore"):
                    r_, b_ = tk.window_tokens(c["body_pos"][i], c["dof_state"][i], c["root_state"][i],
                                              c["action"][i], origin=0)
                rows = holistic_rows(int(n), L_INTENT)
                self.holi[i] = state_from_tokens(*self.stats.norm(r_[rows], b_[rows]))
        self.cands = []
        for texts in self.clips["texts"]:
            self.cands.append([(self.text_cache.index.get(norm_caption(t["caption"]), -1) if self.text_cache else -1,
                                float(t["f_tag"]), float(t["to_tag"])) for t in texts])

    def pick_text(self, clip, s, rng):
        """A caption whose tag span overlaps the GENERATED half of the window (untagged captions always apply)."""
        fps = float(self.clips["fps_eff"][clip])
        f0, f1 = (s + self.n_hist) / fps, (s + self.window) / fps
        cand = [ci for ci, a, b in self.cands[clip] if ci >= 0 and ((a == 0.0 and b == 0.0) or (a < f1 and b > f0))]
        return int(cand[rng.randint(len(cand))]) if cand else -1

    def __getitem__(self, i):
        clip, s = self.meta(i)
        x = super().__getitem__(i)
        rng = self.rng if self.train else np.random.RandomState(i)
        T = float(self.clip_lengths[clip])
        out = dict(x=x, text_idx=self.pick_text(clip, s, rng),
                   progress=float(s + self.n_hist) / max(T, 1.0), total_len=T / 30.0)
        if self.intent:
            # the intent VAE is canonicalised on the NEWEST HISTORY frame, not on the window's frame 0, so its
            # two sequences are tokenised again on their own span (docs/08 §9.4, docs/07 改动 3b)
            a = s + self.n_hist - L_INTENT
            sp = slice(a, a + 2 * L_INTENT)
            c = self.clips
            with np.errstate(invalid="ignore", divide="ignore"):
                r_, b_ = tk.window_tokens(c["body_pos"][clip][sp], c["dof_state"][clip][sp],
                                          c["root_state"][clip][sp], c["action"][clip][sp], origin=L_INTENT - 1)
            st = state_from_tokens(*self.stats.norm(r_, b_))
            out["hist"] = torch.from_numpy(st[:L_INTENT])
            out["fut"] = torch.from_numpy(st[L_INTENT:])
            out["holi"] = torch.from_numpy(self.holi[clip])
        return out


def collate_codeflow(batch, text_cache):
    out = dict(x=torch.stack([b["x"] for b in batch]))
    for k in ("hist", "fut", "holi"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    empty = text_cache.index.get("", -1)          # no caption -> CLIP("") features, the same vector CFG uses
    toks, pooled, lens = zip(*[text_cache.get(b["text_idx"] if b["text_idx"] >= 0 else empty) for b in batch])
    out["text_tokens"] = torch.from_numpy(np.stack(toks))
    out["text_pooled"] = torch.from_numpy(np.stack(pooled))
    out["text_len"] = torch.tensor(lens)
    out["text_dropped"] = torch.tensor([b["text_idx"] < 0 for b in batch])
    out["progress"] = torch.tensor([b["progress"] for b in batch], dtype=torch.float32)
    out["total_len"] = torch.tensor([b["total_len"] for b in batch], dtype=torch.float32)
    return out
