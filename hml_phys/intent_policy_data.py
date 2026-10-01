"""Training windows for the MIND-style intent arch (docs/07 §21): the policy window plus the three state sequences whose frozen-VAE
encodings are MIND's intents.

One item =
  policy window  [H_sparse sparse | H dense | F_act future]   (normalised root/body tokens, canonical frame = newest
                                                                history frame, exactly as PhysWindowDataset)
  hist  [16, 366]  dense history t-15..t      -> VAE -> history intent   I_h  (IIP condition)
  fut   [16, 366]  future t+1..t+16           -> VAE -> immediate intent I_I  (IIP target)
  holi  [16, 366]  the span the chosen caption describes (whole clip if untagged, its [from, to] tag otherwise),
                   uniformly sampled at 16 frames, canonicalised on the span's first frame -> VAE -> I_H (HIP target)
hist / fut come out of the SAME window tokens as the policy rows (same origin), so they are bit-identical to what
the closed loop feeds the VAE at test time. The window index is built with F = 16 so that the immediate-intent
target always exists; the policy only uses the first F_act of those 16 future rows.
"""
import numpy as np
import torch

from hml_phys import tokens as tk
from hml_phys.dataset import PhysWindowDataset, collate
from hml_phys.intent_data import state_from_tokens, holistic_rows, holistic_crop_rows, vae_history_input

L_INTENT = 16


class IntentPolicyDataset(PhysWindowDataset):
    def __init__(self, split, F_act=4, hip_aug=False, p_hip_exact=0.25, span_scalars=False, **kw):
        """hip_aug (training only): the holistic target is drawn like VAE v2's holistic augmentation -- with prob.
        p_hip_exact the exact test-time sampling of the caption's span, otherwise a random sub-span (each end trimmed
        by <= 25%) at 16 evenly spaced frames with a random phase -- so one caption has many holistic targets.
        span_scalars: for time-tagged captions, progress / total length refer to the tagged span (what the evaluator
        rolls out as its own episode) instead of the whole clip."""
        kw = dict(kw); kw["F"] = L_INTENT
        super().__init__(split, **kw)
        assert self.H == L_INTENT, "MIND: history intent = the last 16 frames = the dense history"
        self.F_act = F_act
        self.hip_aug, self.p_hip_exact, self.span_scalars = bool(hip_aug) and self.train, p_hip_exact, bool(span_scalars)
        self._last_tag = (0.0, 0.0)
        # holistic sequence of every clip for untagged captions (tagged ones are computed per item)
        c = self.clips
        self.holi_whole = []
        for i, n in enumerate(c["n_frames"]):
            self.holi_whole.append(self._holistic(i, 0, int(n)))

    def _holistic(self, clip, a, b):
        c = self.clips
        sp = slice(a, b)
        root, body = tk.window_tokens(c["body_pos"][clip][sp], c["dof_state"][clip][sp], c["root_state"][clip][sp],
                                      c["action"][clip][sp], origin=0)
        rows = holistic_rows(b - a, L_INTENT)
        return state_from_tokens(*self.stats.norm(root[rows], body[rows]))

    def pick_text(self, clip, s, rng):
        """as PhysWindowDataset.pick_text, but remember the chosen caption's time tag for the holistic span."""
        fps = float(self.clips["fps_eff"][clip])
        f0, f1 = (s + self.H) / fps, (s + self.H + self.future_len(clip, max(s, 0))) / fps
        cand = [(ci, a, b) for ci, a, b in self.cands[clip] if ((a == 0.0 and b == 0.0) or (a < f1 and b > f0)) and ci >= 0]
        if not cand:
            self._last_tag = (0.0, 0.0)
            return -1
        ci, a, b = cand[rng.randint(len(cand))]
        self._last_tag = (a, b)
        return int(ci)

    def __getitem__(self, i):
        d = super().__getitem__(i)
        clip = d["clip"]
        a_sec, b_sec = self._last_tag
        n = int(self.clips["n_frames"][clip])
        fps = float(self.clips["fps_eff"][clip])
        tagged = not (a_sec == 0.0 and b_sec == 0.0)
        if tagged:
            a = int(np.clip(round(a_sec * fps), 0, n - 2)); b = int(np.clip(round(b_sec * fps), a + 2, n))
        else:
            a, b = 0, n
        if self.hip_aug:
            rng = np.random.RandomState((self.rng.randint(1 << 30) + 7919 * i) % (1 << 31))
            hr = holistic_crop_rows(b - a, L_INTENT, rng, self.p_hip_exact) + a
            c = self.clips
            sp = slice(int(hr[0]), int(hr[-1]) + 1)
            r_, b_ = tk.window_tokens(c["body_pos"][clip][sp], c["dof_state"][clip][sp], c["root_state"][clip][sp],
                                      c["action"][clip][sp], origin=0)
            holi = state_from_tokens(*self.stats.norm(r_[hr - hr[0]], b_[hr - hr[0]]))
        elif tagged:
            holi = self._holistic(clip, a, b)
        else:
            holi = self.holi_whole[clip]
        if self.span_scalars and tagged:
            now = d["start"] + self.H if d["mode"] == "normal" else 0
            d["progress"] = np.float32(np.clip((now - a) / max(1, b - a), 0.0, 1.0))
            d["total_len"] = np.float32((b - a) / fps)
        # dense history and the 16 future rows sit at the end of the (front-padded) layout
        nh = d["n_hist"]                                   # first future row in the padded layout
        root, body = d["root"].numpy(), d["body"].numpy()
        x = state_from_tokens(root, body)
        d["hist"] = torch.from_numpy(vae_history_input(x[nh - L_INTENT:nh]))    # exactly the VAE training input
        d["fut"] = torch.from_numpy(x[nh:nh + L_INTENT].copy())
        d["holi"] = torch.from_numpy(np.ascontiguousarray(holi))
        # policy window: drop the future rows beyond F_act (padding is at the front, so cut from the back)
        cut = L_INTENT - self.F_act
        for k in ("root", "body", "observed_mask", "valid", "frame_index"):
            d[k] = d[k][:len(d[k]) - cut]
        d["n_frames"] = d["n_frames"] - cut
        return d


def collate_intent(batch, text_cache):
    out = collate(batch, text_cache)
    for k in ("hist", "fut", "holi"):
        out[k] = torch.stack([b[k] for b in batch])
    return out
