# Code review brief — v3 (local-root bridge, short future, sparse long history, new canonical origin)

The authoritative spec is `docs/07_motioncraft_adaptation.md` §15 (plus §10–§14 for everything unchanged).
Project rules in `CLAUDE.md` (no val evaluation of any kind; full baseline table when reporting).

## What v3 had to change, and why

v1 (fixed 32-frame future, 182M) reached test R@1 0.254 at 25k steps and overfits after that; v2
(whole-remaining-sequence future, 82M) fixed the overfitting but raised the fall rate from 29% to 75% and
made rollouts 2.3x slower, because it generated up to 320 tokens per replan and executed 4 of them.
Reference implementations consulted: KiMoDo (`vendor_kimodo`), ARDY (`vendor_ardy`), MotionCraft
(`vendor_motioncraft`), the author's newer `vendor_moge_umo`, plus the MIND and SCRIPT papers.

1. **Local-root bridge.** KiMoDo/ARDY/MotionCraft/moge_UMO_ST all convert the predicted global root into a
   local velocity root before the body stage (`KimodoRootConditioner`, byte-identical in both MotionCraft
   clones); our v1/v2 fed the raw 15-d global root. Restored: 4-d `[yaw rate, dx/dt, dy/dt, height]`,
   own normalisation statistics, detached in training, differentiable at test time, last valid row copies
   its predecessor. Because window rows may be non-contiguous (see 3), the finite differences are divided by
   the true frame gap `dt`; `dt = 1` reproduces the reference formula exactly.
2. **Short future.** `whole_sequence` off; F = 32, K = 4.
3. **Sparse long history.** Window = `[H_sparse sparse | H dense | F future]`. Sparse frames are drawn from
   the preceding `L_max - H` frames with SCRIPT eq. (6) (`tokens.sample_sparse_history`); SCRIPT's ablation
   values history at +0.318 R@1 (0.117 without, 0.435 with non-uniform sampling, 0.419 uniform). Tokens are
   computed on the CONTIGUOUS span and the chosen rows gathered afterwards, so each row keeps its true
   instantaneous velocities. Each row carries a signed `frame_index` (first generated frame = 0, history
   negative, true offsets) which drives RoPE and the local-root gaps — ARDY's mechanism, which is what makes
   history length a test-time knob. During training `n_sparse` and `alpha` are drawn per sample.
4. **Canonical origin = newest history frame** (`canonicalize(origin=...)`), because with a 5 s history the
   old origin (oldest window frame) can be metres away from the frames being predicted. Statistics recomputed
   into `token_stats_v3.npz`, which also carries `local_root_mean/std`.

Explicitly NOT done (and must not have crept in): FSQ/any quantisation; whole-sequence per-frame generation;
MIND's intent VAE; disabling pooled text in AdaLN; changing the text encoder; data scale / mirroring; a
per-frame semantic channel; ARDY's heading prefix token (unnecessary because we rotate during canonicalisation).

## Files to review
- `hml_phys/tokens.py` — `canonicalize(origin=)`, `canonicalize_batch(origin=)`, `window_tokens*(origin=)`,
  `root_to_local_root`, `sample_sparse_history`
- `hml_phys/dataset.py` — constructor (`H_sparse/L_max/alpha/randomize_history/p_no_sparse/alpha_range`),
  `draw_history_cfg`, `raw_window`, `__getitem__` (gather, padding at the FRONT, `frame_index`, `n_hist`
  convention = index of the first future row in the padded layout), `collate` (left trim), `TokenStats`
- `hml_phys/mc_model.py` — `to_local_root`, stats buffers, `bridge_dim`, signed `frame_index` → RoPE
- `hml_phys/flow.py` — `frame_index` threading through `euler_sample` (incl. the CFG batch)
- `hml_phys/mc_rollout.py` — `HistoryBuffer` (ring of `L_max`), the plan block, `load_policy`
- `scripts/hml_phys/{train_mc.py, compute_token_stats.py}`
- References: `vendor_kimodo/kimodo/model/twostage_denoiser.py`,
  `vendor_motioncraft/models/raw_motion/hy273_root_conditioning.py`,
  `vendor_ardy/ardy/model/{auto_latent_twostage_denoiser.py, backbone.py, ardy_model.py}`,
  `vendor_motioncraft/models/codeflow/dit_blocks.py` (`_rope_cos_sin` must accept negative positions).

## Evidence already collected (verify, don't trust)
CPU: sparse sampler mean index 73 / 99.5 / 110.2 for alpha 0 / 3 / 5 over a span of 138 (higher = more
recent-biased), always 16 distinct in range; `canonicalize` puts `root_trans = (0,0,h)` at any chosen origin;
local root computed on gathered rows with `frame_index` matches the contiguous computation value-for-value;
dataset invariants (mask layout, monotone `frame_index` with `-1`/`0` at the boundary, valid counts, finite
values) hold over 300 samples; `H_sparse=0` reproduces the old 48-token window with `frame_index -16..31`.
GPU: 40-step training smoke (81.6M, `[16|16|32]`, local_root on); 8-env closed-loop smoke; per-window vs
batched tokenisation agree (root 0, body 5.4e-4); torch local root vs numpy reference 1.4e-6.

## The three checks
1. **Completeness and bugs** — anything that would silently produce wrong training targets, a wrong
   train/test mismatch, or wrong numbers. Pay attention to: the padded-slot convention (padding at the front,
   `observed_mask=1` and `valid=0`), whether padded rows can leak into the loss or the attention, the
   `n_hist` convention, the boundary between history and future in the local-root differences, the rollout
   buffer indexing (`span0`, `dense0`, `hist.n` before the buffer is full), `progress` definition parity
   between training and rollout, the consistency loss masking, resume/EMA/validation, and the CFG branch.
2. **No corner-cutting** — compare against §15 line by line; flag anything simplified, defaulted weaker than
   specified, or claimed in the docs but not implemented.
3. **Contract adherence** — each bullet of §15: done / partial / not done, with evidence.
