# 08 — MoMask residual VQ-VAE: reference spec

Reference extraction for the MoMask-style residual VQ tokenizer we are adding as `hml_phys/rvq.py`
(model + `scripts/hml_phys/train_rvq.py`). Everything below is read off the upstream code, with
`file:line` citations. Paths are relative to `vendor_momask/` (clone of
`https://github.com/EricGuo5513/momask-codes`, gitignored) unless stated otherwise.

Parity harness: `scripts/hml_phys/rvq_parity_test.py`.

---

## 0. The one-line summary

MoMask's tokenizer is a **1-D convolutional VQ-VAE with a residual (RVQ) bottleneck**:
a temporal-conv encoder downsamples 4x, **6 sequential EMA codebooks** each quantise the *residual*
left by the previous one, their code vectors are **summed**, and a mirrored conv decoder upsamples
back. Codebooks are **not** trained by gradient — they are EMA-updated with dead-code random restart.
The only gradient signal into the codebook path is the straight-through estimator plus a commitment
term on the encoder side.

---

## 1. Encoder / decoder architecture

`models/vq/encdec.py`, blocks from `models/vq/resnet.py`.

### 1.1 Encoder (`models/vq/encdec.py:5-34`)

```
Conv1d(input_width -> width, k=3, s=1, p=1)                      # encdec.py:20
ReLU()                                                           # encdec.py:21
repeat down_t times:                                             # encdec.py:23
    Conv1d(width -> width, k=filter_t, s=stride_t, p=pad_t)      # encdec.py:26
    Resnet1D(width, depth, dilation_growth_rate, norm=norm, activation=activation)   # encdec.py:27
Conv1d(width -> output_emb_width, k=3, s=1, p=1)                 # encdec.py:30
```

* `filter_t, pad_t = stride_t * 2, stride_t // 2` (`encdec.py:19`), so with `stride_t=2` every down
  block is `Conv1d(k=4, s=2, p=1)` — an exact halving of T for even T.
* `down_t=2, stride_t=2` (the shipped config, §4) gives **T_latent = T / 4**.
  Verified: `[2,64,263] -> codes [2,16,6]`.
* Encoder output is **NCT** (`[B, code_dim, T_latent]`), no permute at the end.
* `assert output_emb_width == code_dim` (`models/vq/model.py:23`) — the encoder width *is* the code
  dim; there is no projection between encoder and codebook.

### 1.2 Decoder (`models/vq/encdec.py:37-68`)

```
Conv1d(output_emb_width -> width, k=3, s=1, p=1)                 # encdec.py:51
ReLU()                                                           # encdec.py:52
repeat down_t times:                                             # encdec.py:53
    Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, ...)   # encdec.py:56
    Upsample(scale_factor=2, mode='nearest')                     # encdec.py:57
    Conv1d(width -> width, k=3, s=1, p=1)                        # encdec.py:58
Conv1d(width -> width, k=3, s=1, p=1)                            # encdec.py:61
ReLU()                                                           # encdec.py:62
Conv1d(width -> input_emb_width, k=3, s=1, p=1)                  # encdec.py:63
```

* **`scale_factor=2` is hard-coded** (`encdec.py:57`) — the decoder silently ignores `stride_t`.
  Any `stride_t != 2` breaks the encoder/decoder length match. Keep `stride_t = 2`.
* `Decoder.forward` returns `x.permute(0, 2, 1)` (`encdec.py:68`), i.e. **NTC** `[B, T, D]`.
  `RVQVAE.postprocess` (`models/vq/model.py:47-50`) is therefore dead code — `forward` never calls it
  (`model.py:76-78`). Easy parity trap: the decoder already transposes.
* Upsampling is `nearest`, not transposed conv.

### 1.3 Residual block (`models/vq/resnet.py:12-84`)

`ResConv1DBlock(n_in, n_state=n_in, dilation, activation, norm, dropout=0.2)`:

```
x_orig = x
x = norm1(x); x = act1(x)                                        # resnet.py:54-55  (pre-activation)
x = Conv1d(n_in -> n_state, k=3, s=1, p=dilation, dilation=dilation)   # resnet.py:44, 57
x = norm2(x); x = act2(x)                                        # resnet.py:62-63
x = Conv1d(n_state -> n_in, k=1, s=1, p=0)                       # resnet.py:45, 66
x = Dropout(0.2)(x)                                              # resnet.py:46, 67
return x + x_orig                                                # resnet.py:68
```

* **Pre-activation** ordering (norm -> act -> conv), bottleneck is 3x3 dilated then 1x1.
* `padding = dilation` (`resnet.py:16`) so length is preserved.
* **`dropout=0.2` inside every residual block, defaulted, never exposed as a CLI flag.** It is active
  in train mode. Parity comparisons must run both models in `.eval()`.
* `Resnet1D(n_in, n_depth, dilation_growth_rate, reverse_dilation=True, ...)` (`resnet.py:72-84`):
  block `d` (d = 0..n_depth-1) uses `dilation = dilation_growth_rate ** d` (`resnet.py:76`), so with
  `dilation_growth_rate=3, depth=3` the dilations are **1, 3, 9**.
  `reverse_dilation` reverses the list (`resnet.py:78-79`) -> **9, 3, 1**.
* Gotcha: `Resnet1D`'s own default is `reverse_dilation=True`, but the **encoder** calls it without
  the flag (`encdec.py:27`) and the **decoder** passes it explicitly (`encdec.py:56`) — so *both* are
  reversed (9, 3, 1). There is no encoder/decoder asymmetry despite the code reading as if there is.
* `norm=None` -> `nn.Identity()` for both norms (`resnet.py:29-30`). The shipped config uses **no
  normalisation at all** (`--vq_norm` default `None`, `options/vq_option.py:39`).
* `activation='relu'` -> `nn.ReLU()` (`resnet.py:32-34`). `'silu'` maps to a *custom* `x*sigmoid(x)`
  class (`resnet.py:4-9`, `resnet.py:36-38`), `'gelu'` to `nn.GELU()`.

### 1.4 Parameter count

Shipped HumanML3D config (input 263, width 512, code_dim 512, down_t 2, depth 3, n_quant 6,
nb_code 512): **19.44 M parameters** (measured; `train_vq.py:100-105` prints the same figure).

---

## 2. The residual quantiser

`models/vq/residual_vq.py` (stack) and `models/vq/quantizer.py` (single codebook).

### 2.1 Stack configuration (`models/vq/model.py:31-40`)

```python
rvqvae_config = {
    'num_quantizers': args.num_quantizers,            # 6 for the shipped model
    'shared_codebook': args.shared_codebook,          # False
    'quantize_dropout_prob': args.quantize_dropout_prob,   # 0.2
    'quantize_dropout_cutoff_index': 0,               # HARD-CODED 0, not a CLI flag
    'nb_code': nb_code, 'code_dim': code_dim, 'args': args,
}
```

* **Number of layers: 6** (`options/vq_option.py:41` default 3, but README:175 and README:192 both
  say 6, and the released checkpoint is named `rvq_nq6_dc512_nc512_noshare_qdp0.2`, README:207).
* **Shared codebook: off.** `shared_codebook=True` would put *the same module object* into the
  `ModuleList` 6 times (`residual_vq.py:44-45`) — one shared codebook and one shared EMA state, i.e.
  genuinely tied, not just tied-init. Off in the shipped model (`--shared_codebook` is a
  `store_true` flag, `vq_option.py:42`, and the checkpoint name says `noshare`).
* **Codebook size 512, code dim 512** (`vq_option.py:28-29`; the `1024` in `model.py:11` is a dead
  signature default — `train_vq.py:87-98` always passes `opt.nb_code`).
* **No L2 normalisation anywhere.** No `l2norm` / `cosine_sim` path exists in this fork of the
  quantiser; distances are raw squared Euclidean (`quantizer.py:72-74`). (MoGeFlow's older
  part-VQ ancestor did normalise the encoder feature, but that line is commented out — see §6.)

### 2.2 Per-layer codebook: `QuantizeEMAReset` (`models/vq/quantizer.py:35-158`)

**Init (`quantizer.py:43-65`).** The codebook is a **buffer of zeros**, `requires_grad=False`
(`quantizer.py:47`) — it is *not* an `nn.Parameter` and receives **no gradient**. It is lazily
initialised from real data on the first *training* forward:

```python
if self.training and not self.init:      # quantizer.py:136-137
    self.init_codebook(x)                # x = flattened encoder features [(B*T), C]
```

`init_codebook` takes the first `nb_code` rows of `_tile(x)` (`quantizer.py:60-65`); `_tile`
(`quantizer.py:49-58`) repeats the batch until it has at least `nb_code` rows and adds Gaussian noise
with `std = 0.01 / sqrt(code_dim)`. `code_sum = codebook.clone()`, `code_count = ones(nb_code)`.

> **Parity trap:** `init`, `code_sum`, `code_count` are plain Python attributes, **not** buffers, so
> they are absent from `state_dict()`. Only `codebook` round-trips through a checkpoint. A freshly
> constructed model in `.eval()` has an **all-zero codebook** and will return index 0 everywhere.
> Any test or eval must either load a checkpoint or set `codebook`/`init`/`code_sum`/`code_count`
> by hand.

**Assignment (`quantizer.py:67-80`).** Squared Euclidean distance `|x|^2 - 2 x k^T + |k|^2`, then

```python
code_idx = gumbel_sample(-distance, dim=-1, temperature=sample_codebook_temp,
                         stochastic=True, training=self.training)   # quantizer.py:78
```

`gumbel_sample` (`quantizer.py:18-33`) adds Gumbel noise **only if** `training and stochastic and
temperature > 0`; otherwise plain `argmax`. Consequences:

| path | temperature | behaviour |
|---|---|---|
| `RVQVAE.forward` (training) | **0.5, hard-coded** at `model.py:72` | stochastic Gumbel assignment |
| `RVQVAE.forward` (eval) | 0.5 but `training=False` | deterministic argmax |
| `RVQVAE.encode` -> `ResidualVQ.quantize` | default **0** (`residual_vq.py:178`) | deterministic argmax, always |

So **tokenisation is always deterministic argmax, while the training reconstruction path is
stochastic**. `model.py:70-71` shows the commented-out variant that also forces `force_dropout_index=0`.

**EMA update (`quantizer.py:100-123`), train mode only.** Not a gradient update:

```python
code_sum   = mu * code_sum   + (1 - mu) * (one_hot @ x)     # quantizer.py:112
code_count = mu * code_count + (1 - mu) * one_hot.sum(-1)   # quantizer.py:113
usage      = (code_count >= 1.0).float()                    # quantizer.py:115
codebook   = usage * (code_sum / code_count) + (1 - usage) * code_rand   # quantizer.py:116-117
```

* `mu = 0.99` (`vq_option.py:30`).
* **Dead-code reset / expiry:** a code whose EMA count has decayed below **1.0** is *replaced* by a
  random row of `_tile(x)` — a real encoder feature from the current batch plus noise
  (`quantizer.py:108-109, 117`). This is the "Reset" in `QuantizeEMAReset`. `code_rand` is resampled
  every step, so the restart target is always fresh data. There is no counter/age buffer and no
  cool-down.
* The sibling `QuantizeEMA` (`quantizer.py:160-179`) keeps the old code instead of restarting
  (`quantizer.py:175`); it is **not used** — `residual_vq.py:44,47` always builds `QuantizeEMAReset`,
  and the `QuantizeEMA` alternative is commented out at `residual_vq.py:48`.

**Losses and straight-through (`quantizer.py:147-150`).**

```python
commit_loss = F.mse_loss(x, x_d.detach())   # quantizer.py:147
x_d = x + (x_d - x).detach()                # quantizer.py:150  straight-through
```

Note the comment at `quantizer.py:147`: *"It's right. the t2m-gpt paper is wrong on embed loss and
commitment loss."* — there is **only** the commitment term (encoder pulled to code), **no codebook
/ embedding loss**, because the codebook is EMA-driven. The commitment loss is an **unweighted
`mse_loss` inside the quantiser**; the `0.02` weight is applied later, in the trainer (§3).

**Perplexity (`quantizer.py:89-98`, `quantizer.py:120-121`).** `exp(-sum p log(p+1e-7))` over the
batch's code histogram, where `p = count / total`. In **train** mode it is computed from the *raw
batch counts* inside `update_codebook` (`quantizer.py:120-121`); in **eval** mode from
`compute_perplexity` (`quantizer.py:90-98`). Same formula, different call site.

### 2.3 Residual loop (`models/vq/residual_vq.py:99-169`)

```python
quantized_out = 0.;  residual = x
for quantizer_index, layer in enumerate(self.layers):
    if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
        all_indices.append(null_indices);  continue            # residual_vq.py:133-136
    quantized, embed_indices, loss, perplexity = layer(residual, return_idx=True,
                                                       temperature=sample_codebook_temp)
    residual -= quantized.detach()                              # residual_vq.py:146
    quantized_out += quantized                                  # residual_vq.py:147
all_indices  = torch.stack(all_indices, dim=-1)                 # residual_vq.py:156  -> [B, T, Q]
all_losses     = sum(all_losses) / len(all_losses)              # residual_vq.py:157
all_perplexity = sum(all_perplexity) / len(all_perplexity)      # residual_vq.py:158
```

* `quantized_out` is the **plain sum** of the per-layer code vectors — no per-layer scaling.
* Each layer sees `residual`, which carries the **straight-through** gradient of every previous
  layer; only `quantized.detach()` is subtracted, so gradients reach the encoder through
  `quantized_out` and through `residual`.
* **`residual -= quantized.detach()` is in-place** (`residual_vq.py:146`) and `residual` starts as
  the encoder output `x`, so `ResidualVQ.forward` **mutates the encoder output tensor in place**.
  It happens to be autograd-safe (conv backward needs its input, not its output), but our version
  should use the out-of-place form — numerically identical, and it is what `quantize()` already does
  (`residual_vq.py:180`).
* `all_losses` / `all_perplexity` are averaged over the **surviving** layers only (dropped layers
  `continue` before appending), so the commitment loss magnitude does not change with dropout depth.

### 2.4 Quantise-dropout (`models/vq/residual_vq.py:112-127`)

```python
should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
start_drop_quantize_index = num_quant                      # i.e. nothing dropped
if should_quantize_dropout:
    start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)  # = randrange(0, 6)
    null_indices = full([B, T_latent], -1, dtype=long)
```

* One Bernoulli draw **per forward call, shared by the whole batch** (`random.random()`, Python RNG,
  not per-sample).
* When it fires, `start` ~ `Uniform{0, ..., num_quant-1}`; layers with `index > start` are skipped
  and their indices written as **-1**. So the number of **kept** layers is `start + 1`, uniform on
  `{1, ..., num_quant}`. At least one layer always survives; `start = num_quant - 1` means nothing is
  actually dropped.
* `quantize_dropout_cutoff_index = 0` is hard-coded in `model.py:35` (the comment at
  `residual_vq.py:117` describing it as "keep quant layers <= cutoff" is misleading — with 0 it is
  simply the lower bound of the uniform draw).
* **Off at eval** (`self.training` gate) and off in `ResidualVQ.quantize` (`residual_vq.py:171-194`
  has no dropout at all), so tokenisation never drops.
* `force_dropout_index >= 0` (`residual_vq.py:122-126`) is the eval-time knob to truncate to a fixed
  depth; it bypasses the `training` gate.

Measured over 4000 training calls with `num_quant=6` (`probe`, GPU 1 of job 1476696):

| kept layers | prob=0.2 (measured / expected) | prob=1.0 (measured / expected) |
|---|---|---|
| 1 | 138 / 133 | 651 / 667 |
| 2 | 146 / 133 | 669 / 667 |
| 3 | 114 / 133 | 654 / 667 |
| 4 | 147 / 133 | 659 / 667 |
| 5 | 131 / 133 | 685 / 667 |
| 6 | 3324 / 3333 | 682 / 667 |

(with prob p, `P(keep all 6) = (1-p) + p/6`, everything else `p/6`.)

### 2.5 Decoding from indices (`residual_vq.py:64-97`, `model.py:80-88`)

`get_codes_from_indices` right-pads missing quantiser columns with `-1`
(`residual_vq.py:71-72`), gathers with the `-1`s replaced by 0 and then **masks those rows to zero**
(`residual_vq.py:81-89`) — this is exactly how a dropout-truncated code stack decodes. Output layout
is `'q b n d'` = `[Q, B, T, D]`; `forward_decoder` sums over `q` and permutes to `[B, D, T]`
(`model.py:83`).

> **Layout inconsistency to be aware of:** `RVQVAE.encode` returns
> `(code_idx [B, T, Q], all_codes [Q, B, D, T])` — its `all_codes` comes straight from the layer
> outputs, which are **NCT** (`residual_vq.py:188, 191`; verified `[6,2,512,16]`). But
> `get_codes_from_indices` returns **NTC** `[Q, B, T, D]`. The two "all codes" tensors have their last
> two axes swapped. Our `encode` should pick one convention and state it — the contract for
> `hml_phys/rvq.py` says `z_q [B, T/down, n_parts, code_dim]`, i.e. NTC and already summed over
> quantisers, which is the sane choice.

---

## 3. Training losses, optimiser, schedule

`models/vq/vq_trainer.py` (class `RVQTokenizerTrainer`) and `options/vq_option.py`.

### 3.1 Loss (`vq_trainer.py:38-54`)

```python
pred_motion, loss_commit, perplexity = self.vq_model(motions)
loss_rec      = self.l1_criterion(pred_motion, motions)                       # vq_trainer.py:45
pred_local_pos = pred_motion[..., 4 : (joints_num - 1) * 3 + 4]               # vq_trainer.py:46
local_pos      = motions[...,     4 : (joints_num - 1) * 3 + 4]               # vq_trainer.py:47
loss_explicit  = self.l1_criterion(pred_local_pos, local_pos)                 # vq_trainer.py:48
loss = loss_rec + opt.loss_vel * loss_explicit + opt.commit * loss_commit     # vq_trainer.py:50
```

* **Reconstruction norm: `SmoothL1Loss`** (Huber, default beta=1.0), selected by
  `--recons_loss l1_smooth` (`vq_option.py:25`, dispatch at `vq_trainer.py:31-34`). Plain `L1Loss` is
  the only alternative; **there is no MSE option**.
* **There is NO velocity loss**, despite the flag being named `--loss_vel` and the variable
  `loss_vel`. The term weighted by `0.5` is the **same SmoothL1 applied to the RIC local-joint-position
  slice** `[4 : 4 + (J-1)*3]` = channels 4..67 for J=22. The authors say so at `vq_trainer.py:129`:
  *"Note it not necessarily velocity, too lazy to change the name now"*. Do not port a velocity term
  on the strength of the flag name.
* **Commitment weight `0.02`** (`--commit`, `vq_option.py:23`) applied to the already-averaged
  per-layer commitment mean from `ResidualVQ` (§2.3). No codebook loss, no weight on it.
* Nothing is masked: `MotionDataset` yields fixed-length `window_size` crops, so there is no padding.

### 3.2 Optimiser and schedule (`vq_trainer.py:86-87`, `vq_trainer.py:117-125`)

```python
AdamW(params, lr=opt.lr, betas=(0.9, 0.99), weight_decay=opt.weight_decay)   # vq_trainer.py:86
MultiStepLR(opt_vq_model, milestones=opt.milestones, gamma=opt.gamma)        # vq_trainer.py:87
```

* `lr = 2e-4` (`vq_option.py:18`), `betas = (0.9, 0.99)` — **note 0.99, not the torch default 0.999**.
* `weight_decay = 0.0` (`vq_option.py:22`) — AdamW with zero decay, i.e. effectively Adam.
* **Linear warm-up over 2000 iters** (`vq_option.py:17`), implemented by hand:
  `lr = base_lr * (it + 1) / (warm_up_iter + 1)` (`vq_trainer.py:58-64`), applied while
  `it < warm_up_iter` (`vq_trainer.py:117-118`).
* `scheduler.step()` is called **per iteration**, and **only after warm-up ends**
  (`vq_trainer.py:124-125`) — so `--milestones` are in *optimiser steps after the warm-up*, not epochs.
* `milestones = [150000, 250000]` (`vq_option.py:19`), `gamma = 0.05` per README:175 (the CLI default
  at `vq_option.py:20` is `0.1`; the published command overrides it). So LR is
  2e-4 -> 1e-5 at 150k -> 5e-7 at 250k.
* **No gradient clipping** in the VQ trainer (`vq_trainer.py:120-122` is `zero_grad / backward / step`;
  `clip_grad_norm_` is imported but only used by `LengthEstTrainer`, `vq_trainer.py:250-252`).
* **No EMA of the network weights**, no AMP, single GPU (`train_vq.py:38`).
* Seed `3407` (`vq_option.py:66`).

### 3.3 Data, batch, window, iterations

* `window_size = 64` frames at 20 fps = 3.2 s (`vq_option.py:11`); `MotionDataset.__getitem__`
  (`data/t2m_dataset.py:76-87`) takes a contiguous crop whose offset is DETERMINISTIC (the global index is resolved against a cumsum table of per-clip window counts, not sampled), and z-normalises with the
  dataset mean/std.
* Clips shorter than `window_size` are **dropped entirely** (`data/t2m_dataset.py:30-31`).
* `__len__` is the **total number of crop offsets**, not the number of clips
  (`data/t2m_dataset.py:39, 73-74`).
* **`--feat_bias = 5` rescales the std** of the root channels (0:4) and the foot-contact channels
  (last 4) by 1/5 before normalising, leaving RIC/rot/velocity channels alone
  (`data/t2m_dataset.py:41-64`). The rescaled `mean.npy`/`std.npy` are written into the checkpoint's
  `meta/` dir (`data/t2m_dataset.py:63-64`) and must be reused at inference.
* `batch_size`: **README:175 says 256, README:191 says "we use 512 for rvq training"** — an unresolved
  contradiction in the upstream README. CLI default is 256 (`vq_option.py:10`).
* `max_epoch = 50` (`vq_option.py:15`); total iters `= 50 * len(train_loader)` (`vq_trainer.py:97`).

Measured on our copy of HumanML3D (`/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D`, train.txt,
`window_size=64`): 23,384 train ids -> **20,942 clips** survive the 64-frame filter, 3,188,010 frames,
**1,847,722 crop offsets**. Therefore

| batch | iters/epoch | iters for 50 epochs | milestones hit |
|---|---|---|---|
| 256 | 7,217 | **360,850** | both (150k, 250k) |
| 512 | 3,608 | **180,400** | only 150k |

Batch 256 is the setting under which the `[150000, 250000]` milestones make sense; prefer it.

---

## 4. The shipped hyper-parameter set (copy these as defaults)

From `options/vq_option.py`, README:175/191-193, and the released checkpoint name
`rvq_nq6_dc512_nc512_noshare_qdp0.2` (README:207).

| group | parameter | value | citation |
|---|---|---|---|
| arch | `input_width` | 263 (HumanML3D) / 251 (KIT) | `train_vq.py:57, 69` |
| arch | `width` | **512** | `vq_option.py:33` |
| arch | `down_t` | **2** (-> T/4) | `vq_option.py:31` |
| arch | `stride_t` | **2** (do not change; decoder hard-codes x2) | `vq_option.py:32`, `encdec.py:57` |
| arch | `depth` (resblocks per down stage) | **3** | `vq_option.py:34` |
| arch | `dilation_growth_rate` | **3** (dilations 9, 3, 1 after reversal) | `vq_option.py:35`, `resnet.py:76-79` |
| arch | `activation` | **relu** | `vq_option.py:37` |
| arch | `norm` | **None** (Identity) | `vq_option.py:39`, `resnet.py:29-30` |
| arch | resblock dropout | **0.2**, hard-coded | `resnet.py:13` |
| arch | `output_emb_width` | 512 (== `code_dim`, asserted) | `vq_option.py:36`, `model.py:23` |
| RVQ | `num_quantizers` | **6** | README:175, 192 |
| RVQ | `nb_code` | **512** | `vq_option.py:29`, ckpt name `nc512` |
| RVQ | `code_dim` | **512** | `vq_option.py:28`, ckpt name `dc512` |
| RVQ | `shared_codebook` | **False** | `vq_option.py:42`, ckpt name `noshare` |
| RVQ | `quantize_dropout_prob` | **0.2** | README:175, 193 |
| RVQ | `quantize_dropout_cutoff_index` | **0**, hard-coded | `model.py:35` |
| RVQ | `mu` (EMA decay) | **0.99** | `vq_option.py:30` |
| RVQ | `sample_codebook_temp` (train fwd) | **0.5**, hard-coded | `model.py:72` |
| RVQ | L2-normalised codebook | **no** | `quantizer.py:72-80` |
| loss | recon | **SmoothL1** on all channels | `vq_option.py:25`, `vq_trainer.py:34, 45` |
| loss | `loss_vel` (really: RIC local-position slice) | **0.5** | `vq_option.py:24`, `vq_trainer.py:46-50` |
| loss | `commit` | **0.02** | `vq_option.py:23`, `vq_trainer.py:50` |
| optim | optimiser | AdamW, betas **(0.9, 0.99)** | `vq_trainer.py:86` |
| optim | `lr` | **2e-4** | `vq_option.py:18` |
| optim | `weight_decay` | **0.0** | `vq_option.py:22` |
| optim | `warm_up_iter` | **2000**, linear | `vq_option.py:17`, `vq_trainer.py:58-64` |
| optim | `milestones` / `gamma` | **[150000, 250000] / 0.05** (per-iter steps, post-warm-up) | `vq_option.py:19`, README:175 |
| optim | grad clip | **none** | `vq_trainer.py:120-122` |
| data | `window_size` | **64** frames (20 fps) | `vq_option.py:11` |
| data | `batch_size` | **256** (README:191 says 512 — contradiction) | `vq_option.py:10`, README:175/191 |
| data | `max_epoch` | **50** | `vq_option.py:15` |
| data | `feat_bias` | **5** (std rescale on root + foot-contact) | `vq_option.py:58`, `t2m_dataset.py:41-64` |
| misc | `seed` | 3407 | `vq_option.py:66` |

---

## 5. How reconstruction quality and codebook usage are reported

### 5.1 During training (`vq_trainer.py`)

* Scalar logging every `--log_every 10` iters (`vq_option.py:53`, `vq_trainer.py:135-143`):
  `loss, loss_rec, loss_vel, loss_commit, perplexity, lr`.
* **`perplexity` is the only codebook-usage statistic MoMask logs** — the mean over quantiser layers
  (`residual_vq.py:158`) of the per-batch code-histogram perplexity `exp(-sum p log p)`
  (`quantizer.py:120-121`). **There is no "active codes" / "% codebook used" metric anywhere in the
  repo** — no unique-code count, no usage-over-epoch tracking. If we want an active-code number we
  have to add it (see the parity harness, which reports both).
* Once per epoch, full text-to-motion retrieval metrics over the reconstructions via
  `evaluation_vqvae` (`utils/eval_t2m.py:23-140`, called at `vq_trainer.py:107-111, 194-197`): FID,
  Diversity, R-precision @1/2/3 and matching score, computed by feeding **`net(motion)` output** back
  through the frozen Guo evaluator. Best-FID checkpoint saved as `net_best_fid.tar`
  (`utils/eval_t2m.py:97-102`).
* **Upstream selects on the `val` split** (`train_vq.py:116`, `get_dataset_motion_loader(..., 'val')`).
  **Our project bans `val` (project CLAUDE.md §1) — selection must be on `test`.**

### 5.2 Final reconstruction eval (`eval_t2m_vq.py`)

* Runs on the **test** split (`eval_t2m_vq.py:60`), batch 32, over every checkpoint file in the model
  dir (`eval_t2m_vq.py:70-78`).
* Reports FID / Diversity / R@1 / R@2 / R@3 / matching score, **plus MPJPE**
  (`utils/eval_t2m.py:142-224`): motions are un-normalised, `recover_from_ric` turns them into joints,
  and `calculate_mpjpe(gt, pred)` is summed over frames and divided by the frame count
  (`utils/eval_t2m.py:176-215`). The variable is printed as "MAE" in `eval_t2m_vq.py:117` but it is
  MPJPE.
* **`repeat_time = 20` with a 95% CI** (`eval_t2m_vq.py:90-117`). **Our project forbids this**
  (CLAUDE.md §4: one rollout, one metric pass, no repeats, no CIs). Port the metrics, not the
  repetition.

---

## 6. Where MoGeFlow's part-structured VQ differs

`vendor_mogeflow/` files. Note first: **`vendor_mogeflow/models/vq/*` is a byte-identical copy of
MoMask's `models/vq/*`** (verified: `diff` is empty for `model.py`, `quantizer.py`, `residual_vq.py`,
`encdec.py`, `resnet.py`). That copy is only the `momask_rvq` fallback backend
(`vendor_mogeflow/options/codeflow_options.py:18`). The tokenizer MoGeFlow actually uses is a
**different, non-residual, part-structured VQ** living in `vendor_mogeflow/kvctrl/models/`.

### 6.1 Structure: parts instead of residual depth

`vendor_mogeflow/kvctrl/models/vqvae.py:208-427`, class `VQVAE_251`:

* **One encoder per body part** (`vqvae.py:266-274`), each taking only that part's channel subset,
  and **one codebook per part** (`vqvae.py:304-306`), but **a single whole-body decoder**
  (`vqvae.py:280-287`) whose input width is `output_emb_width * num_parts` — the per-part quantised
  latents are **concatenated along channels** (`vqvae.py:420`), not summed.
* **`num_quantizers` is effectively 1**: the quantiser is a bare `QuantizeEMAReset`
  (`vqvae.py:292-306`), there is **no residual stack and no quantise-dropout**. Depth is traded for
  breadth: 6 parts x 1 layer instead of 1 stream x 6 layers.
* Code grid is `[B, T_latent, P]` with `P = 6`; the model's own `encode` flattens it **part-major** to
  `[B, P*T_latent]` (`vqvae.py:336-360`), and `models/codeflow/kv_vq.py:64-79` converts between that
  flat layout and `[B, T, P]`. A real source of transposition bugs.
* Loss/perplexity are **averaged over parts** (`vqvae.py:426-427`), the same shape of aggregation
  MoMask applies over quantiser layers.
* The quantiser itself is the pre-MoMask version (`kvctrl/models/quantize_cnn.py:6-131`): identical
  EMA + `_tile` restart + commitment + straight-through, but **`torch.min(distance)` argmin
  (`quantize_cnn.py:89`) with no Gumbel option at all** — no `sample_codebook_temp`.
* The encoder feature **normalisation** (`x_feature / ||x_feature||`) that the ancestor
  `VQVAE_limb_hml` applied is **commented out and explicitly not used**: "不做 norm，与训练版本一致"
  (`vqvae.py:344-345`, `vqvae.py:412`).
* `kvctrl/models/encdec.py` adds a `stride_t == 1` special case (`filter_t, pad_t = 3, 1`) that
  MoMask lacks; everything else in enc/dec/resnet matches.

### 6.2 The released partition: six parts, **overlap**

* `README.md:121-123`: the released bundle contains
  `rvq/part_vq_hml3d_overlap_best_top3.pth` — a **frozen** part-aware VQ tokenizer, "one codebook per
  joint group" — and `rvq/skeleton_partition.json`, "six-part **overlap** partition".
* The partition is **data loaded from JSON**, not hard-coded: `load_partition_from_file`
  (`vqvae.py:198-205`) reads `data["partSeg"]`, and `partition_file` overrides the built-in list
  (`vqvae.py:242-245`). The built-in fallback (`vqvae.py:247-261`) is the disjoint
  root / spine / L-arm / R-arm / L-leg / R-leg split of the 263-d HumanML3D vector
  (with foot-contact channels 259-260 attached to the left leg and 261-262 to the right).
  The *released* `skeleton_partition.json` is the overlapping variant — channels are shared between
  neighbouring parts, so `sum(len(part)) > 263` and the parts are **not** a partition.
* The JSON is not in the repo (it ships with the HF checkpoint bundle
  `AmberJar/CodeFlow-HumanML3D`), so the exact overlap sets are not inspectable from source.

### 6.3 MoGeFlow's part-VQ hyper-parameters

`vendor_mogeflow/models/codeflow/kv_vq.py:17-55` (`VQ_CFG`, the frozen tokenizer's training config)
and `scripts/launch/train_humanml3d_pscf_standard.sh:35-37`:

| parameter | MoGeFlow part-VQ | MoMask RVQ | citation |
|---|---|---|---|
| quantiser layers | **1** (per part) | 6 (residual) | `vqvae.py:304-306` |
| parts | **6**, one codebook each | 1 | `launch:36`, `vqvae.py:261` |
| `nb_code` | **128** | 512 | `kv_vq.py:31`, `launch:37` |
| `code_dim` / `output_emb_width` | **128** | 512 | `kv_vq.py:30, 38`, `launch:35` |
| `width` | 512 | 512 | `kv_vq.py:35` |
| `down_t` / `stride_t` | 2 / 2 | 2 / 2 | `kv_vq.py:33-34` |
| `depth` / `dilation_growth_rate` | 3 / 3 | 3 / 3 | `kv_vq.py:36-37` |
| quantise-dropout | **none** | 0.2 | — |
| code assignment | argmin, no Gumbel | Gumbel @ T=0.5 in train fwd | `quantize_cnn.py:89` |
| `mu` | 0.99 | 0.99 | `kv_vq.py:32` |
| `commit` / `loss_vel` / recon | 0.02 / 0.5 / l1_smooth | same | `kv_vq.py:27-29` |
| `lr` / warm-up / schedule | 2e-4 / 1000 / step at [200000], gamma 0.05 | 2e-4 / 2000 / [150000, 250000], gamma 0.05 | `kv_vq.py:22-25` |
| batch / window / iters | 256 / 64 / 300,000 | 256 / 64 / 50 epochs | `kv_vq.py:19-21` |

So the concatenated latent MoGeFlow hands to its flow is `6 x 128 = 768` channels per latent frame,
against MoMask's single 512-d summed latent with a 6-deep code stack.

**关于「MoGeFlow 是六层」的澄清（2026-09-21）**：MoGeFlow 的 code 轴确实是 6，但在已发布配方里这个 6 是
**6 个部位**，不是 6 层残差。`README.md:170-176` 的标准配方写死 `tokenizer backend = frozen KV-Control
PartVQ / num_groups 6 / num_codes 128 per group / code_dim 128`，而 `kvctrl/models/vqvae.py:292-306`
的 `build_single_quantizer()` 每个部位只建**一个** `QuantizeEMAReset`，整个 `kvctrl/` 里没有
`num_quantizers` / `ResidualVQ` 字样；`gen_codeflow_t2m.py:147` 也把发布推理路径写死成
`opt.vq_backend = "kv_part"`。另一条 `momask_rvq` 后端（`models/codeflow/vq_tokenizers.py:28-35`）才是
六层残差：`models/codeflow/momask_vq.py:112` 把 `num_parts = args.num_quantizers`，即把 MoMask 的
6 层残差直接塞进 CodeFlow 的 6 个 group 槽位（MoMask 发布值 nb_code 512 / code_dim 512 /
num_quantizers 6）。两种读法都给出 6，但含义不同；我们的设计是**部位宽度 × 残差深度**（6 部位 x 6 层），
是两者的并集。

### 6.4 What MoGeFlow's CodeFlow actually trains on

`vendor_mogeflow/scripts/launch/train_humanml3d_pscf_standard.sh`:

```
--terminal_loss_weight 0.0     # line 62
--clean_loss_weight    0.0     # line 63
```

Both auxiliary terms are **switched off in the standard recipe**: the training objective is the
**flow-matching (velocity) loss only**. The terminal head is still built (`--terminal_mode tied_logits`,
`--terminal_tau_mode codebook_nn`, lines 60-61) and is used at sampling time to snap to codes, but it
receives no training signal in this recipe. The tokenizer is **frozen** throughout — it is loaded from
`--vq_checkpoint` (line 28) via `PartVQTokenizer` (`models/codeflow/kv_vq.py:86-`), and README:121
calls it a "frozen part-aware VQ tokenizer".

Other recipe facts worth having: `--latent_norm_mode codebook` (line 59, latents normalised by
codebook statistics), `--cond_drop_prob 0.1` (line 56) for CFG, `--disable_self_condition` (57),
`--time_schedule uniform` (58), AdamW `lr 1e-4`, `half_cosine` to 1% of peak, 2000 warm-up steps,
`weight_decay 0.01`, `grad_clip 1.0`, bf16 AMP, batch 64, 600 epochs (lines 45-54), and full eval at
96 sampling steps, `cond_scale 6.0`, `repeat_times 1` (lines 69-71).

---

## 7. Where the part-structured setting would deviate from MoMask (RETIRED — reference only)

> **Not in use.** The user settled the architecture on 2026-09-21: MoMask's original whole-body RVQ, no part
> axis (`--structure whole`, see §9). Everything in this section applies only to `--structure part`, which is kept
> in the code for reference. Read §9 for what is actually trained.

Our token is the 435-d physics token of `hml_phys/tokens.py`, split into 6 **disjoint** parts
(`hml_phys/tokens.py:313-356`) with dims **[21, 90, 90, 90, 72, 72]** (root / spine / L-arm / R-arm /
L-leg / R-leg), summing to 435. `PartRVQVAE(input_dim, part_channels, ...)` combines MoMask's
*residual depth* with MoGeFlow's *part breadth*, which forces these departures:

1. **Per-part encoders, shared-shape but not shared-weight.** MoMask has one encoder over all 263
   channels; we need one per part because `part_channels` have different widths (21 vs 90). Follow
   MoGeFlow (`vqvae.py:266-274`): `Encoder(len(part), code_dim, ...)` per part. MoMask's
   `assert output_emb_width == code_dim` still applies per part.
2. **Decoder: concatenate, don't sum, across parts.** Within a part, the `n_quant` residual codes sum
   (MoMask). Across parts, the `n_parts` latents concatenate on the channel axis, so the decoder input
   width is `n_parts * code_dim` (MoGeFlow `vqvae.py:280-287, 420`). With a disjoint partition a
   per-part decoder writing back into its own channels is also defensible and cheaper; a single
   whole-body decoder is the MoGeFlow-faithful choice and lets parts correct each other. **This is a
   design fork the lead has to pick — it is not determined by either reference.**
3. **`n_parts * n_quant` codebooks.** MoMask's `shared_codebook` shares across *quantiser layers*
   (`residual_vq.py:43-45`). With parts there are two axes to share over. The contract's single
   `shared_codebook` flag should mean "share across quantiser layers within a part" (MoMask
   semantics); sharing across parts would tie a 21-d root latent to a 90-d arm latent's codes and is
   almost certainly wrong.
4. **Quantise-dropout granularity.** MoMask draws once per forward for the whole batch and the whole
   (single) stream. With parts, dropping the same depth in every part keeps the parity with MoMask;
   dropping per-part independently is a different (and untested upstream) regime. Recommend: one draw
   per forward, applied to all parts, to stay MoMask-equivalent, and expose the kept-depth in the
   losses dict so the parity harness can check it.
5. **No RIC-position auxiliary loss.** MoMask's `0.5 * SmoothL1` on channels `[4 : 4+(J-1)*3]`
   (`vq_trainer.py:46-48`) is HumanML3D-specific. The analogous slice in our token is
   `local_positions` (channels 15..86, `hml_phys/tokens.py:BODY_SLICES`). Either retarget it there or
   drop it — but do not port the channel indices `4 : 67` literally.
6. **Channel scaling.** MoMask's `--feat_bias 5` down-weights root and foot-contact std
   (`t2m_dataset.py:41-64`). Our token mixes positions, 6-D rotations, velocities and **actions** with
   very different scales; whatever normalisation `hml_phys` already uses for the flow must be reused
   here, and the SmoothL1 `beta=1.0` assumption (errors below 1.0 are quadratic) should be checked
   against our normalised magnitudes.
7. **Codebook init needs per-part data.** `init_codebook` seeds from the first training batch's
   encoder features (`quantizer.py:60-65`). With `nb_code=512` and a 21-d root part, a batch of
   `B*T_latent` rows must be >= 512 or `_tile` will duplicate rows with only 0.01/sqrt(d) noise
   between them. With `batch=256, window=64, down_t=2` we get `256*16 = 4096` rows per part — fine,
   but worth an assert.
8. **Selection split and eval repeats.** `val` is banned (CLAUDE.md §1) and evaluation is single-pass
   (CLAUDE.md §4). Port `evaluation_vqvae` / MPJPE, drop `repeat_time=20` and the CIs, and select on
   `test`.
9. **Active-code reporting.** MoMask reports only perplexity. For a part-structured tokenizer with 6
   separate codebooks, per-part **active-code counts** (unique indices used per epoch) are the
   diagnostic that actually catches a collapsed part; the contract's "per-part usage stats" should
   carry both perplexity and active-code count per part.

---

## 8. Parity harness

`scripts/hml_phys/rvq_parity_test.py`. Run it on a compute node:

```
srun --jobid=<job> --overlap -n1 bash -lc 'source scripts/activate_uniphys.sh >/dev/null 2>&1; \
  cd /iridisfs/scratch/pf2m24/projects/motion_rebot; export CUDA_VISIBLE_DEVICES=1; \
  python scripts/hml_phys/rvq_parity_test.py'
```

It imports `hml_phys.rvq.PartRVQVAE` (skips cleanly with exit code 0 if the module is not there yet),
builds the MoMask `RVQVAE` from `vendor_momask/`, **fits both briefly with MoMask's objective** (an
unfitted autoencoder reconstructs nothing, which makes the residual and dropout checks vacuous), and
runs:

1. **Weight-transfer parity** — a single part covering all channels, same hyper-parameters, MoMask's
   weights copied in; reconstruction, codes, `z_q` and commitment must agree to 1e-5 in eval mode.
   Also reports a **residual-block dilation-order diff**, because the dilation order changes the
   numbers without changing any weight shape, so a mismatch copies cleanly and then silently fails.
2. **Residual property** — reconstruction error must not increase as quantiser layers are added
   (with a vacuity guard against a model that does not reconstruct at all, and the same measurement
   on the reference as a control).
3. **Code validity and round-trip determinism** — integer codes in `[0, nb_code)`, `encode` and
   `decode` deterministic, `decode(encode(x))` equal to `forward(x)`, `decode` accepting `z_q`.
4. **Quantise-dropout** — off at eval, off in `encode()` even in train mode (MoMask's tokenisation
   path never drops), never fires at `prob=0`, and at `prob=1` the kept depth is uniform on
   `{1..n_quant}` (chi-square), against the reference distribution of §2.4. The kept depth is read
   from the losses dict (`perplexity_per_layer` records 0 for a dropped layer) with an
   error-mode-classification fallback that is first calibrated on the reference.
4b. **Dropout draw granularity** — builds a 2-part model and checks whether both parts drop the same
   depth. MoMask draws once per forward; independent per-part draws raise the effective dropout rate
   from `p` to `1 - (1-p)^n_parts`.
5. **Train-time behaviours eval parity cannot see** — whether train-mode code assignment is
   stochastic (MoMask's Gumbel at temperature 0.5, `model.py:72`) and whether the residual blocks
   carry MoMask's `Dropout(0.2)`.

The weight-copy helper matches parameters by their **structural suffix** (`model.<i>....` after the
first `encoder`/`decoder` prefix and any part/layer index) rather than by exact name, so it tolerates
whatever the attribute names turn out to be; it also primes EMA bookkeeping (`code_sum`,
`code_count`, `inited`) whether those live in `state_dict` or, as in MoMask, as plain attributes. If
it cannot resolve a mapping it fails loudly with the unmatched keys rather than silently comparing
garbage.

### 8.1 Parity results against `hml_phys/rvq.py` (2026-09-21)

**Current status: 9 of 9 checks pass, `x_rec max|diff| = 0.00e+00` after weight transfer**, codes
identical, commitment identical (re-run 2026-09-21 18:0x on a compute node, `--trials 300`; the
harness configuration is a scaled-down one — `n_quant=3 / nb_code=64 / code_dim=32 / width=64 /
depth=2` — so structural equivalence is proved, while `nb_code=2048`'s codebook initialisation and
EMA statistics are exercised only by the live training).

```
0b construct   PASS  233.0k params vs reference 233.0k
0c weight xfer PASS  50 enc/dec tensors + 3 codebooks + 9 EMA buffers
1  transfer    PASS  x_rec max|diff| 0.00e+00, codes identical, commit 1.402534
2  residual    PASS  k=1 6.33e-1 -> k=2 4.87e-1 -> k=3 4.46e-1
3  codes/round PASS  decode(encode(x)) == forward(x) to 1e-6
4  q-dropout   PASS  p=1 -> {1:111, 2:92, 3:97} vs reference {1:84, 2:108, 3:108}
4b draw shared PASS  200 forwards, both parts always the same depth
5  train-time  PASS  jitter 5.63e-1 (reference 5.80e-1); 8 active Dropouts (reference 8)
```

The first run (earlier the same day) was 7 of 9. All four defects it found are **fixed**:

* encoder residual-block dilation order `1,3,9` -> `9,3,1` (`Resnet1D` now defaults to
  `reverse_dilation=True`, `rvq.py:39-47`; matches `resnet.py:73` + `encdec.py:27`);
* per-part dropout draws -> **one draw per forward** shared by every group
  (`PartRVQVAE.forward`, `rvq.py:249`; matches `residual_vq.py:112`);
* deterministic `argmin` at train time -> **Gumbel sampling at temperature 0.5**
  (`QuantizeEMAReset.forward`, `rvq.py:116-118`; matches `model.py:72`, `quantizer.py:78`).
  `encode()` still goes through `ResidualVQ.quantize`, which is pure argmin and updates nothing;
* missing `Dropout(0.2)` in the residual blocks -> added (`rvq.py:32,35`).

Deliberate improvements over MoMask that the harness accommodates rather than flags: `code_sum` /
`code_count` / `inited` registered as buffers so the EMA state survives a checkpoint (required for
`--resume`); out-of-place residual subtraction; `Upsample(scale_factor=stride_t)` instead of a
hard-coded 2; random rather than first-`nb_code` rows for codebook init and dead-code restart (the
first 2048 rows of a batch are 128 consecutive clips x 16 frames, i.e. heavily correlated); an
assertion that the first batch holds at least `nb_code` rows where MoMask silently tiles;
`ResidualVQ.quantize` bypassing `layer.forward` so tokenising can never move a codebook. These make
the *training trajectory* non-reproducible against MoMask bit-for-bit; the structure and the
objective are equivalent.

---

## 9. The configuration actually in use (2026-09-21, settled by the user)

> "就用 momask 的 rvq 版本，然后维度调整到 2048，不要走 mogeflow 的 rvq" — and, on the follow-up
> question, **2048 is the codebook size `nb_code`**, with `code_dim` left at MoMask's 512.

### 9.1 Architecture

MoMask's original whole-body RVQ-VAE: **one encoder over every channel of the variant, one
`n_quant=6` residual quantiser, one decoder**. There is no part axis. `scripts/hml_phys/train_rvq.py
--structure whole` (the default) collapses `parts` to a single group `arange(n_channels)`
(`train_rvq.py:66-67`), which makes `PartRVQVAE`'s `part_index` / `inverse_index` identity
permutations and its three `ModuleList`s length 1. `--structure part` keeps the retired
part-structured variant (§7) available.

| | MoMask release | ours |
|---|---|---|
| groups x residual layers | 1 x 6 | **1 x 6** |
| `nb_code` | 512 | **2048** |
| `code_dim` / `output_emb_width` | 512 | 512 |
| `width` / `down_t` / `stride_t` / `depth` / `dilation_growth_rate` | 512 / 2 / 2 / 3 / 3 | same |
| `quantize_dropout_prob` / cutoff / `mu` / train sampling temperature | 0.2 / 0 / 0.99 / 0.5 | same |
| loss | SmoothL1 + 0.5 x SmoothL1(geometric) + 0.02 x commit | same |
| optimiser | AdamW 2e-4, betas (0.9, 0.99), wd 0, 2000-step linear warm-up | same |
| input width | 263 (HumanML3D) | 435 / 366 / 69 |
| parameters | 19.44 M | **20.0 / 19.8 / 18.8 M** |

The parameter difference is exactly the first and last 3-kernel convolutions:
`(435-263) x 512 x 3 x 2 = 0.53 M`. 64-frame windows become 16 latent frames x 6 layers x 2048 entries.

### 9.2 Deliberate deviations from MoMask's training recipe (recorded, not accidental)

* **Budget: 200,000 iterations, not 50 epochs (~360,850 iterations at batch 256).** Roughly 37.6
  epochs over our 1,361,300 training windows. Chosen to fit the three variants into one allocation
  alongside the 1M-step policy training; measured 22-25 ms/it, so ~80 minutes each.
* **`milestones = [150000]`, not `[150000, 250000]`.** The second milestone is unreachable inside a
  200k budget, so writing it changes nothing — but the *effect* is a real deviation: MoMask decays at
  42% of its run and then spends 100k steps at 1e-5 and 110k at 5e-7, while we decay at 75% and get
  only 50k low-LR steps, with no second decay at all.
* **Checkpoint selection is test-split normalised MSE**, not MoMask's val-split FID. The split is
  forced by project CLAUDE.md §1 (val is banned); the metric is a simplification — `rvq_eval.py`
  reports MPJPE-style `local_positions.mm`, but no FID / R-precision.
* **No `feat_bias`.** MoMask scales root and foot-contact channels by 1/5 (`vq_option.py`); we
  normalise with the project's `token_stats_v3` instead.
* **Normalisation statistics are the policy's, not the tokenizer's.** `token_stats_v3.npz` was fitted
  on *policy* windows, whose canonical origin is the newest history frame; RVQ windows are 64
  contiguous frames with origin at frame 0. The origin-dependent channels (`root_trans` 3,
  `root_rot_6d` 6, i.e. 9 of 435) are therefore normalised against the wrong mean and a scale that is
  off by roughly 2x — estimated normalised mean offset about +0.5 for `root_trans_y`. Kept on
  purpose: SmoothL1 acts on the *difference*, so a mean offset never enters the loss, and the scale
  error only over-weights 9 channels. Not worth discarding a run to fix.

### 9.3 Downstream interface (for the VQ-only CodeFlow policy and the VQ + intent-VAE version)

* `hml_phys.rvq.load_rvq(path, device, freeze=True) -> (model, args)` rebuilds a trained tokenizer
  from a checkpoint with a **strict** `load_state_dict` (a silent mismatch would leave random
  codebooks and still produce plausible numbers).
* `PartRVQVAE.freeze()` is mandatory before embedding the tokenizer. `requires_grad_(False)` is *not*
  sufficient: `QuantizeEMAReset` rewrites its codebook buffers inside `@torch.no_grad()`, so one
  `rvq(x)` in train mode moves the codebook (measured 0.42 in codebook L2; 0.0 under `eval()` or when
  only `encode()` is called). `freeze()` sets eval mode, drops gradients, disables quantise-dropout
  and neutralises a later `.train()`.
* `PartRVQVAE.codebooks() -> [n_parts, n_quant, nb_code, code_dim]` for `latent_norm_mode=codebook`
  and nearest-code snapping.
* `encode(x, n_layers=k) -> dict(codes [B, T/4, 1, 6], z_q [B, T/4, 1, 512])`; `decode(codes=)` and
  `decode(z_q=)` agree with `forward` to ~5e-7 in fp32. The group axis is a singleton under
  `--structure whole` and has to be squeezed downstream.
* **Open design fork (needs a decision before the policy is written): the canonical origin.** RVQ
  windows use their own frame 0; the policy and the intent VAE use the newest history frame. Feeding
  policy-canonicalised windows to the frozen tokenizer measured **+29% reconstruction MSE**
  (0.1896 -> 0.2444). The CodeFlow-native answer is to let each generated chunk carry its own frame
  and compose the transform on the policy side (the v3 local-root bridge already does this); the
  alternatives are a re-normalisation adapter, or retraining the tokenizer in the policy's frame.
* Not yet built: a dataset that yields the RVQ codes and the intent latent for the same window, and a
  latent-statistics script analogous to `scripts/hml_phys/compute_intent_latent_stats.py`.

### 9.4 Canonical-origin decision (user, 2026-09-21: "就做我们原生的做法")

**Settled: each generated chunk carries its own canonical frame, and the policy side composes the
transform** (the v3 local-root bridge already does exactly this). The tokenizer is NOT retrained and
no re-normalisation adapter is added; a chunk handed to the frozen tokenizer is canonicalised on its
own first frame, which is precisely the distribution `RVQWindowDataset` was trained on.

One consequence worth stating, because it removes the problem entirely for version 1: **the `action`
variant is origin-independent.** Its 69 channels are per-DOF PD targets, which carry no root frame at
all, so the +29% figure measured earlier applies only to the `token` / `state` variants (whose
`root_trans` / `root_rot_6d` channels do depend on the origin). A CodeFlow policy that generates
action codes is therefore unaffected by the choice; only the state-side of version 2 has to respect it.
