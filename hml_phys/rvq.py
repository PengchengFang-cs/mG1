"""Residual VQ-VAE over our physics tokens (docs/08).

A port of MoMask's RVQ-VAE (vendor_momask @ EricGuo5513/momask-codes:
models/vq/{model,encdec,resnet,residual_vq,quantizer}.py).

**The configuration in use (user, 2026-09-21: "就用 momask 的 rvq 版本…不要走 mogeflow 的 rvq") is MoMask's
original, whole-body one**: `part_channels` is a single group holding every channel of the variant, so this class
collapses to one encoder + one `n_quant`-layer ResidualVQ + one decoder. `scripts/hml_phys/train_rvq.py
--structure whole` (the default) builds exactly that; the three trained tokenizers are 20.0M / 19.8M / 18.8M
parameters against MoMask's own 19.44M. MoGeFlow's part-structured VQ is NOT used.

  per group g:  channels_g -> Conv1d stack with `down_t` stride-2 stages (4x temporal downsampling)
                           -> ResidualVQ: `n_quant` EMA codebooks applied to the running residual
                           -> mirrored decoder -> channels_g
`shared_codebook` shares one codebook across the residual layers, as in MoMask.

Passing more than one group keeps the part-structured variant alive (`--structure part`), which is retained only
for reference. Channel variants (see hml_phys/rvq_data.py):
  action : the 69 PD-action channels only
  token  : all 435 channels
  state  : the 366 non-action channels
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- MoMask models/vq/resnet.py
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="relu", dropout=0.2):
        super().__init__()
        self.activation1 = nn.ReLU() if activation == "relu" else nn.SiLU()
        self.activation2 = nn.ReLU() if activation == "relu" else nn.SiLU()
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)      # MoMask's only regulariser (weight decay is 0)

    def forward(self, x):
        h = self.dropout(self.conv2(self.activation2(self.conv1(self.activation1(x)))))
        return x + h


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=3, reverse_dilation=True, activation="relu"):
        """MoMask's default is reverse_dilation=True: both the encoder and the decoder run dilations 9, 3, 1."""
        super().__init__()
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** d, activation=activation)
                  for d in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------- MoMask models/vq/encdec.py
class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation=3, activation="relu"):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks = [nn.Conv1d(in_dim, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                                        Resnet1D(width, depth, dilation, activation=activation)))
        blocks.append(nn.Conv1d(width, out_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim=512, down_t=2, stride_t=2, width=512, depth=3, dilation=3, activation="relu"):
        super().__init__()
        blocks = [nn.Conv1d(emb_dim, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(Resnet1D(width, depth, dilation, reverse_dilation=True, activation=activation),
                                        nn.Upsample(scale_factor=stride_t, mode="nearest"),
                                        nn.Conv1d(width, width, 3, 1, 1)))
        blocks += [nn.Conv1d(width, width, 3, 1, 1), nn.ReLU(), nn.Conv1d(width, out_dim, 3, 1, 1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------- MoMask models/vq/quantizer.py (QuantizeEMAReset)
class QuantizeEMAReset(nn.Module):
    """EMA codebook with dead-code reset, exactly MoMask's: nearest-neighbour assignment, straight-through gradient,
    commitment = MSE(encoder output, detached code), codebook updated by EMA (mu) and unused entries replaced by
    tiled/jittered encoder outputs. code_sum / code_count are buffers so a checkpoint restores the EMA state."""
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code, self.code_dim, self.mu = nb_code, code_dim, mu
        self.register_buffer("codebook", torch.zeros(nb_code, code_dim))
        self.register_buffer("code_sum", torch.zeros(nb_code, code_dim))
        self.register_buffer("code_count", torch.zeros(nb_code))
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool))

    def _tile(self, x):
        n, d = x.shape
        if n < self.nb_code:
            out = x.repeat((self.nb_code + n - 1) // n, 1)
            return out + torch.randn_like(out) * (0.01 / np.sqrt(d))
        return x

    @torch.no_grad()
    def init_codebook(self, x):
        tiled = self._tile(x)
        out = tiled[torch.randperm(tiled.shape[0], device=x.device)][:self.nb_code]
        self.codebook.copy_(out); self.code_sum.copy_(out); self.code_count.fill_(1.0)
        self.inited.fill_(True)

    def quantize(self, x, temperature=0.0):
        """nearest code; with temperature > 0 in training mode, MoMask's Gumbel sampling over -distance."""
        k = self.codebook.t()
        d = (x ** 2).sum(-1, keepdim=True) - 2 * (x @ k) + (k ** 2).sum(0, keepdim=True)
        if self.training and temperature > 0:
            u = torch.zeros_like(d).uniform_(0, 1)          # MoMask quantizer.py:14-16: g = -log(-log(u))
            gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
            return ((-d) / temperature + gumbel).argmax(-1)
        return d.argmin(-1)

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        onehot.scatter_(0, code_idx.view(1, -1), 1)
        code_sum, code_count = onehot @ x, onehot.sum(-1)
        tiled = self._tile(x)
        rand = tiled[torch.randperm(tiled.shape[0], device=x.device)][:self.nb_code]
        self.code_sum.mul_(self.mu).add_(code_sum, alpha=1 - self.mu)
        self.code_count.mul_(self.mu).add_(code_count, alpha=1 - self.mu)
        usage = (self.code_count[:, None] >= 1.0).float()
        self.codebook.copy_(usage * (self.code_sum / self.code_count[:, None].clamp_min(1e-8)) + (1 - usage) * rand)
        prob = code_count / code_count.sum().clamp_min(1e-8)
        return torch.exp(-(prob * (prob + 1e-7).log()).sum())

    @torch.no_grad()
    def perplexity(self, code_idx):
        cnt = torch.bincount(code_idx.view(-1), minlength=self.nb_code).float()
        prob = cnt / cnt.sum().clamp_min(1e-8)
        return torch.exp(-(prob * (prob + 1e-7).log()).sum())

    def forward(self, x, temperature=0.5):
        """x [N, C, T] -> quantised [N, C, T], code indices [N, T], commitment loss, perplexity.
        MoMask assigns with Gumbel sampling at temperature 0.5 while training (models/vq/model.py:72)."""
        N, C, T = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, C)
        if self.training and not bool(self.inited):
            assert flat.shape[0] >= self.nb_code, f"codebook init needs >= {self.nb_code} rows, got {flat.shape[0]}"
            self.init_codebook(flat.detach())
        idx = self.quantize(flat, temperature)
        q = F.embedding(idx, self.codebook)
        perp = self.update_codebook(flat.detach(), idx) if self.training else self.perplexity(idx)
        commit = F.mse_loss(flat, q.detach())
        q = flat + (q - flat).detach()                       # straight-through
        return q.view(N, T, C).permute(0, 2, 1).contiguous(), idx.view(N, T), commit, perp


class ResidualVQ(nn.Module):
    """MoMask models/vq/residual_vq.py: `n_quant` codebooks applied to the running residual; at training time, with
    probability `dropout_prob`, a random suffix of the layers is dropped (their indices recorded as -1)."""
    def __init__(self, n_quant=6, nb_code=512, code_dim=512, mu=0.99, shared_codebook=False, dropout_prob=0.2,
                 dropout_cutoff=0):
        super().__init__()
        self.n_quant, self.dropout_prob, self.dropout_cutoff = n_quant, dropout_prob, dropout_cutoff
        if shared_codebook:
            layer = QuantizeEMAReset(nb_code, code_dim, mu)
            self.layers = nn.ModuleList([layer] * n_quant)
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(nb_code, code_dim, mu) for _ in range(n_quant)])

    def draw_drop_from(self):
        """MoMask draws the quantise-dropout depth ONCE per forward of the whole model; PartRVQVAE therefore draws it
        here and passes the same depth to every part (drawing per part would raise the rate to 1-(1-p)^n_parts)."""
        if self.training and self.dropout_prob > 0 and torch.rand(()).item() < self.dropout_prob:
            return int(torch.randint(self.dropout_cutoff, self.n_quant, ()).item()) + 1
        return self.n_quant

    def forward(self, x, drop_from=None, temperature=0.5):
        """x [N, C, T] -> (quantised sum, indices [N, T, n_quant] (-1 where dropped), mean commitment,
        mean perplexity, per-layer perplexity [n_quant], per-layer commitment [n_quant]).
        The two per-layer vectors hold 0 for the layers dropped this step, so their shape is always n_quant."""
        N, C, T = x.shape
        residual, out = x, torch.zeros_like(x)
        idxs, losses, perps = [], [], []
        null_idx = torch.full((N, T), -1, dtype=torch.long, device=x.device)
        drop_from = self.draw_drop_from() if drop_from is None else drop_from
        perp_layers, commit_layers = x.new_zeros(self.n_quant), x.new_zeros(self.n_quant)
        for i, layer in enumerate(self.layers):
            if i >= drop_from:
                idxs.append(null_idx)
                continue
            q, idx, commit, perp = layer(residual, temperature)
            residual = residual - q.detach()
            out = out + q
            idxs.append(idx); losses.append(commit); perps.append(perp)
            perp_layers[i] = perp.detach(); commit_layers[i] = commit.detach()
        return (out, torch.stack(idxs, -1), torch.stack(losses).mean(),
                torch.stack(perps).mean(), perp_layers, commit_layers)

    @torch.no_grad()
    def quantize(self, x, n_layers=None):
        """deterministic encode with the first `n_layers` codebooks (default: all)."""
        n = self.n_quant if n_layers is None else n_layers
        residual, out, idxs = x, torch.zeros_like(x), []
        for layer in self.layers[:n]:
            flat = residual.permute(0, 2, 1).reshape(-1, residual.shape[1])
            idx = layer.quantize(flat)
            q = F.embedding(idx, layer.codebook).view(residual.shape[0], residual.shape[2], -1).permute(0, 2, 1)
            residual = residual - q
            out = out + q
            idxs.append(idx.view(residual.shape[0], residual.shape[2]))
        return out, torch.stack(idxs, -1)

    @torch.no_grad()
    def dequantize(self, idxs):
        """idxs [N, T, q] (-1 = layer dropped) -> summed code embeddings [N, C, T]."""
        out = None
        for i, layer in enumerate(self.layers[:idxs.shape[-1]]):
            idx = idxs[..., i]
            q = F.embedding(idx.clamp_min(0), layer.codebook) * (idx >= 0).float()[..., None]
            out = q if out is None else out + q
        return out.permute(0, 2, 1).contiguous()


class PartRVQVAE(nn.Module):
    def __init__(self, input_dim, part_channels, width=512, down_t=2, depth=3, dilation=3, code_dim=512,
                 nb_code=512, n_quant=6, shared_codebook=False, dropout_prob=0.2, mu=0.99, activation="relu"):
        """part_channels: list of 1-D integer arrays, the channel indices of each part inside the `input_dim` token."""
        super().__init__()
        self.input_dim, self.n_parts, self.down = input_dim, len(part_channels), 2 ** down_t
        self.code_dim, self.nb_code, self.n_quant = code_dim, nb_code, n_quant
        self.dims = [len(c) for c in part_channels]
        self.register_buffer("part_index", torch.from_numpy(np.concatenate(part_channels)).long(), persistent=False)
        self.slices = np.cumsum([0] + self.dims).tolist()
        self.register_buffer("inverse_index", torch.from_numpy(np.argsort(np.concatenate(part_channels))).long(),
                             persistent=False)
        self.encoders = nn.ModuleList([Encoder(d, code_dim, down_t, 2, width, depth, dilation, activation) for d in self.dims])
        self.decoders = nn.ModuleList([Decoder(d, code_dim, down_t, 2, width, depth, dilation, activation) for d in self.dims])
        self.quantizers = nn.ModuleList([ResidualVQ(n_quant, nb_code, code_dim, mu, shared_codebook, dropout_prob)
                                         for _ in self.dims])

    def to_parts(self, x):
        g = x.index_select(-1, self.part_index)
        return [g[..., self.slices[i]:self.slices[i + 1]] for i in range(self.n_parts)]

    def from_parts(self, parts):
        return torch.cat(parts, -1).index_select(-1, self.inverse_index)

    def forward(self, x):
        """x [B, T, D] normalised tokens -> (reconstruction [B, T, D], dict of losses / stats)."""
        rec, commits, perps, per_layer, commit_layer = [], [], [], [], []
        drop_from = self.quantizers[0].draw_drop_from()      # one draw per forward, shared by all parts (MoMask)
        for i, xp in enumerate(self.to_parts(x)):
            h = self.encoders[i](xp.permute(0, 2, 1).float())
            q, _, commit, perp, perp_layers, commit_layers = self.quantizers[i](h, drop_from=drop_from)
            rec.append(self.decoders[i](q).permute(0, 2, 1))
            commits.append(commit); perps.append(perp)
            per_layer.append(perp_layers); commit_layer.append(commit_layers)
        return self.from_parts(rec), dict(commit=torch.stack(commits).mean(), perplexity=torch.stack(perps).mean(),
                                          commit_per_part=torch.stack(commits), perplexity_per_part=torch.stack(perps),
                                          perplexity_per_layer=torch.stack(per_layer),
                                          commit_per_layer=torch.stack(commit_layer))

    @torch.no_grad()
    def encode(self, x, n_layers=None):
        """-> dict(codes [B, T/down, n_parts, n_quant], z_q [B, T/down, n_parts, code_dim])."""
        codes, zq = [], []
        for i, xp in enumerate(self.to_parts(x)):
            h = self.encoders[i](xp.permute(0, 2, 1).float())
            q, idx = self.quantizers[i].quantize(h, n_layers)
            codes.append(idx); zq.append(q.permute(0, 2, 1))
        return dict(codes=torch.stack(codes, 2), z_q=torch.stack(zq, 2))

    @torch.no_grad()
    def decode(self, codes=None, z_q=None):
        """codes [B, T', n_parts, n_quant] or z_q [B, T', n_parts, code_dim] -> tokens [B, T, D]."""
        rec = []
        for i in range(self.n_parts):
            q = self.quantizers[i].dequantize(codes[:, :, i]) if z_q is None else z_q[:, :, i].permute(0, 2, 1)
            rec.append(self.decoders[i](q).permute(0, 2, 1))
        return self.from_parts(rec)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        """Make this tokenizer safe to embed in a downstream policy (docs/08 §9).

        `requires_grad_(False)` alone is NOT enough: QuantizeEMAReset updates its codebook inside `@torch.no_grad()`
        by writing the buffers in place, which autograd cannot stop. A single `rvq(x)` in train mode therefore moves
        the codebook (measured: 0.42 in codebook L2 for one forward, 0.0 under eval / under `encode`). This method
        puts the module in eval mode, drops every gradient, and disables quantise-dropout, so both `forward` and
        `encode` become deterministic and read-only. `train()` will not undo it -- rebuild the model to train again.
        """
        self.eval()
        self.requires_grad_(False)
        for q in self.quantizers:
            q.dropout_prob = 0.0
        self.train = lambda mode=True: self                  # a downstream .train() must not re-arm the EMA
        return self

    def codebooks(self):
        """-> [n_parts, n_quant, nb_code, code_dim]; what a CodeFlow head needs for `latent_norm_mode=codebook`."""
        return torch.stack([torch.stack([layer.codebook for layer in q.layers]) for q in self.quantizers])


def load_rvq(path, device="cpu", freeze=True):
    """Rebuild a trained tokenizer from one of `scripts/hml_phys/train_rvq.py`'s checkpoints.

    -> (model, args_dict). The checkpoint carries everything needed: the training `args`, the channel groups and
    the geometric-channel indices. Frozen by default -- see `PartRVQVAE.freeze` for why that matters.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = PartRVQVAE(int(a["n_channels"]), [np.asarray(p) for p in ck["parts"]], width=int(a["width"]),
                       down_t=int(a["down_t"]), depth=int(a["depth"]), dilation=int(a["dilation"]),
                       code_dim=int(a["code_dim"]), nb_code=int(a["nb_code"]), n_quant=int(a["n_quant"]),
                       shared_codebook=bool(a["shared_codebook"]), dropout_prob=float(a["dropout_prob"]),
                       mu=float(a["mu"]))
    model.load_state_dict(ck["model"])                       # strict: a silent mismatch would give random codebooks
    model.to(device)
    return (model.freeze() if freeze else model), a


def rvq_losses(x_rec, x, stats, explicit_idx=None, w_explicit=0.5, w_commit=0.02):
    """MoMask's VQ objective (models/vq/vq_trainer.py:45-50): smooth-L1 on the whole representation, plus the same
    on the 'explicit' geometric channels (their recovered local joint positions; here the local_positions channels of
    the variant, if it has any), plus the commitment term."""
    rec = F.smooth_l1_loss(x_rec, x)
    expl = F.smooth_l1_loss(x_rec[..., explicit_idx], x[..., explicit_idx]) if explicit_idx is not None else rec.new_zeros(())
    return rec + w_explicit * expl + w_commit * stats["commit"], dict(rec=rec, explicit=expl, commit=stats["commit"])
