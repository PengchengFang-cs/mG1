"""Causal transformer denoiser with per-token noise-level embedding and text cross-attention (UniPhys-style, ADAPT sizes)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__(); self.dim, self.theta = dim, theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (...,) -> (..., dim)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.theta) * torch.arange(half, device=x.device) / half)
        ang = x.float()[..., None] * freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class DenoiserTransformer(nn.Module):
    def __init__(self, x_dim: int = 125, text_dim: int = 512, d_model: int = 512, n_layers: int = 8, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, k_emb: int = 64, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.k_embed = SinusoidalEmb(k_emb)
        self.t_embed = SinusoidalEmb(d_model)
        self.in_mlp = nn.Sequential(nn.Linear(x_dim + k_emb, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.text_proj = nn.Linear(text_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d_model, x_dim)
        self.register_buffer("null_text", torch.zeros(1, text_dim), persistent=False)

    def forward(self, x: torch.Tensor, k: torch.Tensor, text: torch.Tensor | None, force_uncond: bool = False) -> torch.Tensor:
        """x: (B,T,x_dim) noised tokens; k: (B,T) noise levels (int for ddpm, float ~1000*t for flow); text: (B,text_dim) or None."""
        B, T, _ = x.shape
        h = self.in_mlp(torch.cat([x, self.k_embed(k)], dim=-1))
        h = h + self.t_embed(torch.arange(T, device=x.device))[None]
        if text is None or force_uncond:
            text = self.null_text.expand(B, -1)
        mem = self.text_proj(text)[:, None, :]  # (B,1,d)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.decoder(h, mem, tgt_mask=mask, tgt_is_causal=True)
        return self.out(h)


class AdaLNBlock(nn.Module):
    """Pre-norm causal self-attention + MLP block with adaLN-Zero modulation (DiT style) from a per-token condition."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, c, mask):
        # c: (B,T,d) per-token condition (noise level + text)
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + s1) + b1
        h = h + g1 * self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm2(h) * (1 + s2) + b2
        h = h + g2 * self.mlp(x)
        return h


class AdaLNDenoiser(nn.Module):
    """Same interface as DenoiserTransformer, but noise level and text enter through adaLN-Zero modulation
    (per token: k-embedding + text projection), which conditions every layer strongly (MoGeFlow/DiT style)."""
    def __init__(self, x_dim: int = 125, text_dim: int = 512, d_model: int = 512, n_layers: int = 8, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, k_emb: int = 64, max_len: int = 64, n_loops: int = 1):
        super().__init__()
        self.d_model, self.n_loops = d_model, n_loops
        self.k_embed = SinusoidalEmb(k_emb)
        self.k_mlp = nn.Sequential(nn.Linear(k_emb, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.t_embed = SinusoidalEmb(d_model)
        self.in_proj = nn.Linear(x_dim, d_model)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([AdaLNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.loop_embed = nn.Embedding(max(n_loops, 1), d_model) if n_loops > 1 else None
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out = nn.Linear(d_model, x_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("null_text", torch.zeros(1, text_dim), persistent=False)

    def forward(self, x, k, text, force_uncond: bool = False):
        B, T, _ = x.shape
        if text is None or force_uncond:
            text = self.null_text.expand(B, -1)
        c = self.k_mlp(self.k_embed(k)) + self.text_proj(text)[:, None, :]          # (B,T,d)
        h = self.in_proj(x) + self.t_embed(torch.arange(T, device=x.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for loop in range(max(self.n_loops, 1)):
            cl = c + (self.loop_embed.weight[loop] if self.loop_embed is not None else 0)
            for blk in self.blocks:
                h = blk(h, cl, mask)
        s, b = self.ada_out(c).chunk(2, dim=-1)
        return self.out(self.norm_out(h) * (1 + s) + b)
