"""MIND intent arch (docs/07 §21): MIND's multi-scale intent mechanism on top of the part-structured policy.

  TextAdapter : CLIP ViT-L/14 word features -> 2-layer Transformer encoder (MIND's "lightweight text adapter"),
                shared by HIP and IIP.
  IntentDiT   : small flow DiT over intent latents (4 x 32 = the frozen VAE's encoding of 16 frames). Blocks =
                AdaLN-Zero self-attention + cross-attention to a memory + SwiGLU FFN (MIND: text via cross-attention).
    HIP : tokens = 4 noisy holistic-intent latents;                  memory = adapted text
    IIP : tokens = [4 history-intent latents (observed) | 4 noisy immediate-intent latents];
          memory = [adapted text | HIP final-layer hidden states];   cond also gets the progress / total-length scalars
  The "final-layer hidden representation" (MIND §4.3-4.4) is the block stack's output before the output head, taken
  at the intent tokens. Following docs/07 §21.4-4 it is read from a forward on the CLEAN latent at t = 1 (the same
  input state as the last sampling step at test time); gradients from the downstream losses flow into HIP / IIP.
  IntentPolicy: TextAdapter + HIP + IIP + the action-only part-structured policy (PartPhysPolicyDiT, v5 base), which
  receives the two hidden-state sets as 4 + 4 extra tokens in its joint-attention text stream (docs/07 §21.4-2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hml_phys.mc_model import _blocks
from hml_phys.part_model import PartPhysPolicyDiT

TimestepEmbedder, AdaLNModulation, FinalLayer = _blocks.TimestepEmbedder, _blocks.AdaLNModulation, _blocks.FinalLayer
MultiHeadAttention, SwiGLU = _blocks.MultiHeadAttention, _blocks.SwiGLU

N_LAT, D_LAT = 4, 32          # SMPL default: 16 frames / 4x temporal downsampling, 32-d latent (VAE v2)
# G1 runs at 50 fps with L_INTENT = 28, so its VAE emits 7 latent frames; IntentDiT therefore takes n_lat
# as an argument instead of reading the module constant (the SMPL default keeps every existing ckpt loading).


class TextAdapter(nn.Module):
    """MIND appendix A: 'a two-layer transformer encoder as the text adapter' on frozen CLIP token features."""
    def __init__(self, in_dim=768, dim=384, heads=6, layers=2, ff_mult=4, dropout=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * ff_mult, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens, text_len):
        L = tokens.shape[1]
        pad = torch.arange(L, device=tokens.device)[None] >= text_len[:, None]
        h = self.enc(self.inp(tokens), src_key_padding_mask=pad)
        return self.out_norm(h).masked_fill(pad[..., None], 0.0), ~pad       # memory, memory_valid


class IntentBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.mod = AdaLNModulation(dim, num=3)                                  # zero-init -> identity at start
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, heads, dropout)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x, cond, mem, mem_valid):
        m1, m2, m3 = self.mod(cond)
        h = (1 + m1.scale) * self.n1(x) + m1.shift
        x = x + m1.gate * self.self_attn(h, h)
        h = (1 + m2.scale) * self.n2(x) + m2.shift
        x = x + m2.gate * self.cross_attn(h, mem, key_valid=mem_valid)
        h = (1 + m3.scale) * self.n3(x) + m3.shift
        return x + m3.gate * self.ffn(h)


class IntentDiT(nn.Module):
    """x0-predicting flow DiT over n_lat intent latents, optionally preceded by n_lat observed prefix latents."""
    def __init__(self, dim=384, heads=6, depth=4, mlp_ratio=4.0, prefix=False, n_scalar=0, mem_extra=0,
                 n_lat=N_LAT, d_lat=D_LAT):
        super().__init__()
        self.prefix, self.dim = prefix, dim
        self.n_lat, self.d_lat = int(n_lat), int(d_lat)
        n_tok = self.n_lat * (2 if prefix else 1)
        self.inp = nn.Linear(self.d_lat + 1, dim)                              # [latent | observed flag]
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim)); nn.init.normal_(self.pos, std=0.02)
        self.t_embed = TimestepEmbedder(dim)
        self.scalar_embed = None
        if n_scalar:
            self.scalar_embed = nn.Sequential(nn.Linear(n_scalar, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.scalar_embed[-1].weight); nn.init.zeros_(self.scalar_embed[-1].bias)
        self.mem_type = nn.Parameter(torch.zeros(1 + (1 if mem_extra else 0), dim))   # text / extra-memory type
        self.blocks = nn.ModuleList([IntentBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.out = FinalLayer(dim, self.d_lat)

    def forward(self, z, t, mem, mem_valid, prefix_latent=None, scalars=None, mem_extra=None):
        """z [B,4,32] noisy latents; returns (x0 prediction [B,4,32], final-layer hidden [B,4,dim])."""
        B = z.shape[0]
        dtype = z.dtype
        if self.prefix:
            obs = torch.cat([torch.ones(B, self.n_lat, 1, device=z.device, dtype=dtype),
                             torch.zeros(B, self.n_lat, 1, device=z.device, dtype=dtype)], 1)
            x = torch.cat([prefix_latent.to(dtype), z], 1)
        else:
            obs = torch.zeros(B, self.n_lat, 1, device=z.device, dtype=dtype)
            x = z
        x = self.inp(torch.cat([x, obs], -1)) + self.pos.to(dtype)
        cond = self.t_embed(t.float()).to(dtype)
        if self.scalar_embed is not None:
            cond = cond + self.scalar_embed(scalars.to(dtype))
        m = mem.to(dtype) + self.mem_type[0].to(dtype)
        mv = mem_valid
        if mem_extra is not None:
            m = torch.cat([m, mem_extra.to(dtype) + self.mem_type[1].to(dtype)], 1)
            mv = torch.cat([mv, torch.ones(B, mem_extra.shape[1], dtype=torch.bool, device=z.device)], 1)
        for blk in self.blocks:
            x = blk(x, cond, m, mv)
        hid = x[:, -self.n_lat:]
        return self.out(hid, cond), hid


class IntentPolicy(nn.Module):
    def __init__(self, policy_kw, intent_dim=384, intent_heads=6, intent_depth=4, intent_mlp=4.0,
                 text_token_dim=768):
        super().__init__()
        self.adapter = TextAdapter(text_token_dim, intent_dim, intent_heads)
        self.hip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=False)
        self.iip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=True, n_scalar=2, mem_extra=1)
        self.policy = PartPhysPolicyDiT(**policy_kw)
        H = self.policy.hidden_dim
        assert self.policy.text_mode == "sentence_xattn", "the intent arch builds on the v5 policy (docs/07 §21.5)"
        self.hol_proj = nn.Sequential(nn.LayerNorm(intent_dim), nn.Linear(intent_dim, H))
        self.imm_proj = nn.Sequential(nn.LayerNorm(intent_dim), nn.Linear(intent_dim, H))
        self.intent_type = nn.Parameter(torch.zeros(2, H))

    def intent_tokens(self, h_hip, h_iip, keep):
        """[B,8,H] joint-stream tokens + validity; samples with keep=False (text dropped / CFG unconditional) carry
        no intent tokens at all (masked out of the attention)."""
        toks = torch.cat([self.hol_proj(h_hip) + self.intent_type[0], self.imm_proj(h_iip) + self.intent_type[1]], 1)
        valid = keep[:, None].expand(-1, toks.shape[1])
        return toks, valid

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_params_by_part(self):
        return {k: sum(p.numel() for p in getattr(self, k).parameters())
                for k in ("adapter", "hip", "iip", "policy", "hol_proj", "imm_proj")}
