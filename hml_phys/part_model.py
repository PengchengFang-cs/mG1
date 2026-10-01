"""Part-structured physics policy (MoGeFlow style, docs/07 §17).

Replaces the two-stage root-then-body denoiser. The root is no longer a separate stage conditioning the
body through a narrow bridge: it is part 0 of six peers, and all six are denoised jointly in one pass.

    per-part LayerNorm(affine) + Linear(2*d_p+2 -> part_dim)    [LN(z_part) | observed mask | mean, std of z_part]
        -> concat into one per-frame token of width hidden = 6 * part_dim
        -> one shared FrameMotionTextDiT trunk (double-stream + single-stream, joint text attention)
        -> per-part FinalLayer (own zero-init AdaLN) -> d_p

Copied from MoGeFlow (vendor_mogeflow/models/codeflow/part_structured_motion_code_flow.py): the per-part
in/out projections around a shared full-width trunk, the hidden = parts x part_dim constraint with part
boundaries aligned to attention heads, cond = timestep + pooled text, and joint text attention.
Dropped: everything to do with the codebook (the tokens here are continuous physics features).
Kept from our own design: the observed-history prefix (MoGeFlow has no history conditioning at all),
signed frame indices, and the scalar conditions (progress, total length).

Optional text cross-attention (v5, docs/07 §18; `text_cross_attention=True`), ported from the user's moge_UMO_ST
(vendor_moge_umo/models/codeflow/dit_blocks.py: FrameMotionTextDiT._inject_local_text, llm2vec_cache.local_proj):
after every double-stream block, motion += tanh(gate_i) * joint_attn_i(LN(motion), text_memory), where
  - the softmax runs over the text tokens only, so motion cannot ignore text the way it can inside joint attention
    (v3 probe: body-stream motion queries put 7.2% of their mass on text vs 20.8% uniform);
  - joint_attn_i is the block's own attention module (same q/kv/out weights, no RoPE), as in moge_UMO_ST;
  - text_memory = LayerNorm + Linear over the raw CLIP token features, separate from the joint-attention text stream;
  - gate_i is a zero-initialised scalar, so at init the model is exactly the v4 model.

text_mode (docs/07 §18.1):
  "joint_tokens"   (v4, first v5 run): the 50 CLIP token features enter the joint attention, pooled CLIP enters AdaLN;
                   the optional cross-attention re-reads the same tokens.
  "sentence_xattn" (v5, moge_UMO_ST `sentence` mode, llm2vec_cache.py:175-200 + kimodo_like_flow_dit.py:246-263):
                   the joint attention sees ONE sentence token (projected pooled CLIP), AdaLN carries no text,
                   and word-level text reaches motion ONLY through the gated cross-attention, whose keys exclude
                   CLIP's start-of-text token (slot 0, identical for every caption, a caption-independent sink).
  "xattn_only"     (v5b, user 2026-09-19): as sentence_xattn but WITHOUT the sentence token -- the joint attention
                   carries no text at all (the double blocks' text stream is empty, its parameters are frozen and
                   unused), AdaLN carries no text; all text, sentence-level and word-level, enters only through the
                   gated cross-attention (MIND: text only via cross-attention).
"""
import numpy as np
import torch
import torch.nn as nn

from hml_phys.tokens import ROOT_DIM, BODY_DIM, PART_NAMES, part_channels, part_dims
from hml_phys.mc_model import _blocks

FrameMotionTextDiT = _blocks.FrameMotionTextDiT
TimestepEmbedder = _blocks.TimestepEmbedder
FinalLayer = _blocks.FinalLayer

TOKEN_DIM = ROOT_DIM + BODY_DIM


class PartPhysPolicyDiT(nn.Module):
    def __init__(self, hidden_dim=504, num_heads=12, depth_double=3, depth_single=6, mlp_ratio=4.0,
                 dropout=0.0, text_token_dim=768, text_pooled_dim=768, max_text_tokens=50, n_scalar_cond=2,
                 text_cross_attention=False, text_mode="joint_tokens", part_dims=None):
        """part_dims: group widths of a FLAT input vector, used instead of the physics token's body parts.
        CodeFlow passes [code_dim] * n_quant, i.e. the six residual levels of the frozen RVQ take the place of
        MoGeFlow's six body groups (docs/08 §10); the grouping is then the identity and nothing is gathered."""
        super().__init__()
        if part_dims is None:
            chans = part_channels()
        else:
            off = np.cumsum([0] + list(part_dims))
            chans = [np.arange(off[i], off[i + 1], dtype=np.int64) for i in range(len(part_dims))]
        self.dims = [len(c) for c in chans]
        self.n_parts = len(chans)
        assert hidden_dim % self.n_parts == 0, f"hidden {hidden_dim} must be divisible by {self.n_parts} parts"
        self.part_dim = hidden_dim // self.n_parts
        assert hidden_dim % num_heads == 0, "hidden must be divisible by the head count"
        head_dim = hidden_dim // num_heads
        assert self.part_dim % head_dim == 0, \
            f"part_dim {self.part_dim} must be a whole number of heads ({head_dim}) so part and head boundaries align"
        self.hidden_dim, self.max_text_tokens = hidden_dim, max_text_tokens
        # channel indices as a buffer so gather/scatter follow the module to the right device
        self.register_buffer("part_index", torch.from_numpy(np.concatenate(chans)), persistent=False)
        self.part_slices = np.cumsum([0] + self.dims).tolist()
        inv = np.argsort(np.concatenate(chans))
        self.register_buffer("inverse_index", torch.from_numpy(inv), persistent=False)

        self.part_norms = nn.ModuleList([nn.LayerNorm(d, elementwise_affine=True, eps=1e-6) for d in self.dims])
        # input per part: [LayerNorm(z_p) | observed mask | per-frame mean(z_p), std(z_p)]. The last two restore what
        # LayerNorm removes: MoGeFlow normalises homogeneous VQ code vectors, ours are heterogeneous physics channels
        # (root height, velocities, ...) where the per-frame mean/std carry 12-32% of the variance (docs/07 §17).
        self.part_inputs = nn.ModuleList([nn.Linear(2 * d + 2, self.part_dim) for d in self.dims])
        self.part_outputs = nn.ModuleList([FinalLayer(hidden_dim, d) for d in self.dims])       # zero-init AdaLN heads

        assert text_mode in ("joint_tokens", "sentence_xattn", "xattn_only"), text_mode
        assert text_mode == "joint_tokens" or text_cross_attention, f"{text_mode} needs the text cross-attention"
        self.text_mode = text_mode
        self.timestep_embed = TimestepEmbedder(hidden_dim)
        if text_mode == "joint_tokens":
            self.token_proj = nn.Linear(text_token_dim, hidden_dim)
            self.pooled_proj = nn.Sequential(nn.Linear(text_pooled_dim, hidden_dim), nn.SiLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif text_mode == "sentence_xattn":   # moge sentence mode: one projected sentence token in the joint attention
            self.sentence_proj = nn.Linear(text_pooled_dim, hidden_dim)
        self.scalar_embed = nn.Sequential(nn.Linear(n_scalar_cond, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.scalar_embed[-1].weight); nn.init.zeros_(self.scalar_embed[-1].bias)
        self.backbone = FrameMotionTextDiT(hidden_size=hidden_dim, num_heads=num_heads,
                                           depth_double=depth_double, depth_single=depth_single,
                                           mlp_ratio=mlp_ratio, dropout=dropout,
                                           rope_axes_dims=[head_dim])
        if text_mode == "xattn_only":   # empty text stream: these parameters would only ever see zero-length input
            for blk in self.backbone.double_blocks:
                for mod in (blk.text_mod, blk.text_ffn):
                    for p_ in mod.parameters():
                        p_.requires_grad_(False)
        self.text_cross_attention = bool(text_cross_attention)
        self.record_xattn, self.xattn_stats = False, []
        if self.text_cross_attention:
            # built under a forked RNG (as moge_UMO_ST does) so every other weight initialises exactly as in v4
            with torch.random.fork_rng(devices=[]):
                self.xattn_text_proj = nn.Sequential(nn.LayerNorm(text_token_dim), nn.Linear(text_token_dim, hidden_dim))
                self.xattn_query_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6) for _ in range(depth_double)])
                self.xattn_gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(depth_double)])

    # ---- token <-> parts
    def to_parts(self, x):
        """[B,T,435] -> list of 6 [B,T,d_p] in part order."""
        g = x.index_select(-1, self.part_index)
        return [g[..., self.part_slices[i]:self.part_slices[i + 1]] for i in range(self.n_parts)]

    def from_parts(self, parts):
        """list of 6 [B,T,d_p] -> [B,T,435] in the original channel order."""
        return torch.cat(parts, -1).index_select(-1, self.inverse_index)

    def text_condition(self, text_tokens, text_pooled, text_len):
        B, L, _ = text_tokens.shape
        pad = torch.arange(L, device=text_tokens.device)[None] >= text_len[:, None]
        pad[:, 0] = False
        return self.token_proj(text_tokens), self.pooled_proj(text_pooled), pad

    def forward(self, z, observed_mask, t, text_tokens, text_pooled, text_len, scalars,
                valid=None, frame_index=None, extra_tokens=None, extra_valid=None):
        """z [B,T,435] imputed noisy token, observed_mask [B,T] (1 = observed history), t [B] in (0,1),
        scalars [B,n]. Returns the x0 prediction [B,T,435]."""
        B, T, _ = z.shape
        dtype = z.dtype
        assert text_tokens.shape[1] <= self.max_text_tokens, \
            f"{text_tokens.shape[1]} text tokens > max_text_tokens {self.max_text_tokens}"
        if self.text_mode == "joint_tokens":
            tokens, pooled, pad = self.text_condition(text_tokens.to(dtype), text_pooled.to(dtype), text_len)
            cond = self.timestep_embed(t.float()).to(dtype) + pooled + self.scalar_embed(scalars.to(dtype))
            mem_valid = ~pad
        else:
            if self.text_mode == "sentence_xattn":
                tokens = self.sentence_proj(text_pooled.to(dtype))[:, None]                  # [B,1,H]
            else:                                                                            # xattn_only: no joint text
                tokens = z.new_zeros(B, 0, self.hidden_dim)
            pad = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=z.device)
            if extra_tokens is not None:   # intent arch: intent tokens join the joint-attention text stream (docs/07 §21)
                tokens = torch.cat([tokens, extra_tokens.to(dtype)], 1)
                pad = torch.cat([pad, ~extra_valid.bool()], 1)
            cond = self.timestep_embed(t.float()).to(dtype) + self.scalar_embed(scalars.to(dtype))
            L = text_tokens.shape[1]
            idx = torch.arange(L, device=z.device)[None]
            mem_valid = (idx < text_len[:, None]) & (idx > 0)                                # drop CLIP start token
            assert bool(mem_valid.any(1).all()), "every caption (incl. CLIP('')) has >= 1 token after the start token"
        if frame_index is None:   # contiguous window fallback
            frame_index = torch.arange(T, device=z.device)[None].expand(B, T) - observed_mask.sum(1).long()[:, None]
        pos = frame_index.long().unsqueeze(-1)
        valid = torch.ones(B, T, dtype=torch.bool, device=z.device) if valid is None else valid.bool()
        m = observed_mask.to(dtype)[..., None]
        parts_in = []
        for i, zp in enumerate(self.to_parts(z)):
            h = self.part_norms[i](zp)
            zf = zp.float()
            mu = zf.mean(-1, keepdim=True)
            sd = (zf.var(-1, unbiased=False, keepdim=True) + self.part_norms[i].eps).sqrt()   # LayerNorm's own statistics
            parts_in.append(self.part_inputs[i](torch.cat(
                [h, m.expand(-1, -1, zp.shape[-1]).to(h.dtype), mu.to(h.dtype), sd.to(h.dtype)], -1)))
        motion = torch.cat(parts_in, -1)                                  # [B,T,hidden]
        if self.text_cross_attention:
            memory = self.xattn_text_proj(text_tokens.to(dtype)).masked_fill(~mem_valid[..., None], 0.0)
            hidden = self._trunk_with_text_xattn(motion, tokens, cond, valid, pad, pos, memory, mem_valid)
        else:
            hidden = self.backbone(motion=motion, text=tokens, cond=cond, motion_valid=valid,
                                   text_padding_mask=pad, motion_pos_ids=pos)
        out = self.from_parts([head(hidden, cond) for head in self.part_outputs])
        return out * valid.to(dtype)[..., None]   # MoGeFlow masks its prediction by validity as well

    def _trunk_with_text_xattn(self, motion, text, cond, motion_valid, text_padding_mask, motion_pos_ids, memory,
                               memory_valid):
        """FrameMotionTextDiT.forward (vendor_motioncraft dit_blocks.py:708-768, control adapter unused) with the
        moge_UMO_ST text cross-attention injected after each double-stream block."""
        bb = self.backbone
        text_valid = ~text_padding_mask
        for i, block in enumerate(bb.double_blocks):
            motion, text = block(motion, text, cond, motion_valid=motion_valid, text_valid=text_valid,
                                 pos_ids=motion_pos_ids, rope_axes_dims=bb.rope_axes_dims)
            update = block.joint_attn(self.xattn_query_norms[i](motion), memory,
                                      key_valid=memory_valid, query_valid=motion_valid)
            gate = torch.tanh(self.xattn_gates[i]).to(motion.dtype)
            if self.record_xattn:   # probe only: gate and the size of the injected update relative to the stream
                vm = motion_valid[..., None].to(motion.dtype)
                self.xattn_stats.append(dict(
                    gate=float(gate), rel=float((gate * update * vm).float().norm() / (motion * vm).float().norm().clamp_min(1e-8))))
            motion = motion + gate * update
        text_pos = torch.zeros(text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
                               device=motion_pos_ids.device, dtype=motion_pos_ids.dtype)
        x = torch.cat([motion, text], dim=1)
        valid = torch.cat([motion_valid, text_valid], dim=1)
        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
        for block in bb.single_blocks:
            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=bb.rope_axes_dims)
        return x[:, : motion.shape[1]]

    def num_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
