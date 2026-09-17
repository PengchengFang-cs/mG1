"""MotionCraft-style root-first / body-second DiT for physics tokens (docs/07 §1–§4, §10).

Reuses MotionCraft's blocks verbatim (vendor_motioncraft/models/codeflow/dit_blocks.py: FrameMotionTextDiT =
double-stream joint text-motion blocks + single-stream blocks, AdaLN-Zero, RMSNorm q/k, SwiGLU, RoPE, and
TimestepEmbedder). Differences from HY273UnifiedKimodo, all forced by the physics token layout:
  * root stream 15-d, body stream 420-d (no contact channels -> the whole token is ODE state)
  * observed_mask is frame-level (history prefix); expanded to feature level for the input projections
  * root stage input = full token (root + body, as in MotionCraft), output = root token only
  * bridge root -> body: the predicted root token itself (15-d, detached in training; observed root on history
    frames), instead of the file-backed 4-d "local root" finite-difference feature
  * AdaLN cond = timestep + pooled text + scalar conditions (progress, total_len) ; no heading c_dir, no task id
  * text = precomputed CLIP ViT-L/14 tokens [B,50,768] + pooled [B,768] projected like FrozenCLIPTextEncoder
"""
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

from hml_phys.tokens import ROOT_DIM, BODY_DIM

MC_ROOT = Path("/iridisfs/scratch/pf2m24/projects/motion_rebot/vendor_motioncraft")


def _load_dit_blocks():
    path = MC_ROOT / "models" / "codeflow" / "dit_blocks.py"
    spec = importlib.util.spec_from_file_location("_mc_dit_blocks", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_blocks = _load_dit_blocks()
FrameMotionTextDiT = _blocks.FrameMotionTextDiT
TimestepEmbedder = _blocks.TimestepEmbedder


class PhysPolicyDiT(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, root_depth_double=2, root_depth_single=4,
                 body_depth_double=3, body_depth_single=6, mlp_ratio=2.0, dropout=0.0,
                 text_token_dim=768, text_pooled_dim=768, max_text_tokens=50, n_scalar_cond=2,
                 detach_root_bridge=True, root_dim=ROOT_DIM, body_dim=BODY_DIM):
        super().__init__()
        self.root_dim, self.body_dim, self.hidden_dim = root_dim, body_dim, hidden_dim
        self.detach_root_bridge = detach_root_bridge
        self.max_text_tokens = max_text_tokens
        # input / output projections  (root: [z_root | z_body | mask_root | mask_body] like MotionCraft's root stage,
        # which consumes the full state; body: [bridge_root | z_body | mask_body])
        self.root_input_proj = nn.Linear((root_dim + body_dim) * 2, hidden_dim)
        self.body_input_proj = nn.Linear(root_dim + body_dim * 2, hidden_dim)
        self.root_output_proj = nn.Linear(hidden_dim, root_dim)
        self.body_output_proj = nn.Linear(hidden_dim, body_dim)
        nn.init.zeros_(self.root_output_proj.weight); nn.init.zeros_(self.root_output_proj.bias)
        nn.init.zeros_(self.body_output_proj.weight); nn.init.zeros_(self.body_output_proj.bias)
        # conditions
        self.timestep_embed = TimestepEmbedder(hidden_dim)
        self.token_proj = nn.Linear(text_token_dim, hidden_dim)
        self.pooled_proj = nn.Sequential(nn.Linear(text_pooled_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.scalar_embed = nn.Sequential(nn.Linear(n_scalar_cond, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.scalar_embed[-1].weight); nn.init.zeros_(self.scalar_embed[-1].bias)  # additive, starts neutral
        kw = dict(hidden_size=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.root_backbone = FrameMotionTextDiT(depth_double=root_depth_double, depth_single=root_depth_single, **kw)
        self.body_backbone = FrameMotionTextDiT(depth_double=body_depth_double, depth_single=body_depth_single, **kw)

    def text_condition(self, text_tokens, text_pooled, text_len):
        """-> (tokens [B,L,h], pooled [B,h], padding_mask [B,L] True=pad). Empty text (len 0) keeps slot 0 valid."""
        B, L, _ = text_tokens.shape
        pad = torch.arange(L, device=text_tokens.device)[None] >= text_len[:, None]
        pad[:, 0] = False
        return self.token_proj(text_tokens), self.pooled_proj(text_pooled), pad

    def forward(self, z_root, z_body, observed_mask, t, text_tokens, text_pooled, text_len, scalars, return_details=False, valid=None):
        """z_root [B,T,15], z_body [B,T,420] (imputed noisy tokens), observed_mask [B,T] (1=history),
        t [B] in (0,1) (1 = clean), text_* from the CLIP cache, scalars [B,n] (progress, total_len/10).
        Returns x0 prediction (root [B,T,15], body [B,T,420])."""
        B, T, _ = z_root.shape
        dtype = z_root.dtype
        tokens, pooled, pad = self.text_condition(text_tokens.to(dtype), text_pooled.to(dtype), text_len)
        cond = self.timestep_embed(t.float()).to(dtype) + pooled + self.scalar_embed(scalars.to(dtype))
        pos = torch.arange(T, device=z_root.device).view(1, T, 1).expand(B, T, 1)
        valid = torch.ones(B, T, dtype=torch.bool, device=z_root.device) if valid is None else valid.bool()
        m = observed_mask.to(dtype)[..., None]
        root_in = torch.cat([z_root, z_body, m.expand(-1, -1, self.root_dim + self.body_dim)], -1)
        h_root = self.root_backbone(motion=self.root_input_proj(root_in), text=tokens, cond=cond, motion_valid=valid,
                                    text_padding_mask=pad, motion_pos_ids=pos)
        x0_root = self.root_output_proj(h_root)
        bridge = x0_root.detach() if (self.training and self.detach_root_bridge) else x0_root
        # history frames of the bridge are known exactly: use the observed root there
        bridge = bridge * (1 - m) + z_root * m
        body_in = torch.cat([bridge, z_body, m.expand(-1, -1, self.body_dim)], -1)
        h_body = self.body_backbone(motion=self.body_input_proj(body_in), text=tokens, cond=cond, motion_valid=valid,
                                    text_padding_mask=pad, motion_pos_ids=pos)
        x0_body = self.body_output_proj(h_body)
        if return_details:
            return x0_root, x0_body, dict(bridge=bridge)
        return x0_root, x0_body

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
