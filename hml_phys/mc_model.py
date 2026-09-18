"""MotionCraft-style root-first / body-second DiT for physics tokens (docs/07 §1–§4, §10).

Reuses MotionCraft's blocks verbatim (vendor_motioncraft/models/codeflow/dit_blocks.py: FrameMotionTextDiT =
double-stream joint text-motion blocks + single-stream blocks, AdaLN-Zero, RMSNorm q/k, SwiGLU, RoPE, and
TimestepEmbedder). Differences from HY273UnifiedKimodo, all forced by the physics token layout:
  * root stream 15-d, body stream 420-d (no contact channels -> the whole token is ODE state)
  * observed_mask is frame-level (history prefix); expanded to feature level for the input projections
  * root stage input = full token (root + body, as in MotionCraft), output = root token only
  * bridge root -> body: the predicted root is converted to the 4-d LOCAL root (yaw rate, dx*fps, dy*fps,
    height) exactly as KiMoDo / ARDY / MotionCraft do (KimodoRootConditioner), normalised with its own
    statistics, detached in training and differentiable at test time; history frames use the observed root
  * signed positional indices: the first generated frame is index 0, history frames are negative and carry
    their true frame offset, so a non-uniformly sampled long history is understood correctly
  * AdaLN cond = timestep + pooled text + scalar conditions (progress, total_len) ; no heading c_dir, no task id
  * text = precomputed CLIP ViT-L/14 tokens [B,50,768] + pooled [B,768] projected like FrozenCLIPTextEncoder
"""
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

from hml_phys.tokens import ROOT_DIM, BODY_DIM, ROOT_SLICES, LOCAL_ROOT_DIM

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
                 detach_root_bridge=True, root_dim=ROOT_DIM, body_dim=BODY_DIM,
                 local_root=True, fps=30.0, root_stats=None, local_root_stats=None):
        super().__init__()
        self.root_dim, self.body_dim, self.hidden_dim = root_dim, body_dim, hidden_dim
        self.detach_root_bridge = detach_root_bridge
        self.max_text_tokens = max_text_tokens
        self.local_root, self.fps = local_root, float(fps)
        self.bridge_dim = LOCAL_ROOT_DIM if local_root else root_dim
        # statistics needed by the bridge: un-normalise the predicted root, then re-normalise the local root
        for name, st, dim in (("root", root_stats, root_dim), ("local_root", local_root_stats, LOCAL_ROOT_DIM)):
            mean = torch.zeros(dim) if st is None else torch.as_tensor(st[0], dtype=torch.float32)
            std = torch.ones(dim) if st is None else torch.as_tensor(st[1], dtype=torch.float32)
            self.register_buffer(f"{name}_mean", mean); self.register_buffer(f"{name}_std", std)
        # input / output projections  (root: [z_root | z_body | mask_root | mask_body] like MotionCraft's root stage,
        # which consumes the full state; body: [bridge_root | z_body | mask_body])
        self.root_input_proj = nn.Linear((root_dim + body_dim) * 2, hidden_dim)
        self.body_input_proj = nn.Linear(self.bridge_dim + body_dim * 2, hidden_dim)
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

    def to_local_root(self, root_norm, valid=None, frame_index=None):
        """normalised root [B,T,15] -> normalised local root [B,T,4] = [yaw rate, dx*fps, dy*fps, height].

        Mirrors KimodoRootConditioner (vendor_motioncraft/models/raw_motion/hy273_root_conditioning.py):
        un-normalise, finite-difference in fp32, last valid row copies its predecessor, re-normalise with
        the local-root statistics. `frame_index` gives the true frame offsets so that non-contiguous rows
        (the sparse long history) are divided by their actual gap; contiguous rows give dt = 1.
        """
        r = root_norm.float() * self.root_std + self.root_mean
        pos = r[..., ROOT_SLICES["root_trans"][0]:ROOT_SLICES["root_trans"][1]]
        rot6 = r[..., ROOT_SLICES["root_rot_6d"][0]:ROOT_SLICES["root_rot_6d"][1]]
        head = rot6[..., 0:2]                                   # horizontal part of the body x axis
        head = head / head.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        B, T, _ = r.shape
        out = torch.zeros(B, T, LOCAL_ROOT_DIM, device=r.device, dtype=torch.float32)
        if T >= 2:
            if frame_index is None:
                dt = torch.ones(B, T - 1, device=r.device, dtype=torch.float32)
            else:
                dt = (frame_index[:, 1:] - frame_index[:, :-1]).float().clamp_min(1.0)
            cross = head[:, :-1, 0] * head[:, 1:, 1] - head[:, :-1, 1] * head[:, 1:, 0]
            dot = (head[:, :-1] * head[:, 1:]).sum(-1)
            out[:, :-1, 0] = torch.atan2(cross, dot) * self.fps / dt
            out[:, :-1, 1:3] = (pos[:, 1:, :2] - pos[:, :-1, :2]) * self.fps / dt[..., None]
        out[..., 3] = pos[..., 2]
        if T >= 2:  # last valid frame has no successor -> copy the previous row's velocity channels
            n_valid = (valid.sum(1).long() if valid is not None
                       else torch.full((B,), T, device=r.device, dtype=torch.long))
            idx = (n_valid - 1).clamp(min=1)
            src = out.gather(1, (idx - 1).view(B, 1, 1).expand(B, 1, LOCAL_ROOT_DIM))[:, 0, :3]
            out.scatter_(1, idx.view(B, 1, 1).expand(B, 1, 3), src.unsqueeze(1))
        out = (out - self.local_root_mean) / self.local_root_std
        return out.to(root_norm.dtype)

    def text_condition(self, text_tokens, text_pooled, text_len):
        """-> (tokens [B,L,h], pooled [B,h], padding_mask [B,L] True=pad). Empty text (len 0) keeps slot 0 valid."""
        B, L, _ = text_tokens.shape
        pad = torch.arange(L, device=text_tokens.device)[None] >= text_len[:, None]
        pad[:, 0] = False
        return self.token_proj(text_tokens), self.pooled_proj(text_pooled), pad

    def forward(self, z_root, z_body, observed_mask, t, text_tokens, text_pooled, text_len, scalars,
                return_details=False, valid=None, frame_index=None):
        """z_root [B,T,15], z_body [B,T,420] (imputed noisy tokens), observed_mask [B,T] (1=history),
        t [B] in (0,1) (1 = clean), text_* from the CLIP cache, scalars [B,n] (progress, total_len/10),
        frame_index [B,T] signed true frame offsets (first generated frame = 0, history negative); when
        omitted it falls back to 0..T-1 shifted so that the first non-observed frame is 0.
        Returns x0 prediction (root [B,T,15], body [B,T,420])."""
        B, T, _ = z_root.shape
        dtype = z_root.dtype
        tokens, pooled, pad = self.text_condition(text_tokens.to(dtype), text_pooled.to(dtype), text_len)
        cond = self.timestep_embed(t.float()).to(dtype) + pooled + self.scalar_embed(scalars.to(dtype))
        if frame_index is None:
            n_hist = observed_mask.sum(1).long()
            frame_index = torch.arange(T, device=z_root.device)[None].expand(B, T) - n_hist[:, None]
        pos = frame_index.long().unsqueeze(-1)
        valid = torch.ones(B, T, dtype=torch.bool, device=z_root.device) if valid is None else valid.bool()
        m = observed_mask.to(dtype)[..., None]
        root_in = torch.cat([z_root, z_body, m.expand(-1, -1, self.root_dim + self.body_dim)], -1)
        h_root = self.root_backbone(motion=self.root_input_proj(root_in), text=tokens, cond=cond, motion_valid=valid,
                                    text_padding_mask=pad, motion_pos_ids=pos)
        x0_root = self.root_output_proj(h_root)
        bridge = x0_root.detach() if (self.training and self.detach_root_bridge) else x0_root
        # history frames of the bridge are known exactly: use the observed root there
        bridge = bridge * (1 - m) + z_root * m
        if self.local_root:  # KiMoDo/ARDY/MotionCraft: the body stage sees the local (velocity) root
            bridge = self.to_local_root(bridge, valid=valid, frame_index=frame_index).to(dtype)
        body_in = torch.cat([bridge, z_body, m.expand(-1, -1, self.body_dim)], -1)
        h_body = self.body_backbone(motion=self.body_input_proj(body_in), text=tokens, cond=cond, motion_valid=valid,
                                    text_padding_mask=pad, motion_pos_ids=pos)
        x0_body = self.body_output_proj(h_body)
        if return_details:
            return x0_root, x0_body, dict(bridge=bridge)
        return x0_root, x0_body

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
