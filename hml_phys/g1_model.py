"""Text-driven action policy for the G1 (docs/04 line A).

Base architecture follows MotionCraft's own `RawFlowDiT` (vendor_motioncraft/models/raw_motion/raw_flow_dit.py:
`input_proj = nn.Linear(source_dim + mask_dim, hidden)` -> `FrameMotionTextDiT` -> `output_proj`), i.e. a FLAT
input projection.

Why flat and not the 6-part projection the SMPL policy uses: that part structure came from MoGeFlow, where each
body part owns its own VQ codebook, so splitting the channels is what the codebooks require.  We have no VQ, and
the ablation never paid: v3 (flat, two-stream) reached 0.52 relative to the physics ground truth and v4
(part-structured) 0.54 -- inside the noise.  On G1 it would also have to be redefined from scratch, because
`hml_phys/tokens.py:_PART_BODIES` is written against SMPL's 24-body skeleton while G1 has 29 joints.  So the
part structure is dropped here and the variable under test stays the intent mechanism.

What IS carried over from the SMPL work, because it was measured to matter:
  * the intent mechanism (HIP/IIP hidden states as extra tokens in the joint attention) -- the only change that
    ever moved R@1 (0.55 -> 0.90 relative to the physics ground truth)
  * no sparse long history (four independent pieces of evidence that it hurts: fall rate 24% -> 8.6%)
  * `sentence_xattn` text routing: one sentence token in the joint stream plus gated word-level cross attention

Not carried over (dead ends): VQ codebooks in the action path, the intent-residual completion signal.

Shapes: token 96 = proprio 67 + action 29; the policy generates the ACTION channels of the future rows only,
exactly as route A does on the SMPL side (the states are produced by the simulator, not by the policy).
"""
import numpy as np
import torch
import torch.nn as nn

from hml_phys.g1_data import ACTION_DIM, PROPRIO_DIM, TOKEN_DIM
from hml_phys.part_model import FinalLayer, FrameMotionTextDiT, TimestepEmbedder


class G1FlowPolicy(nn.Module):
    """Flat MotionCraft-style flow DiT over the 96-d G1 token."""

    def __init__(self, hidden_dim=512, num_heads=8, depth_double=3, depth_single=6, mlp_ratio=4.0,
                 dropout=0.0, text_token_dim=768, text_pooled_dim=768, max_text_tokens=50,
                 n_scalar_cond=2, text_mode="sentence_xattn", text_cross_attention=True,
                 token_dim=TOKEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden must be divisible by the head count"
        self.hidden_dim, self.max_text_tokens = hidden_dim, max_text_tokens
        self.token_dim, self.action_dim = int(token_dim), int(action_dim)
        head_dim = hidden_dim // num_heads

        # MotionCraft concatenates the observed-mask channels onto the input rather than folding them per part
        self.input_proj = nn.Linear(self.token_dim + 1, hidden_dim)
        self.input_norm = nn.LayerNorm(self.token_dim, elementwise_affine=True, eps=1e-6)
        self.output_head = FinalLayer(hidden_dim, self.action_dim)          # zero-init AdaLN head

        assert text_mode in ("joint_tokens", "sentence_xattn", "xattn_only"), text_mode
        assert text_mode == "joint_tokens" or text_cross_attention, f"{text_mode} needs the cross attention"
        self.text_mode = text_mode
        self.timestep_embed = TimestepEmbedder(hidden_dim)
        if text_mode == "joint_tokens":
            self.token_proj = nn.Linear(text_token_dim, hidden_dim)
            self.pooled_proj = nn.Sequential(nn.Linear(text_pooled_dim, hidden_dim), nn.SiLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif text_mode == "sentence_xattn":
            self.sentence_proj = nn.Linear(text_pooled_dim, hidden_dim)
        self.scalar_embed = nn.Sequential(nn.Linear(n_scalar_cond, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.scalar_embed[-1].weight); nn.init.zeros_(self.scalar_embed[-1].bias)

        self.backbone = FrameMotionTextDiT(hidden_size=hidden_dim, num_heads=num_heads,
                                          depth_double=depth_double, depth_single=depth_single,
                                          mlp_ratio=mlp_ratio, dropout=dropout, rope_axes_dims=[head_dim])
        if text_mode == "xattn_only":
            for blk in self.backbone.double_blocks:
                for mod in (blk.text_mod, blk.text_ffn):
                    for p_ in mod.parameters():
                        p_.requires_grad_(False)
        self.text_cross_attention = bool(text_cross_attention)
        if self.text_cross_attention:
            with torch.random.fork_rng(devices=[]):        # same construction order as the SMPL policy
                self.xattn_text_proj = nn.Sequential(nn.LayerNorm(text_token_dim),
                                                     nn.Linear(text_token_dim, hidden_dim))
                self.xattn_query_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6) for _ in range(depth_double)])
                self.xattn_gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(depth_double)])

    # ------------------------------------------------------------------ helpers
    def action_slice(self):
        return slice(self.token_dim - self.action_dim, self.token_dim)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _trunk_with_text_xattn(self, motion, text, cond, motion_valid, text_padding_mask, motion_pos_ids,
                               memory, memory_valid):
        """FrameMotionTextDiT.forward with the moge_UMO_ST gated text cross-attention injected after each
        double-stream block -- copied from hml_phys/part_model.py:185-209 so the two policies share the exact
        same text path (only the input/output projections differ)."""
        bb = self.backbone
        text_valid = ~text_padding_mask
        for i, block in enumerate(bb.double_blocks):
            motion, text = block(motion, text, cond, motion_valid=motion_valid, text_valid=text_valid,
                                 pos_ids=motion_pos_ids, rope_axes_dims=bb.rope_axes_dims)
            update = block.joint_attn(self.xattn_query_norms[i](motion), memory,
                                      key_valid=memory_valid, query_valid=motion_valid)
            motion = motion + torch.tanh(self.xattn_gates[i]).to(motion.dtype) * update
        text_pos = torch.zeros(text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
                               device=motion_pos_ids.device, dtype=motion_pos_ids.dtype)
        x = torch.cat([motion, text], dim=1)
        valid = torch.cat([motion_valid, text_valid], dim=1)
        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
        for block in bb.single_blocks:
            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=bb.rope_axes_dims)
        return x[:, : motion.shape[1]]

    # ------------------------------------------------------------------ forward
    def forward(self, z, observed_mask, t, text_tokens, text_pooled, text_len, scalars,
                valid=None, frame_index=None, extra_tokens=None, extra_valid=None):
        """z [B,T,96] the imputed noisy token (history rows clean, future rows noised on the action channels),
        observed_mask [B,T] with 1 on history rows, t [B] in (0,1).  -> action prediction [B,T,29]."""
        B, T, D = z.shape
        assert D == self.token_dim, f"expected {self.token_dim} channels, got {D}"
        dtype = z.dtype
        assert text_tokens.shape[1] <= self.max_text_tokens

        if self.text_mode == "joint_tokens":
            L = text_tokens.shape[1]
            pad = torch.arange(L, device=z.device)[None] >= text_len[:, None]
            pad[:, 0] = False
            tokens = self.token_proj(text_tokens.to(dtype))
            cond = self.timestep_embed(t.float()).to(dtype) + self.pooled_proj(text_pooled.to(dtype)) \
                + self.scalar_embed(scalars.to(dtype))
            mem_valid = ~pad
        else:
            if self.text_mode == "sentence_xattn":
                tokens = self.sentence_proj(text_pooled.to(dtype))[:, None]
            else:
                tokens = z.new_zeros(B, 0, self.hidden_dim)
            pad = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=z.device)
            if extra_tokens is not None:                  # intent tokens join the joint attention stream
                tokens = torch.cat([tokens, extra_tokens.to(dtype)], 1)
                pad = torch.cat([pad, ~extra_valid.bool()], 1)
            cond = self.timestep_embed(t.float()).to(dtype) + self.scalar_embed(scalars.to(dtype))
            L = text_tokens.shape[1]
            idx = torch.arange(L, device=z.device)[None]
            mem_valid = (idx < text_len[:, None]) & (idx > 0)      # drop CLIP's start token
            assert bool(mem_valid.any(1).all()), "every caption (incl. CLIP('')) needs >= 1 token after <sot>"

        if frame_index is None:
            frame_index = torch.arange(T, device=z.device)[None].expand(B, T) - observed_mask.sum(1).long()[:, None]
        pos = frame_index.long().unsqueeze(-1)
        valid = torch.ones(B, T, dtype=torch.bool, device=z.device) if valid is None else valid.bool()
        m = observed_mask.to(dtype)[..., None]
        motion = self.input_proj(torch.cat([self.input_norm(z), m], -1))

        if self.text_cross_attention:
            memory = self.xattn_text_proj(text_tokens.to(dtype)).masked_fill(~mem_valid[..., None], 0.0)
            hidden = self._trunk_with_text_xattn(motion, tokens, cond, valid, pad, pos, memory, mem_valid)
        else:
            hidden = self.backbone(motion=motion, text=tokens, cond=cond, motion_valid=valid,
                                   text_padding_mask=pad, motion_pos_ids=pos)
        # MoGeFlow masks its prediction by validity; keep the same convention
        return self.output_head(hidden, cond) * valid.to(dtype)[..., None]


class G1IntentPolicy(nn.Module):
    """G1FlowPolicy + MIND's intent mechanism, carried over unchanged from the SMPL route A.

    The intent VAE must be retrained for G1: the SMPL one encodes 366-d states, G1's are 67-d.
    """

    def __init__(self, policy_kw, vae_ckpt, latent_stats, intent_dim=384, intent_heads=6, intent_depth=4,
                 intent_mlp=1.5, text_token_dim=768, device="cpu", n_lat=None, d_lat=32):
        super().__init__()
        from hml_phys.intent_model import IntentDiT, TextAdapter
        from hml_phys.intent_vae import load_intent_vae
        self.policy = G1FlowPolicy(**policy_kw)
        assert self.policy.text_mode == "sentence_xattn", "route A builds on the sentence-xattn text path"
        self.adapter = TextAdapter(text_token_dim, intent_dim, intent_heads)
        # G1: L_INTENT 28 frames / 4x downsampling = 7 latent frames (SMPL's is 16/4 = 4)
        if n_lat is None:
            from hml_phys.g1_data import L_INTENT
            n_lat = L_INTENT // 4
        self.n_lat, self.d_lat = int(n_lat), int(d_lat)
        self.hip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=False,
                             n_lat=self.n_lat, d_lat=self.d_lat)
        self.iip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=True,
                             n_scalar=2, mem_extra=1, n_lat=self.n_lat, d_lat=self.d_lat)
        H = self.policy.hidden_dim
        self.hol_proj = nn.Sequential(nn.LayerNorm(intent_dim), nn.Linear(intent_dim, H))
        self.imm_proj = nn.Sequential(nn.LayerNorm(intent_dim), nn.Linear(intent_dim, H))
        self.intent_type = nn.Parameter(torch.zeros(2, H))
        self.vae, _ = load_intent_vae(vae_ckpt, "cpu")
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        z = np.load(latent_stats)
        self.register_buffer("lat_mean", torch.from_numpy(z["mean"]).float(), persistent=True)
        self.register_buffer("lat_std", torch.from_numpy(z["std"]).float(), persistent=True)
        self.to(device)

    def trainable(self):
        frozen = {id(p) for p in self.vae.parameters()}
        return [p for p in self.parameters() if id(p) not in frozen]

    def encode_latent(self, x):
        with torch.no_grad():
            return (self.vae.encode(x.float())[1] - self.lat_mean) / self.lat_std

    def intent_tokens(self, h_hip, h_iip, keep):
        toks = torch.cat([self.hol_proj(h_hip) + self.intent_type[0],
                          self.imm_proj(h_iip) + self.intent_type[1]], 1)
        return toks, keep[:, None].expand(-1, toks.shape[1])

    def num_params(self):
        return sum(p.numel() for p in self.trainable())

    def forward(self, *a, **kw):
        return self.policy(*a, **kw)
