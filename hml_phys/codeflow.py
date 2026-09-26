"""MoGeFlow-style structured motion-code frame flow over a frozen RVQ tokenizer (docs/08 §10).

The frozen tokenizer of `hml_phys/rvq.py` turns a 64-frame window into 16 latent frames, each carrying the
`n_quant = 6` residual code vectors of `code_dim = 512`. Those six levels take the place of MoGeFlow's six body
groups: the policy is `PartPhysPolicyDiT(part_dims=[code_dim] * n_quant)`, i.e. per-level input/output
projections around one shared FrameMotionTextDiT trunk, exactly as in v4/v5.

  latent window (16 frames) = [ 8 observed history | 8 generated future ]
  per latent frame          = 6 levels x 512, normalised per level by that level's codebook statistics
                              (MoGeFlow's `latent_norm_mode=codebook`)

Rectified flow with the project's conventions (t = 1 clean, x0 prediction, velocity-space loss); **flow loss
only**, no terminal / code-classification head -- MoGeFlow's released recipe has `terminal loss 0.0`
(vendor_mogeflow/README.md:186). At sampling time each level's predicted vector is snapped to its nearest
codebook entry, the levels are summed and the frozen decoder turns them back into frames.

`extra_tokens` on the trunk is untouched, so version 2 feeds the existing HIP/IIP intent tokens straight in.

TWO-TOKENIZER ("码本对照") MODE -- `rvq_obs_ckpt` (user, 2026-09-22)
------------------------------------------------------------------
Version 1 above uses ONE tokenizer for both halves of the window, so a "token" CodeFlow has to predict the codes
of the whole future physics state (435 channels), a much heavier task than route A's, which OBSERVES the full
435-d state and only generates the 69 action channels.  That confound is removed by giving the policy a second
frozen tokenizer used for the OBSERVATION only:

    history (435 channels) --[observation tokenizer, its own codebook statistics]--> 8 latent frames
    future  ( 69 channels) --[generation  tokenizer, its own codebook statistics]--> 8 latent frames

Both tokenizers have `n_quant = 6` levels of `code_dim = 512`, so both halves are latent frames of the same
width (3072) and sit in one sequence; the trunk tells them apart through `observed_mask`, which it already
takes.  Nothing else changes, so the only difference left against route A is discrete codes vs continuous
actions -- exactly the question being asked.  The two halves are still encoded as SEPARATE windows (see
`scripts/hml_phys/train_codeflow.py: prepare`).
"""
import numpy as np
import torch
import torch.nn as nn

from hml_phys.part_model import PartPhysPolicyDiT
from hml_phys.rvq import load_rvq
from hml_phys.rvq_data import variant_local_channels


class CodeFlowPolicy(nn.Module):
    def __init__(self, rvq_ckpt, device="cpu", hidden_dim=504, num_heads=12, depth_double=3, depth_single=6,
                 mlp_ratio=4.0, dropout=0.0, text_mode="joint_tokens", text_cross_attention=False,
                 max_text_tokens=50, n_scalar_cond=2, norm_mode="codebook", rvq_obs_ckpt=None):
        """rvq_obs_ckpt: optional SECOND frozen tokenizer that encodes the observed history (see the module
        docstring).  `rvq_ckpt` always stays the GENERATION tokenizer -- the codes the policy predicts, snaps
        and decodes -- so a checkpoint trained without this argument behaves exactly as before."""
        super().__init__()
        self.rvq, self.rvq_args = load_rvq(rvq_ckpt, device="cpu", freeze=True)   # eval + no grad + no EMA (docs/08 §9.3)
        assert self.rvq.n_parts == 1, "CodeFlow expects the whole-body tokenizer (--structure whole)"
        self.n_quant, self.code_dim = int(self.rvq.n_quant), int(self.rvq.code_dim)
        self.down = int(self.rvq.down)
        self.variant = str(self.rvq_args["variant"])
        self.n_channels = int(self.rvq_args["n_channels"])
        self.latent_dim = self.n_quant * self.code_dim

        # per-level normalisation of the code space. MoGeFlow's `latent_norm_mode=codebook` takes the statistics
        # from the codebook itself, which is exactly the distribution the quantised latents live in.
        assert norm_mode in ("codebook", "none")
        self.norm_mode = str(norm_mode)
        mean, std = self._code_stats(self.rvq)                            # [Q, D] each
        self.register_buffer("code_mean", mean, persistent=True)
        self.register_buffer("code_std", std, persistent=True)

        if rvq_obs_ckpt:
            self.rvq_obs, self.rvq_obs_args = load_rvq(rvq_obs_ckpt, device="cpu", freeze=True)
            assert self.rvq_obs.n_parts == 1, "the observation tokenizer must be whole-body too (--structure whole)"
            assert (int(self.rvq_obs.n_quant), int(self.rvq_obs.code_dim)) == (self.n_quant, self.code_dim), \
                (f"the two tokenizers must share the latent layout: observation "
                 f"{self.rvq_obs.n_quant}x{self.rvq_obs.code_dim} vs generation {self.n_quant}x{self.code_dim}")
            assert int(self.rvq_obs.down) == self.down, \
                f"the two tokenizers must share the temporal downsampling ({self.rvq_obs.down} vs {self.down})"
            self.obs_variant = str(self.rvq_obs_args["variant"])
            self.obs_n_channels = int(self.rvq_obs_args["n_channels"])
            m, s = self._code_stats(self.rvq_obs)
            self.register_buffer("obs_code_mean", m, persistent=True)
            self.register_buffer("obs_code_std", s, persistent=True)
            # where the generation variant's channels sit inside the observation variant's vector, so one loaded
            # window feeds both tokenizers. NON-persistent: a fixed index table, it has no business in the
            # checkpoint or in the EMA (which stores everything as float).
            self.register_buffer("gen_local",
                                 torch.from_numpy(variant_local_channels(self.obs_variant, self.variant)).long(),
                                 persistent=False)
        else:
            self.rvq_obs, self.rvq_obs_args = None, None
            self.obs_variant, self.obs_n_channels = self.variant, self.n_channels

        self.policy = PartPhysPolicyDiT(hidden_dim=hidden_dim, num_heads=num_heads, depth_double=depth_double,
                                        depth_single=depth_single, mlp_ratio=mlp_ratio, dropout=dropout,
                                        max_text_tokens=max_text_tokens, n_scalar_cond=n_scalar_cond,
                                        text_cross_attention=text_cross_attention, text_mode=text_mode,
                                        part_dims=[self.code_dim] * self.n_quant)
        self.to(device)

    # ---------------------------------------------------------------- code space <-> flat latent
    @property
    def dual(self):
        """True when a separate OBSERVATION tokenizer was loaded (the 码本对照 control)."""
        return self.rvq_obs is not None

    def _code_stats(self, rvq):
        """a tokenizer's per-level codebook (mean, std) -- MoGeFlow's `latent_norm_mode=codebook`."""
        books = rvq.codebooks()[0]                                        # [Q, nb_code, D]
        if self.norm_mode == "codebook":
            return books.mean(1), books.std(1).clamp_min(1e-4)
        return torch.zeros_like(books[:, 0]), torch.ones_like(books[:, 0])

    def _space(self, obs=False):
        """-> (codebooks [Q, nb_code, D], mean [Q, D], std [Q, D]) of the generation or the observation space."""
        if not obs:
            return self.rvq.codebooks()[0], self.code_mean, self.code_std
        assert self.dual, "no observation tokenizer was loaded (pass rvq_obs_ckpt / --rvq_obs)"
        return self.rvq_obs.codebooks()[0], self.obs_code_mean, self.obs_code_std

    def normalise(self, z_q, obs=False):
        """quantised levels [B, T', Q, D] -> flat normalised latent [B, T', Q*D]."""
        _, mean, std = self._space(obs)
        z = (z_q - mean) / std
        return z.reshape(*z.shape[:-2], self.latent_dim)

    def denormalise(self, z, obs=False):
        """flat normalised latent [B, T', Q*D] -> code-space levels [B, T', Q, D]."""
        _, mean, std = self._space(obs)
        z = z.reshape(*z.shape[:-1], self.n_quant, self.code_dim)
        return z * std + mean

    @torch.no_grad()
    def encode_window(self, x):
        """normalised GENERATION-variant window [B, W, C] -> (flat latent [B, W/down, Q*D], codes [B, W/down, Q]).

        `PartRVQVAE.encode` returns the SUM of the levels in `z_q`, so the per-level vectors are recovered from
        the code indices instead, which is exact: level q's vector is its codebook row for that index.
        """
        assert x.shape[-1] == self.n_channels, \
            f"the generation tokenizer takes {self.n_channels} ({self.variant}) channels, got {x.shape[-1]}"
        codes = self.rvq.encode(x)["codes"][:, :, 0]                      # [B, T', Q], the single whole-body group
        return self.normalise(self.lookup(codes)), codes

    @torch.no_grad()
    def encode_obs(self, x):
        """normalised OBSERVATION-variant window [B, W, C_obs] -> (flat latent [B, W/down, Q*D], codes).

        The history half of the 码本对照 control: the SECOND tokenizer and ITS codebook statistics. The result
        lives in a different code space from `encode_window`'s, which is fine because the two halves are never
        mixed -- they are concatenated as separate latent frames and separated by `observed_mask`. Only the
        generated half is ever noised, snapped or decoded.
        """
        assert self.dual, "no observation tokenizer was loaded (pass rvq_obs_ckpt / --rvq_obs)"
        assert x.shape[-1] == self.obs_n_channels, \
            f"the observation tokenizer takes {self.obs_n_channels} ({self.obs_variant}) channels, got {x.shape[-1]}"
        codes = self.rvq_obs.encode(x)["codes"][:, :, 0]
        return self.normalise(self.lookup(codes, obs=True), obs=True), codes

    def gen_channels_of(self, x):
        """OBSERVATION-variant tensor [..., C_obs] -> the GENERATION variant's channels [..., C_gen]."""
        assert self.dual, "no observation tokenizer was loaded (pass rvq_obs_ckpt / --rvq_obs)"
        return x.index_select(-1, self.gen_local)

    def lookup(self, codes, obs=False):
        """code indices [.., Q] -> per-level code vectors [.., Q, D]; a dropped level (-1) contributes zero."""
        books = self._space(obs)[0]                                       # [Q, nb_code, D]
        out = []
        for q in range(self.n_quant):
            idx = codes[..., q]
            v = torch.nn.functional.embedding(idx.clamp_min(0), books[q])
            out.append(v * (idx >= 0)[..., None].to(v.dtype))
        return torch.stack(out, -2)

    def snap(self, z):
        """flat normalised prediction [B, T', Q*D] -> (code indices [B, T', Q], snapped levels [B, T', Q, D]).

        Nearest entry per level in raw code space, i.e. the tokenizer's own assignment rule (`rvq.py:112-118`
        with temperature 0). GENERATION space only -- feed it the generated latent frames, never the observed
        ones, which in the two-tokenizer mode belong to the observation codebook.
        """
        v = self.denormalise(z)
        books = self.rvq.codebooks()[0]
        idxs, snapped = [], []
        for q in range(self.n_quant):
            b, a = books[q], v[..., q, :]
            d = (a ** 2).sum(-1, keepdim=True) - 2 * a @ b.t() + (b ** 2).sum(-1)[None, None]
            i = d.argmin(-1)
            idxs.append(i); snapped.append(torch.nn.functional.embedding(i, b))
        return torch.stack(idxs, -1), torch.stack(snapped, -2)

    @torch.no_grad()
    def decode_codes(self, codes):
        """GENERATED code indices [B, T', Q] -> normalised token window [B, T'*down, C] of the generation variant."""
        return self.rvq.decode(codes=codes[:, :, None])

    def num_params(self):
        """trainable parameters only -- the tokenizer is frozen."""
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    def forward(self, z, observed_mask, t, text_tokens, text_pooled, text_len, scalars, **kw):
        return self.policy(z, observed_mask, t, text_tokens, text_pooled, text_len, scalars, **kw)


class CodeFlowIntentPolicy(CodeFlowPolicy):
    """Version 2: the CodeFlow policy with MIND's intent mechanism bolted on UNCHANGED (docs/07 §21).

    Same holistic (HIP) and immediate (IIP) intent predictors, same frozen intent VAE, same two intent tokens in
    the joint attention stream. The only difference from route A is what the policy generates: codes of the next
    chunk instead of raw action channels.
    """
    def __init__(self, rvq_ckpt, vae_ckpt, latent_stats, device="cpu", intent_dim=384, intent_heads=6,
                 intent_depth=4, intent_mlp=1.5, text_token_dim=768, **kw):
        kw.setdefault("text_mode", "sentence_xattn")
        kw.setdefault("text_cross_attention", True)
        super().__init__(rvq_ckpt, device="cpu", **kw)
        from hml_phys.intent_model import IntentDiT, TextAdapter
        from hml_phys.intent_vae import load_intent_vae
        self.adapter = TextAdapter(text_token_dim, intent_dim, intent_heads)
        self.hip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=False)
        self.iip = IntentDiT(intent_dim, intent_heads, intent_depth, intent_mlp, prefix=True, n_scalar=2, mem_extra=1)
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
        """every parameter except the frozen modules (the tokenizer(s) and the intent VAE)."""
        mods = [self.rvq, self.vae] + ([self.rvq_obs] if self.dual else [])
        frozen = set(id(p) for m in mods for p in m.parameters())
        return [p for p in self.parameters() if id(p) not in frozen]

    def encode_latent(self, x):
        """normalised state sequence [B, 16, 366] -> normalised VAE latent."""
        with torch.no_grad():
            return (self.vae.encode(x.float())[1] - self.lat_mean) / self.lat_std

    def intent_tokens(self, h_hip, h_iip, keep):
        toks = torch.cat([self.hol_proj(h_hip) + self.intent_type[0], self.imm_proj(h_iip) + self.intent_type[1]], 1)
        return toks, keep[:, None].expand(-1, toks.shape[1])

    def num_params(self):
        return sum(p.numel() for p in self.trainable())
