"""Diffusion over token sequences with per-token noise levels. History tokens stay clean (k=0);
loss only on future tokens. v-prediction, cosine schedule, DDIM sampling with CFG (ADAPT settings)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alphas_cumprod(K: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(K + 1, dtype=torch.float64)
    f = torch.cos(((steps / K) + s) / (1 + s) * math.pi / 2) ** 2
    ac = (f / f[0])[1:]  # K entries for k=1..K
    return ac.clamp(1e-5, 0.9999).float()


class TokenDiffusion(nn.Module):
    """noise level k in {0..K}: k=0 means clean; alphas_cumprod indexed by k-1 for k>=1."""

    def __init__(self, model: nn.Module, K: int = 20, n_hist: int = 5, uncond_prob: float = 0.1, clip_noise: float = 1.0,
                 hist_noise_k: int = 0, hist_loss_w: float = 0.0):
        """hist_noise_k > 0: during training, history tokens get a random noise level in [0, hist_noise_k]
        (UniPhys-style stabilization: makes the policy tolerant to its own state drift). At sampling the history is
        presented at level min(hist_noise_k, stab_level) via `stab_level`. hist_loss_w: weight of the loss on history tokens."""
        super().__init__()
        self.model, self.K, self.n_hist, self.uncond_prob, self.clip_noise = model, K, n_hist, uncond_prob, clip_noise
        self.hist_noise_k, self.hist_loss_w = hist_noise_k, hist_loss_w
        ac = torch.cat([torch.ones(1), cosine_alphas_cumprod(K)])  # index k directly, ac[0]=1 (clean)
        self.register_buffer("ac", ac, persistent=False)

    # --- forward process helpers (per token) ---
    def _coef(self, k: torch.Tensor):
        a = self.ac[k][..., None]
        return a.sqrt(), (1 - a).sqrt()

    def q_sample(self, x0, k, noise):
        sa, sb = self._coef(k)
        return sa * x0 + sb * noise

    def v_target(self, x0, k, noise):
        sa, sb = self._coef(k)
        return sa * noise - sb * x0

    def x0_from_v(self, xk, k, v):
        sa, sb = self._coef(k)
        return sa * xk - sb * v

    def eps_from_v(self, xk, k, v):
        sa, sb = self._coef(k)
        return sb * xk + sa * v

    # --- training ---
    def loss(self, x0: torch.Tensor, text: torch.Tensor | None):
        """x0: (B,T,D) normalized tokens; text: (B,512). Returns scalar loss and per-token loss (B,T)."""
        B, T, D = x0.shape
        k = torch.randint(1, self.K + 1, (B, T), device=x0.device)
        k[:, : self.n_hist] = torch.randint(0, self.hist_noise_k + 1, (B, self.n_hist), device=x0.device) if self.hist_noise_k > 0 else 0
        noise = torch.randn_like(x0).clamp(-self.clip_noise, self.clip_noise) if self.clip_noise else torch.randn_like(x0)
        xk = self.q_sample(x0, k, noise)
        uncond = (torch.rand(B, device=x0.device) < self.uncond_prob) if text is not None else torch.zeros(B, dtype=torch.bool, device=x0.device)
        text_in = None
        if text is not None:
            text_in = text.clone(); text_in[uncond] = 0.0
        v_pred = self.model(xk, k, text_in)
        tgt = self.v_target(x0, k, noise)
        per_tok = F.mse_loss(v_pred, tgt, reduction="none").mean(-1)  # (B,T)
        fut = per_tok[:, self.n_hist:]
        loss = fut.mean()
        if self.hist_loss_w > 0:
            loss = loss + self.hist_loss_w * per_tok[:, : self.n_hist].mean()
        return loss, per_tok

    # --- sampling ---
    @torch.no_grad()
    def sample(self, hist: torch.Tensor, n_future: int, text: torch.Tensor | None, steps: int = 2, guidance: float = 2.5, eta: float = 0.0, stab_level: int = 0):
        """hist: (B,H,D) clean history tokens (normalized). Returns full (B,H+n_future,D) with predicted future.
        stab_level > 0: tell the model the history sits at that noise level (scaled by sqrt(alpha_bar), no noise added)."""
        B, H, D = hist.shape
        dev = hist.device
        hist_in = hist * self.ac[stab_level].sqrt() if stab_level > 0 else hist
        x = torch.cat([hist_in, torch.randn(B, n_future, D, device=dev).clamp(-self.clip_noise, self.clip_noise)], 1)
        ks = torch.linspace(self.K, 0, steps + 1).round().long().tolist()  # e.g. [20, 10, 0]
        for i in range(steps):
            k_cur, k_next = ks[i], ks[i + 1]
            k = torch.zeros(B, H + n_future, dtype=torch.long, device=dev); k[:, :H] = stab_level; k[:, H:] = k_cur
            v_u = self.model(x, k, None, force_uncond=True)
            v = v_u
            if text is not None and guidance != 0:
                v_c = self.model(x, k, text)
                v = v_u + guidance * (v_c - v_u)
            x0 = self.x0_from_v(x, k, v)
            eps = self.eps_from_v(x, k, v)
            if k_next == 0:
                x_new = x0
            else:
                a_next = self.ac[k_next]
                sigma = eta * (((1 - self.ac[k_cur] / a_next) * (1 - a_next) / (1 - self.ac[k_cur])).sqrt())
                x_new = a_next.sqrt() * x0 + (1 - a_next - sigma**2).sqrt() * eps + sigma * torch.randn_like(x0)
            x = torch.cat([hist_in, x_new[:, H:]], 1)
        return torch.cat([hist, x[:, H:]], 1)
