"""Rectified-flow (flow matching) variant of TokenDiffusion with the same interface.
Per-token continuous time t in [0,1]: t=1 clean data, t=0 pure noise (history tokens fixed at t=1).
x_t = t * x1 + (1-t) * x0_noise ; velocity target v = x1 - x0_noise ; model predicts v.
Timestep sampling: logit-normal (MoGeFlow) or uniform. Sampling: Euler from t=0 to 1 with CFG.
The denoiser receives k = round(t * k_scale) as its (float) noise-level input.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFlow(nn.Module):
    def __init__(self, model: nn.Module, n_hist: int = 5, uncond_prob: float = 0.1, clip_noise: float = 1.0,
                 time_schedule: str = "logit_normal", p_mean: float = 0.0, p_std: float = 1.0, k_scale: float = 1000.0,
                 hist_noise_t: float = 0.0, hist_loss_w: float = 0.0, pred: str = "v", v_eps: float = 1e-2, loss_space: str = "v"):
        """pred: 'v' (network outputs velocity) or 'x0' (network outputs the clean token; velocity derived as
        (x0_hat - x_t)/(1-t), MotionCraft style). loss_space: 'v' or 'x0' (space in which the MSE is taken)."""
        super().__init__()
        self.model, self.n_hist, self.uncond_prob, self.clip_noise = model, n_hist, uncond_prob, clip_noise
        self.time_schedule, self.p_mean, self.p_std, self.k_scale = time_schedule, p_mean, p_std, k_scale
        self.hist_noise_t, self.hist_loss_w = hist_noise_t, hist_loss_w   # history tokens sampled at t in [1-hist_noise_t, 1]
        self.pred, self.v_eps, self.loss_space = pred, v_eps, loss_space
        self.K = 1  # for compatibility with code that reads .K

    def _v_from_out(self, out, xt, t):
        """Convert network output to velocity (data - noise) and clean estimate."""
        if self.pred == "x0":
            x1_hat = out
            v = (x1_hat - xt) / (1.0 - t[..., None]).clamp_min(self.v_eps)
        else:
            v = out
            x1_hat = xt + (1.0 - t[..., None]) * v
        return v, x1_hat

    def _sample_t(self, shape, device):
        if self.time_schedule == "logit_normal":
            return torch.sigmoid(torch.randn(shape, device=device) * self.p_std + self.p_mean)
        return torch.rand(shape, device=device)

    def _k(self, t):  # noise-level input for the denoiser (float, larger = cleaner, like 1000*t)
        return t * self.k_scale

    def loss(self, x1: torch.Tensor, text: torch.Tensor | None):
        B, T, D = x1.shape
        t = self._sample_t((B, T), x1.device)
        if self.hist_noise_t > 0:
            t[:, : self.n_hist] = 1.0 - torch.rand(B, self.n_hist, device=x1.device) * self.hist_noise_t
        else:
            t[:, : self.n_hist] = 1.0
        x0 = torch.randn_like(x1)
        if self.clip_noise: x0 = x0.clamp(-self.clip_noise, self.clip_noise)
        xt = t[..., None] * x1 + (1 - t[..., None]) * x0
        text_in = None
        if text is not None:
            uncond = torch.rand(B, device=x1.device) < self.uncond_prob
            text_in = text.clone(); text_in[uncond] = 0.0
        out = self.model(xt, self._k(t), text_in)
        v_pred, x1_hat = self._v_from_out(out, xt, t)
        if self.loss_space == "x0":
            per_tok = F.mse_loss(x1_hat, x1, reduction="none").mean(-1)
        else:
            per_tok = F.mse_loss(v_pred, x1 - x0, reduction="none").mean(-1)
        loss = per_tok[:, self.n_hist:].mean()
        if self.hist_loss_w > 0:
            loss = loss + self.hist_loss_w * per_tok[:, : self.n_hist].mean()
        return loss, per_tok

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, n_future: int, text: torch.Tensor | None, steps: int = 2, guidance: float = 2.5, eta: float = 0.0, stab_level: float = 0.0, solver: str = "euler"):
        """ODE integration t: 0 -> 1 on future tokens (solver: euler | heun); history fixed at t=1 (or 1-stab_level)."""
        B, H, D = hist.shape; dev = hist.device
        t_hist = 1.0 - float(stab_level)
        hist_in = hist if stab_level <= 0 else (t_hist * hist + (1 - t_hist) * torch.randn_like(hist).clamp(-self.clip_noise, self.clip_noise))
        x = torch.cat([hist_in, torch.randn(B, n_future, D, device=dev).clamp(-self.clip_noise, self.clip_noise)], 1)
        ts = torch.linspace(0, 1, steps + 1, device=dev)
        def vel(x, t_scalar):
            t = torch.full((B, H + n_future), t_scalar, device=dev); t[:, :H] = t_hist
            out = self.model(x, self._k(t), None, force_uncond=True)
            if text is not None and guidance != 0:
                out = out + guidance * (self.model(x, self._k(t), text) - out)   # CFG in the network's output space
            v, _ = self._v_from_out(out, x, t)
            return v
        for i in range(steps):
            t_cur, t_next = float(ts[i]), float(ts[i + 1]); dt = t_next - t_cur
            v1 = vel(x, t_cur)
            if solver == "heun" and t_next < 1.0:
                x_pred = torch.cat([hist_in, (x + dt * v1)[:, H:]], 1)
                v2 = vel(x_pred, t_next)
                x_new = x + dt * 0.5 * (v1 + v2)
            else:
                x_new = x + dt * v1
            x = torch.cat([hist_in, x_new[:, H:]], 1)
        return torch.cat([hist, x[:, H:]], 1)
