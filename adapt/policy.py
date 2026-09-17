"""Closed-loop diffusion policy: history buffer of tokens [a_{j-1}, o_j], sample future, execute a_t."""
from __future__ import annotations
import numpy as np
import torch
from adapt.data import ACT_DIM, OBS_DIM, TOKEN_DIM
from adapt.model import DenoiserTransformer
from adapt.diffusion import TokenDiffusion


def build_obs_torch(v, w, g, q, qd, prev_a, zero_lin_vel=True):
    if zero_lin_vel:
        v = torch.zeros_like(v)
    return torch.cat([v, 0.2 * w, g, q, 0.05 * qd, prev_a], dim=-1)


class DiffusionPolicy:
    def __init__(self, ckpt_path: str, device="cuda", steps=2, guidance=2.5, n_future=15, use_ema=True, exec_steps=1, stab_level=0, solver="euler"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ck["args"]
        if a.get("arch", "xattn") == "adaln":
            from adapt.model import AdaLNDenoiser
            self.model = AdaLNDenoiser(d_model=a["d_model"], n_layers=a["layers"], n_loops=a.get("n_loops", 1)).to(device)
        else:
            self.model = DenoiserTransformer(d_model=a["d_model"], n_layers=a["layers"]).to(device)
        self.model.load_state_dict(ck["ema"] if use_ema else ck["model"]); self.model.eval()
        if a.get("gen", "ddpm") == "flow":
            from adapt.flow import TokenFlow
            self.dm = TokenFlow(self.model, n_hist=a["n_hist"], time_schedule=a.get("flow_schedule", "logit_normal"), p_mean=a.get("flow_p_mean", 0.0), p_std=a.get("flow_p_std", 1.0), pred=a.get("flow_pred", "v"), v_eps=a.get("flow_v_eps", 1e-2), loss_space=a.get("flow_loss_space", "v")).to(device)
        else:
            self.dm = TokenDiffusion(self.model, K=a["K"], n_hist=a["n_hist"]).to(device)
        self.n_hist, self.n_future, self.steps, self.guidance = a["n_hist"], n_future, steps, guidance
        self.mean = torch.as_tensor(ck["token_mean"], device=device); self.std = torch.as_tensor(ck["token_std"], device=device)
        self.device = device
        self.buf = None  # (N, n_hist, TOKEN_DIM) raw tokens
        self.prev_a = None
        self.exec_steps = exec_steps; self.plan = None; self.plan_i = 0; self.stab_level = stab_level; self.solver = solver

    @torch.inference_mode()
    def reset(self, n_envs: int, env_ids=None):
        if self.buf is None or env_ids is None:
            self.buf = torch.zeros(n_envs, self.n_hist, TOKEN_DIM, device=self.device)
            self.prev_a = torch.zeros(n_envs, ACT_DIM, device=self.device)
            self.filled = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        else:
            self.buf[env_ids] = 0; self.prev_a[env_ids] = 0; self.filled[env_ids] = 0

    @torch.no_grad()
    def act(self, obs96: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """obs96: (N,96) current observation (already contains prev action). text_emb: (N,512). Returns a_t (N,29)."""
        tok = torch.cat([self.prev_a, obs96], dim=-1)  # token_t = [a_{t-1}, o_t]
        self.buf = torch.roll(self.buf, shifts=-1, dims=1); self.buf[:, -1] = tok
        self.filled = torch.clamp(self.filled + 1, max=self.n_hist)
        # envs with fewer than n_hist real tokens: replicate the oldest real token backwards
        for i in torch.where(self.filled < self.n_hist)[0].tolist():
            f = int(self.filled[i]); self.buf[i, : self.n_hist - f] = self.buf[i, self.n_hist - f]
        if self.plan is None or self.plan_i >= self.exec_steps:
            hist = (self.buf - self.mean) / self.std
            kw = {"solver": self.solver} if hasattr(self.dm, "time_schedule") else {}
            out = self.dm.sample(hist, self.n_future, text_emb, steps=self.steps, guidance=self.guidance, stab_level=self.stab_level, **kw)
            fut = out[:, self.n_hist:] * self.std + self.mean        # future tokens j=t+1.. = [a_t, o_{t+1}], ...
            self.plan = fut[:, :, :ACT_DIM]; self.plan_i = 0
        a_t = self.plan[:, self.plan_i]; self.plan_i += 1
        self.prev_a = a_t.clone()
        return a_t
