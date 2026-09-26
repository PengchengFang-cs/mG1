"""Rectified-flow helpers for the CodeFlow policy (docs/08 §10).

Same conventions as everywhere else in this project: t = 1 is clean, the network predicts x0, the loss is taken
in VELOCITY space with the 1/clamp(1-t, v_eps) factor, and the observed part of the window is never noised.
MoGeFlow averages the loss per group, so here it is averaged over the six residual LEVELS -- which keeps the
deep, low-energy levels from being drowned out by level 1.

**Flow loss only.** MoGeFlow's released HumanML3D recipe sets `terminal loss 0.0` (vendor_mogeflow/README.md:186),
so there is no code-classification head and no snapping term in the objective; snapping happens only at sampling
time (`CodeFlowPolicy.snap`).
"""
import torch

from hml_phys.intent_flow import build_state_elem, velocity_pair


def latent_mask(B, n_lat, n_lat_hist, device):
    """-> observed_mask [B, n_lat] (1 on the history latent frames) and gen [B, n_lat, 1] (1 where generated)."""
    obs = torch.zeros(B, n_lat, device=device)
    obs[:, :n_lat_hist] = 1.0
    return obs, (1.0 - obs)[..., None]


def logit_normal_t(B, device, generator=None):
    """MoGeFlow/SD3's logit-normal timestep schedule (docs/07 §17.1), mapped so that t = 1 is clean."""
    u = torch.randn(B, device=device, generator=generator)
    return torch.sigmoid(u)


def level_loss(v_hat, v, gen, n_quant, code_dim):
    """velocity error averaged inside each residual level, then over the levels (MoGeFlow's per-group averaging).

    v_hat, v : [B, T', Q*D]   gen : [B, T', 1]
    -> (scalar loss, dict level -> scalar)
    """
    e = (v_hat - v) ** 2 * gen
    e = e.reshape(*e.shape[:2], n_quant, code_dim)
    denom = gen.sum().clamp_min(1.0) * code_dim
    per = {f"L{q + 1}": e[:, :, q].sum() / denom for q in range(n_quant)}
    return torch.stack(list(per.values())).mean(), per


@torch.no_grad()
def sample_codes(model, x0_obs, obs, gen, text, text_u, scalars, num_steps=32, cfg_scale=3.5,
                 generator=None, extra=None, extra_u=None, extra_valid=None):
    """Euler on an ascending t grid with x0-space CFG, then snap to the codebooks.

    x0_obs : [B, T', Q*D] the observed (history) latents, clean; the generated rows are ignored.
    -> (codes [B, T', Q], x0 [B, T', Q*D])
    """
    B = x0_obs.shape[0]
    dev = x0_obs.device
    z = build_state_elem(x0_obs, gen, torch.zeros(B, device=dev), generator=generator)
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=dev)
    for i in range(num_steps):
        t = grid[i].expand(B)
        kw_c = dict(extra_tokens=extra, extra_valid=extra_valid) if extra is not None else {}
        kw_u = dict(extra_tokens=extra_u, extra_valid=extra_valid) if extra_u is not None else {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            x0_c = model(z, obs, t, text[0], text[1], text[2], scalars, **kw_c)
            if cfg_scale != 1.0:
                x0_u = model(z, obs, t, text_u[0], text_u[1], text_u[2], scalars, **kw_u)
            else:
                x0_u = None
        x0_c = x0_c.float()
        if x0_u is not None:
            x0_hat = x0_u.float() + cfg_scale * (x0_c - x0_u.float())   # CFG in x0 space (MotionCraft convention)
        else:
            x0_hat = x0_c
        v = (x0_hat - z) / (1.0 - grid[i]).clamp_min(0.05)
        z = z + v * (grid[i + 1] - grid[i])
        z = z * gen + x0_obs * (1 - gen)                            # the history never moves
    codes, snapped = model.snap(z)
    return codes, z
