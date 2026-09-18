"""Rectified flow for the physics policy — MotionCraft conventions (flow_schedule.py / train_hy273_raw_flow.py):
  t = 1 clean, t = 0 noise;  z_t = t*x0 + (1-t)*eps;  v_target = x0 - eps
  network predicts x0;  loss in velocity space with both sides divided by clamp(1-t, velocity_t_eps)
  observed frames are hard-imputed (z_imp = z*(1-m) + x0*m) and excluded from the flow loss.
Sampling: Euler on the ascending grid 0->1 (num_steps), x0-space CFG, observed frames re-imposed each step.
"""
import torch
import torch.nn.functional as F


def sample_t(batch, device, p_mean=-0.8, p_std=0.8, eps=1e-4, generator=None):
    n = torch.randn(batch, device=device, generator=generator)
    return torch.sigmoid(n * p_std + p_mean).clamp(eps, 1 - eps)


def build_state(x0, mask_frames, t, noise=None, generator=None):
    """x0 [B,T,D], mask_frames [B,T] (1 = observed), t [B]. -> z_imp, eps, v_target"""
    eps = (torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator) if noise is None else noise)
    tt = t.view(-1, 1, 1)
    z = tt * x0 + (1 - tt) * eps
    m = mask_frames[..., None].to(x0.dtype)
    z_imp = z * (1 - m) + x0 * m
    return z_imp, eps, x0 - eps


def velocity_pair(x0_hat, x0, z_imp, t, v_eps=0.05):
    denom = (1.0 - t.view(-1, 1, 1)).clamp_min(v_eps)
    return (x0_hat - z_imp) / denom, (x0 - z_imp) / denom


def masked_mse(pred, target, mask_frames, channel_weight=None, valid=None):
    """mean over unmasked (future) AND valid frames; optional per-channel weights [D]."""
    err = (pred - target) ** 2
    if channel_weight is not None:
        err = err * channel_weight.view(1, 1, -1)
    m = (1 - mask_frames)
    if valid is not None:
        m = m * valid
    m = m[..., None].to(err.dtype)
    return (err * m).sum() / (m.sum() * err.shape[-1]).clamp_min(1.0)


def masked_smooth_l1(pred, target, mask):
    err = F.smooth_l1_loss(pred, target, reduction="none")
    m = mask.to(err.dtype)
    while m.dim() < err.dim():
        m = m[..., None]
    return (err * m).sum() / (m.expand_as(err).sum()).clamp_min(1.0)


@torch.no_grad()
def euler_sample(model, x_obs_root, x_obs_body, mask_frames, text, text_uncond, scalars, num_steps=32, cfg_scale=3.5,
                 v_eps=1e-4, generator=None, valid=None, frame_index=None):
    """x_obs_* : tokens with the history filled (future entries ignored). text / text_uncond: (tokens, pooled, len).
    Returns x0 prediction after the last step (root, body) with observed frames imposed."""
    B, T, _ = x_obs_root.shape
    dev = x_obs_root.device
    m = mask_frames[..., None].to(x_obs_root.dtype)
    z_root = torch.randn(x_obs_root.shape, device=dev, generator=generator, dtype=x_obs_root.dtype) * (1 - m) + x_obs_root * m
    z_body = torch.randn(x_obs_body.shape, device=dev, generator=generator, dtype=x_obs_body.dtype) * (1 - m) + x_obs_body * m
    grid = torch.linspace(0, 1, num_steps + 1, device=dev)
    for i in range(num_steps):
        t = grid[i].expand(B); dt = grid[i + 1] - grid[i]
        if cfg_scale != 1.0:
            zr = torch.cat([z_root, z_root]); zb = torch.cat([z_body, z_body]); mm = torch.cat([mask_frames, mask_frames])
            tok = torch.cat([text_uncond[0], text[0]]); po = torch.cat([text_uncond[1], text[1]]); ln = torch.cat([text_uncond[2], text[2]])
            sc = torch.cat([scalars, scalars]); tt = torch.cat([t, t]); vv = None if valid is None else torch.cat([valid, valid])
            fi = None if frame_index is None else torch.cat([frame_index, frame_index])
            xr, xb = model(zr, zb, mm, tt, tok, po, ln, sc, valid=vv, frame_index=fi)
            xr = xr[:B] + cfg_scale * (xr[B:] - xr[:B]); xb = xb[:B] + cfg_scale * (xb[B:] - xb[:B])
        else:
            xr, xb = model(z_root, z_body, mask_frames, t, text[0], text[1], text[2], scalars, valid=valid, frame_index=frame_index)
        if i == num_steps - 1:
            x0r, x0b = xr, xb
            break
        denom = (1.0 - t.view(-1, 1, 1)).clamp_min(v_eps)
        z_root = z_root + dt * (xr - z_root) / denom
        z_body = z_body + dt * (xb - z_body) / denom
        z_root = z_root * (1 - m) + x_obs_root * m
        z_body = z_body * (1 - m) + x_obs_body * m
    x0r = x0r * (1 - m) + x_obs_root * m
    x0b = x0b * (1 - m) + x_obs_body * m
    return x0r, x0b
