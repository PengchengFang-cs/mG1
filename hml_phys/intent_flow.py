"""Flow utilities for route A (docs/07 §21). Same conventions as hml_phys/flow.py (t = 1 clean, x0 prediction,
velocity-space loss, Euler on an ascending grid, x0-space CFG).

Action-only policy: the policy token keeps its 435 channels, but in the future rows only the 69 action channels are
generated; every other element is imputed (history rows: observed values; future state channels: fixed zeros).
"""
import numpy as np
import torch

from hml_phys.tokens import action_channels_in_token, part_channels, PART_NAMES, ROOT_DIM, BODY_DIM

TOKEN_DIM = ROOT_DIM + BODY_DIM


def action_channel_mask(device):
    m = torch.zeros(TOKEN_DIM, device=device)
    m[torch.from_numpy(action_channels_in_token()).to(device)] = 1.0
    return m


def generated_elements(observed_mask, valid, act_mask):
    """[B,T,D] float: 1 where the policy generates (future, valid rows x action channels)."""
    fut = (1.0 - observed_mask) * valid
    return fut[..., None] * act_mask[None, None]


def policy_input(x0, observed_mask, act_mask):
    """zero the non-action channels of the future rows (never generated, never shown to the policy)."""
    fut = (1.0 - observed_mask)[..., None]
    return x0 * (1.0 - fut * (1.0 - act_mask[None, None]))


def build_state_elem(x0, gen, t, noise=None, generator=None):
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator) if noise is None else noise
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    z = tt * x0 + (1 - tt) * eps
    return z * gen + x0 * (1 - gen)


def velocity_pair(x0_hat, x0, z, t, v_eps=0.05):
    denom = (1.0 - t.view(-1, *([1] * (x0.dim() - 1)))).clamp_min(v_eps)
    return (x0_hat - z) / denom, (x0 - z) / denom


def action_part_indices(device):
    """per body part, the indices of its action channels (the root part owns none and is skipped)."""
    act = set(action_channels_in_token().tolist())
    out = {}
    for n, c in zip(PART_NAMES, part_channels()):
        a = [int(i) for i in c if int(i) in act]
        if a:
            out[n] = torch.tensor(a, device=device)
    return out


def policy_loss(v_hat, v, gen, part_idx):
    """MoGeFlow per-part averaging (docs/07 §17.1) restricted to the generated elements: mean squared velocity error
    over each part's action channels on generated rows, then the mean over the parts that own actions."""
    per = {}
    for n, idx in part_idx.items():
        g = gen[..., idx]
        e = ((v_hat[..., idx] - v[..., idx]) ** 2 * g).sum()
        per[n] = e / g.sum().clamp_min(1.0)
    return torch.stack(list(per.values())).mean(), per


def latent_loss(v_hat, v):
    return ((v_hat - v) ** 2).mean()


@torch.no_grad()
def sample_latent(net, B, mem, mem_valid, mem_u, mem_valid_u, num_steps=32, cfg_scale=3.5, generator=None,
                  prefix=None, scalars=None, extra=None, extra_u=None, device="cuda", n_lat=None, d_lat=None):
    """Euler + x0-space CFG over [B,n_lat,d_lat] intent latents. The unconditional branch uses (mem_u, extra_u).

    The shape defaults to the SMPL side's 4 x 32; G1's intent window is 28 frames, so its DiTs carry 7 latent
    frames and pass their own `net.n_lat` / `net.d_lat`."""
    from hml_phys.intent_model import N_LAT, D_LAT
    n_lat = int(n_lat if n_lat is not None else getattr(net, "n_lat", N_LAT))
    d_lat = int(d_lat if d_lat is not None else getattr(net, "d_lat", D_LAT))
    z = torch.randn((B, n_lat, d_lat), device=device, generator=generator)
    grid = torch.linspace(0, 1, num_steps + 1, device=device)
    cat = lambda a, b: None if a is None else torch.cat([a, b])
    for i in range(num_steps):
        t = grid[i].expand(B); dt = grid[i + 1] - grid[i]
        if cfg_scale != 1.0:
            x, _ = net(torch.cat([z, z]), torch.cat([t, t]), torch.cat([mem_u, mem]), torch.cat([mem_valid_u, mem_valid]),
                       prefix_latent=cat(prefix, prefix), scalars=cat(scalars, scalars),
                       mem_extra=None if extra is None else torch.cat([extra_u, extra]))
            x = x[:B] + cfg_scale * (x[B:] - x[:B])
        else:
            x, _ = net(z, t, mem, mem_valid, prefix_latent=prefix, scalars=scalars, mem_extra=extra)
        x = x.float()
        if i == num_steps - 1:
            return x
        z = z + dt * (x - z) / (1.0 - t.view(-1, 1, 1)).clamp_min(1e-4)


@torch.no_grad()
def sample_actions(policy, x_obs, observed_mask, gen, text, text_u, scalars, intent_tokens, num_steps=32,
                   cfg_scale=3.5, generator=None, valid=None, frame_index=None):
    """Euler + x0-space CFG for the action-only policy. Conditional branch: (text, intent tokens); unconditional
    branch: (CLIP(''), no intent tokens) -- the same state as a text-dropped training sample."""
    B = x_obs.shape[0]
    dev = x_obs.device
    fixed = 1.0 - gen
    z = torch.randn(x_obs.shape, device=dev, generator=generator, dtype=x_obs.dtype) * gen + x_obs * fixed
    grid = torch.linspace(0, 1, num_steps + 1, device=dev)
    K = intent_tokens.shape[1]
    ones = torch.ones(B, K, dtype=torch.bool, device=dev)
    for i in range(num_steps):
        t = grid[i].expand(B); dt = grid[i + 1] - grid[i]
        if cfg_scale != 1.0:
            cat2 = lambda a: None if a is None else torch.cat([a, a])
            x = policy(torch.cat([z, z]), cat2(observed_mask), torch.cat([t, t]),
                       torch.cat([text_u[0], text[0]]), torch.cat([text_u[1], text[1]]), torch.cat([text_u[2], text[2]]),
                       cat2(scalars), valid=cat2(valid), frame_index=cat2(frame_index),
                       extra_tokens=cat2(intent_tokens), extra_valid=torch.cat([~ones, ones]))
            x = x[:B] + cfg_scale * (x[B:] - x[:B])
        else:
            x = policy(z, observed_mask, t, text[0], text[1], text[2], scalars, valid=valid, frame_index=frame_index,
                       extra_tokens=intent_tokens, extra_valid=ones)
        x = x.float()
        if i == num_steps - 1:
            return x * gen + x_obs * fixed
        z = z + dt * (x - z) / (1.0 - t.view(-1, 1, 1)).clamp_min(1e-4)
        z = z * gen + x_obs * fixed


def intent_hidden(net, latent, s, generator=None, **kw):
    """Final-layer hidden states of an intent DiT read on a (partly) noised latent: z = s*latent + (1-s)*eps, t = s
    (conditioning augmentation, docs/07 §21.8). s == 1.0 is the clean-latent read-out of §21.4-4 and draws no noise,
    so checkpoints trained without augmentation behave exactly as before. s: python float or [B] tensor."""
    B = latent.shape[0]
    if isinstance(s, float) and s == 1.0:
        return net(latent, torch.ones(B, device=latent.device), **kw)[1]
    s = s if torch.is_tensor(s) else torch.full((B,), float(s), device=latent.device)
    eps = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    z = s.view(-1, 1, 1) * latent + (1 - s.view(-1, 1, 1)) * eps
    return net(z, s, **kw)[1]
