"""Flow helpers for the end-to-end G1 policy, mirroring `hml_phys/intent_flow.py`.

That module is bound to the SMPL token layout -- its TOKEN_DIM, `action_channels_in_token()` and
`PART_NAMES` all describe the 435-d physics token and its six body groups. The G1 token is a different
shape, so these are the same operations over our own layout rather than an edit to the MIND line's copy:

    token 72 = proprio 51 | action 21        actions are the LAST 21 channels

Conventions kept identical to the MIND line, because the policy and the loss were tuned together:
x0 prediction, the velocity-space loss of `flow.velocity_pair`, logit-normal t, and the loss averaged over
generated elements only. The per-part averaging is replaced by a single group: G1's 21 joints would need a
body-part split defined from scratch, and the MIND line already measured that part structure does not pay.
"""
import numpy as np
import torch

PROPRIO_DIM = 51
ACTION_DIM = 21
TOKEN_DIM = PROPRIO_DIM + ACTION_DIM


def action_channel_mask(device):
    """[72] float, 1 on the action channels."""
    m = torch.zeros(TOKEN_DIM, device=device)
    m[PROPRIO_DIM:] = 1.0
    return m


def generated_elements(observed_mask, act_mask, valid=None):
    """[B,T,72] float: 1 where the policy generates -- future rows x action channels."""
    fut = 1.0 - observed_mask
    if valid is not None:
        fut = fut * valid
    return fut[..., None] * act_mask[None, None]


def policy_input(x0, observed_mask, act_mask):
    """Zero the non-action channels of the future rows: never generated, never shown.

    This is the same guard `G1E2EWindows.mask_future_proprio` applies when building a window; doing it here
    too means a window built with obs_future="all" (for a diagnostic) still cannot leak into the loss path.
    """
    fut = (1.0 - observed_mask)[..., None]
    return x0 * (1.0 - fut * (1.0 - act_mask[None, None]))


def build_state_elem(x0, gen, t, noise=None, generator=None):
    """Per-ELEMENT noising: only generated elements are noised, observed ones stay clean (x0 prediction)."""
    if noise is None:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    tb = t[:, None, None]
    z = tb * x0 + (1.0 - tb) * noise
    return x0 * (1.0 - gen) + z * gen, noise


def velocity_pair(x0_hat, x0, z, t, v_eps=0.05):
    """Same as flow.velocity_pair: compare velocities rather than x0, with t floored at v_eps."""
    tb = t[:, None, None].clamp_min(v_eps)
    return (x0_hat - z) / (1.0 - tb + 1e-8), (x0 - z) / (1.0 - tb + 1e-8)


def policy_loss(v_hat, v, gen):
    """Mean squared velocity error over the generated elements only."""
    e = ((v_hat - v) ** 2 * gen).sum()
    return e / gen.sum().clamp_min(1.0)


def latent_loss(v_hat, v):
    return ((v_hat - v) ** 2).mean()


def observed_mask(B, H, T, device):
    """[B,T] with 1 on the H history rows."""
    m = torch.zeros(B, T, device=device)
    m[:, :H] = 1.0
    return m
