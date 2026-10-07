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


def generated_elements(observed_mask, valid, act_mask):
    """[B,T,72] float: 1 where the policy generates -- future rows x action channels.

    Argument order matches intent_flow.generated_elements deliberately. Anyone writing a G1 sampler by
    copying a reference call site would otherwise pass `valid` where `act_mask` is expected and get a
    silently wrong mask instead of an error. `valid` may be None when every row is valid.
    """
    fut = 1.0 - observed_mask
    if valid is not None:
        fut = fut * valid
    return fut[..., None] * act_mask[None, None]


def policy_input(x0, observed_mask, act_mask, n_future_proprio=0):
    """Zero the proprio channels of the future rows the closed loop cannot observe.

    `n_future_proprio` is how many future rows keep their proprio, and it MUST match the dataset's
    `obs_future` ("first" -> 1, "none" -> 0). Row H is the state the loop IS in when it plans: with our
    recorder's pairing (row t = state the action was applied in, g1e2e_record_rollouts.py:196-199) the
    action to emit is a_H, which is applied in s_H, so withholding row H's proprio leaves the policy
    predicting a feedback controller's output 40-160 ms ahead of the state that controller reacts to.

    An earlier version took no such argument and zeroed EVERY row with observed_mask == 0, row H
    included. That made `obs_future="first"` bit-identical to `"none"` and destroyed the one channel
    `G1E2EWindows.mask_future_proprio` deliberately keeps; the closed loop zeroed it too, so training and
    inference agreed on the broken version and no mismatch check could find it. Three independent code
    reviews on 2026-10-02 identified it as the primary defect behind 512/512 falls.
    """
    fut = 1.0 - observed_mask
    # cumsum over the future rows numbers them 1, 2, 3, ...; keep the first n, zero the rest. Done with
    # tensor ops rather than an int(H) so there is no device sync in the training loop.
    zero_prop = fut * (fut.cumsum(1) > n_future_proprio).to(x0.dtype)
    return x0 * (1.0 - zero_prop[..., None] * (1.0 - act_mask[None, None]))


def build_state_elem(x0, gen, t, noise=None, generator=None):
    """Per-ELEMENT noising: only generated elements are noised, observed ones stay clean (x0 prediction)."""
    if noise is None:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    tb = t[:, None, None]
    z = tb * x0 + (1.0 - tb) * noise
    return x0 * (1.0 - gen) + z * gen, noise


def velocity_pair(x0_hat, x0, z, t, v_eps=0.05):
    """Same as flow.velocity_pair and intent_flow.velocity_pair: compare velocities rather than x0.

    The DENOMINATOR is floored, not t. Flooring t instead leaves 1/(1-t) unbounded: at t -> 1-1e-4 the
    denominator reaches 1e-4, a 500x velocity inflation and 2.5e5x on the squared error for that sample.
    With the logit-normal default (p_mean -0.8) the largest t drawn in 200k samples was 0.949, so the two
    forms differ by at most 1.04x here -- but `dist="uniform"` or a positive p_mean puts ~5% of samples
    past 0.95 and the difference becomes severe. Bounded at 1/v_eps = 20, as the reference is.
    """
    denom = (1.0 - t[:, None, None]).clamp_min(v_eps)
    return (x0_hat - z) / denom, (x0 - z) / denom


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
