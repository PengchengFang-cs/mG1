"""The residual correction head that `scripts/g1e2e_train_residual_ppo.py` trains and
`scripts/g1e2e_eval_closed_loop.py --residual` evaluates.

It lives here rather than inside the training script so the two cannot drift apart: the evaluation
rebuilds the head from the checkpoint's own recorded shape and scale, and a mismatch is an assertion
rather than a silently different controller.

    input   normalised proprio (51) | normalised base action (21) | reference block (27) | progress | duration/10
    output  a tanh-squashed Gaussian correction in RAW action units, bounded by +-scale

The reference block is in the input because the reward is a function of it: without it neither the
actor nor the critic can observe what is being optimised.
"""
import torch
import torch.nn as nn


class Residual(nn.Module):
    """Diagonal-Gaussian correction on the base action. tanh-squashed and scaled so it can shift an
    action by at most `scale` in raw units: a correction, never a replacement."""

    def __init__(self, res_in, action_dim, scale, init_log_std=-2.0, min_log_std=-4.0):
        super().__init__()
        self.res_in, self.action_dim = int(res_in), int(action_dim)
        self.scale, self.min_log_std = float(scale), float(min_log_std)
        self.pi = nn.Sequential(nn.Linear(res_in, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, action_dim))
        self.vf = nn.Sequential(nn.Linear(res_in, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        nn.init.zeros_(self.pi[-1].weight)
        nn.init.zeros_(self.pi[-1].bias)      # start as an exact no-op on top of the base policy

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.clamp_min(self.min_log_std).exp())

    def value(self, x):
        return self.vf(x).squeeze(-1)

    def squash(self, u):
        return torch.tanh(u) * self.scale

    def act_mean(self, x):
        """The deterministic correction -- what evaluation applies. No exploration noise."""
        return self.squash(self.pi(x))


def load_residual(path, device, action_dim):
    """Rebuild a trained residual from its checkpoint, with its own recorded input width and scale."""
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    res = Residual(int(ck["res_in"]), action_dim, a["residual_scale"],
                   a["init_log_std"], a.get("min_log_std", -4.0)).to(device)
    res.load_state_dict(ck["model"])
    res.eval().requires_grad_(False)
    return res, ck
