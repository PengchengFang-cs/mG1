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

# The 21 actuated joints in the order the URDF and hml_phys/g1_21dof_config.py both give them.
DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
_I = {n: i for i, n in enumerate(DOF_NAMES)}

# Which joints the residual may move. RobotDancing (arXiv 2509.20717) Table III measures this
# directly and it is the single largest effect in that paper: against an absolute-action baseline,
# a residual on ALL DoFs cut global position error by 4.5%, while a residual restricted to bilateral
# hip/knee PITCH cut it by 15.7% -- more than three times as much. Our own failure attribution
# points at the same place independently: the 19 clips only we fail have reference vertical-velocity
# peaks 1.8x the succeeding group's (STATUS.md §5.8), i.e. they look like leg problems.
MASKS = {
    "all": None,                                          # every joint, the default
    "hipknee": ["left_hip_pitch_joint", "left_knee_joint",       # RobotDancing's exact set
                "right_hip_pitch_joint", "right_knee_joint"],
    "legs": ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
             "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"],
    "legs_all": DOF_NAMES[:12],                           # both legs, all six joints each
}


def mask_indices(name):
    """-> a list of DoF indices, or None for 'all'."""
    assert name in MASKS, f"unknown --residual-mask {name!r}; choose from {sorted(MASKS)}"
    names = MASKS[name]
    return None if names is None else [_I[n] for n in names]


class Residual(nn.Module):
    """Diagonal-Gaussian correction on the base action. tanh-squashed and scaled so it can shift an
    action by at most `scale` in raw units: a correction, never a replacement."""

    def __init__(self, res_in, action_dim, scale, init_log_std=-2.0, min_log_std=-4.0,
                 active=None):
        super().__init__()
        self.res_in, self.action_dim = int(res_in), int(action_dim)
        self.scale, self.min_log_std = float(scale), float(min_log_std)
        # When the residual is restricted to a subset of joints, the Gaussian itself is restricted:
        # the head emits only those dimensions and they are scattered back into a zero vector. Keeping
        # 21 dimensions and multiplying by a 0/1 mask would leave PPO exploring -- and paying variance
        # on -- dimensions that provably cannot change the outcome.
        self.active = None if active is None else [int(i) for i in active]
        n_out = action_dim if self.active is None else len(self.active)
        if self.active is not None:
            self.register_buffer("idx", torch.tensor(self.active, dtype=torch.long))
        self.pi = nn.Sequential(nn.Linear(res_in, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, n_out))
        self.vf = nn.Sequential(nn.Linear(res_in, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((n_out,), float(init_log_std)))
        nn.init.zeros_(self.pi[-1].weight)
        nn.init.zeros_(self.pi[-1].bias)      # start as an exact no-op on top of the base policy

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.clamp_min(self.min_log_std).exp())

    def value(self, x):
        return self.vf(x).squeeze(-1)

    def squash(self, u):
        d = torch.tanh(u) * self.scale
        if self.active is None:
            return d
        out = torch.zeros(*d.shape[:-1], self.action_dim, device=d.device, dtype=d.dtype)
        out[..., self.idx] = d
        return out

    def act_mean(self, x):
        """The deterministic correction -- what evaluation applies. No exploration noise."""
        return self.squash(self.pi(x))


def load_residual(path, device, action_dim):
    """Rebuild a trained residual from its checkpoint, with its own recorded input width and scale."""
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    res = Residual(int(ck["res_in"]), action_dim, a["residual_scale"],
                   a["init_log_std"], a.get("min_log_std", -4.0),
                   active=ck.get("active")).to(device)
    res.load_state_dict(ck["model"])
    res.eval().requires_grad_(False)
    return res, ck
