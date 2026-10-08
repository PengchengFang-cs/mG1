"""Residual PPO on top of the frozen behaviour-cloned G1 policy.

WHY A RESIDUAL AND NOT END-TO-END PPO. Our BC policy is a flow-matching model: one action row costs a
10-step Euler integration with classifier-free guidance. Measured throughput in the closed loop is
512 envs x 505 control steps in 1200 s = 215 env-steps/s, where a standard humanoid PPO run sits above
1e5. End-to-end PPO on the flow policy is therefore out of reach on two cards, and it would also need
diffusion-RL machinery (a tractable log-prob through the denoising chain) that we have no reason to take
on. ResMimic (arXiv 2510.05070) does exactly what this script does instead: freeze the general motion
tracker, add a residual that conditions on the task, and train only the residual with ordinary PPO.

    frozen BC policy (flow, 10 steps, cfg_action 1.0)  ->  a_base      [25 Hz, held 2 control steps]
                      residual head (Gaussian MLP)     ->  delta       [50 Hz, every control step]
                                                 a = a_base + delta

The residual is an ordinary diagonal-Gaussian policy, so the PPO ratio is exact. Acting at 50 Hz while
the base acts at 25 Hz is deliberate: the fast feedback the base policy lacks is precisely what the
measurements say is missing (STATUS.md §5.5a).

REWARD = THE TEACHER'S OWN REWARD, which the env already computes and the first version of this script
threw away. `config_eval.yaml` loads `rewards_teleop_omnih2o_teacher.yaml` (OmniH2O Table 15), so
`env.step()`'s third return value is the full objective the tracker we are trying to beat was trained on:

    7 task terms   vr_3keypoints 50 | body_position_extend 30 | selected_joint_position 32 |
                   selected_joint_vel 16 | body_rotation 20 | body_vel 8 | body_ang_vel 8   (x dt)
    termination    -250, and `_reward_termination = reset_buf * ~time_out_buf`, so a clip that simply
                   ENDS is not penalised while a fall is
    16 regularizers  lower/upper action rate, torques, dof_acc, dof_vel, slippage, feet_ori,
                   in_the_air, orientation, stumble, feet_air_time, limits, contact forces

Three reasons this beats the hand-rolled three-term reward it replaces, all of them from the two code
reviews of 2026-10-07:

  1. `compute_reward()` runs at legged_robot.py:478, BEFORE `reset_idx` at :481. The hand-rolled reward
     read `obs` AFTER `env.step` returned, which for any terminating env is the FRESHLY RESET state --
     so a fall was paid the best tracking reward of the whole episode. Using `rew_buf` is immune.
  2. The hand-rolled reward kept 1.00 of the teacher's 3.28 of task reward and dropped the rest. All 12
     leg joints, pelvis yaw and every velocity constraint sat in its null space, and it carried no
     termination penalty against a field convention of -200..-250 (OmniH2O, CLOT, RuN, RobotDancing).
  3. Measured on the teacher's own recorded reference block, `max_j ||ref_j - robot_j||` has median
     0.428 m, of which the left virtual hand contributes a near-constant ~0.33 m offset that no
     controller can remove; `exp(-10 d)` therefore sat at 0.014 and the 0.05 alive bonus was the
     largest term in the reward. The teacher's kernels are calibrated against its own error scale.

Maximising the teacher's own objective is also exactly the right target: beating the teacher's survival
means collecting more of the reward the teacher was trained to collect. `--w-track` / `--w-alive` keep
the hand-rolled terms available as optional extras; both default to 0.

TERMINATION. Contact, gravity, and reference distance at --ref-dist, which defaults to the 1.5 m the
teacher was trained with rather than the 0.5 m success criterion -- see the flag's own help for the
measurement that settled it. `terminate_by_1time_motion` is left ON: the env then ends an episode when
the clip runs out, marks it in `time_out_buf`, and restarts that env on its own clip at t=0. The first
version of this script turned that OFF and emulated the clip end in Python without resetting the
simulator -- `motion_lib_base.py:741` clamps the phase, so the reference froze on its last frame and the
robot kept simulating against it for every step until it drifted 0.5 m away. That corrupted roughly a
fifth of all collected transitions, starting at iteration 2.

GAE treats a clip end as a CONTINUATION, not a terminal: the post-reset observation genuinely is the
next state in this auto-resetting task, time-to-go is observable to the critic (the progress scalar is
in the residual's input), and `_reward_termination` already withholds the -250 there. Only real
terminations -- falls, tilts, losing the reference -- cut the bootstrap.

WHAT THIS CANNOT FIX. 14 of the 512 clips fail for the teacher as well, and their references ask for
3.1 m/s root speed and 5.9 rad/s joint rates in 1.4 s -- outside what G1 can do (STATUS.md §5.8). The
reward would be pointing at an unrealisable target there. The reachable ceiling for this script is
498/512 = 97.3%; the 19 clips only we fail are the target.

Run on a compute node, inside a persistent Slurm step:
    python -u scripts/g1e2e_train_residual_ppo.py \
      --policy outputs/g1e2e/push_G_lr25e6/best.pt \
      --refs data/g1_e2e/refs_train_part1.pkl --text-cache data/g1_e2e/text_clipL14_full \
      --out outputs/g1e2e/residual_ppo --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H2H = REPO / "external/FRoM-W1/H-ACT/human2humanoid"
LEGGED_GYM = H2H / "legged_gym"
CFG_DIR = LEGGED_GYM / "legged_gym/cfg/cfg_g1"
CONTROL_HZ = 50

# The reference block's layout inside the env observation, as g1e2e_record_rollouts.py pins it.
REF_DIM = 27
REF_SLICE = slice(48, 48 + REF_DIM)
REF_DIFF = slice(0, 9)       # reference-next-frame minus robot position, 3 points x 3, heading frame
REF_VEL = slice(18, 27)      # reference body velocity, 3 points x 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="the frozen BC checkpoint")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--text-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--rollout-steps", type=int, default=24, help="control steps per PPO iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="stop the epoch loop early once the mean KL exceeds this; 0 disables")
    ap.add_argument("--residual-scale", type=float, default=0.3,
                    help="the residual is tanh-squashed and multiplied by this, in RAW action units. "
                         "The checkpoint's own action normaliser has per-joint std mean 0.94 "
                         "(min 0.42), so 0.3 is about 0.32 of one action std: a correction, not a "
                         "replacement. The closed-loop noise sweep (STATUS.md §5.6) puts relative "
                         "sigma <= 0.20 inside the resolution floor and sigma 0.45 clearly harmful, "
                         "so 0.3 raw is a saturation BOUND the residual should rarely reach.")
    ap.add_argument("--init-log-std", type=float, default=-2.0)
    ap.add_argument("--min-log-std", type=float, default=-4.0,
                    help="floor on log_std, so the ratio cannot blow up once the policy sharpens")
    ap.add_argument("--rew-scale", type=float, default=0.1,
                    help="pure rescaling of the env reward. The teacher's objective runs to ~3.3/step "
                         "positive and -5 on termination, so undiscounted returns reach several "
                         "hundred; 0.1 keeps the value targets in a range Adam handles without "
                         "changing the optimum.")
    ap.add_argument("--penalty-scale", type=float, default=0.5,
                    help="the teacher reward's penalty multiplier; 0.5 is config_eval.yaml's own value")
    ap.add_argument("--w-track", type=float, default=0.0,
                    help="OPTIONAL extra shaping on top of the env reward: exp(-track_k * mean point "
                         "error). Off by default -- the env reward already tracks all bodies.")
    ap.add_argument("--w-alive", type=float, default=0.0,
                    help="OPTIONAL per-step survival bonus. Off by default: OmniH2O, whose reward this "
                         "is, uses none, and the termination penalty already carries the pressure.")
    ap.add_argument("--w-rest", type=float, default=0.0,
                    help="OPTIONAL L1 joint-velocity cost where the reference is still. Off by "
                         "default: the env reward already pays dof_vel -0.004 and the teacher's own "
                         "mean|dof_vel| at rest is 0.52 rad/s, so this term taxed the balance "
                         "controller it was meant to protect.")
    ap.add_argument("--track-k", type=float, default=2.0,
                    help="only used when --w-track > 0. r_track = exp(-k * MEAN point error in m). "
                         "Mean, not max: the left virtual hand carries a ~0.33 m near-constant "
                         "retargeting offset and took 90%% of the max, so the max had almost no "
                         "usable gradient. k=2 spans the measured 0.2-0.6 m range.")
    ap.add_argument("--rest-vel-thresh", type=float, default=0.1,
                    help="the reference counts as still when its mean body speed is below this (m/s)")
    ap.add_argument("--ref-dist", type=float, default=1.5,
                    help="terminate when the MEAN body distance to the reference exceeds this. 1.5 m "
                         "is config_eval.yaml's own value and what the teacher was TRAINED with. "
                         "0.5 m -- the recorder's value, and OmniH2O's success CRITERION rather than "
                         "its training threshold -- was measured here to end 89%% of episodes after "
                         "only 80 of a 365-step clip (6.2 resets/step against the 1.40 whole clips "
                         "would give). Our policy is conditioned on text and has no reference, so a "
                         "0.5 m leash truncates most of every clip, spends the -250 on losing the "
                         "reference rather than on falling -- which is the only thing we measure "
                         "(CLAUDE.md §13) -- and never lets the residual see the back half of a clip. "
                         "0 switches the leash OFF entirely, which makes training terminate on exactly "
                         "what evaluation terminates on, so the logged fall fraction becomes "
                         "comparable to the reported fall rate and the residual sees whole clips. That "
                         "is safe HERE because the leash's job is to stop a policy trading the task "
                         "for survival, and a residual bounded at 0.32 of one action std on top of a "
                         "FROZEN policy that already does the task cannot stand still even if it "
                         "wanted to -- and the teacher's reward pays nothing for standing still, since "
                         "there is no alive bonus.")
    ap.add_argument("--num-steps", type=int, default=10, help="flow sampling steps for the base policy")
    ap.add_argument("--cfg-action", type=float, default=1.0,
                    help="the base policy's action guidance. 1.0 is plain conditional and the only "
                         "setting that keeps the task conditioning intact (STATUS.md §5.7d).")
    ap.add_argument("--cfg-scale", type=float, default=2.5, help="guidance for the intent predictors")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly after this much wall clock (0 = run all --iters)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt if it exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args, overrides = ap.parse_known_args()
    bad = [o for o in overrides if o.startswith("-")]
    assert not bad, f"unrecognised option(s) {bad}; hydra overrides are key=value, not flags"

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = Path(args.refs).resolve()
    policy_path = Path(args.policy).resolve()
    text_cache = Path(args.text_cache).resolve()
    for p in (refs, policy_path):
        assert p.exists(), p
    assert text_cache.is_dir(), text_cache
    assert (LEGGED_GYM / "resources/robots/g1/urdf/g1_21dof.urdf").exists(), "21-DoF asset missing"
    os.chdir(LEGGED_GYM)
    sys.path.insert(0, str(H2H))
    sys.path.insert(0, str(REPO))

    from isaacgym import gymapi          # noqa: E402  (before torch, on purpose)
    import numpy as np                   # noqa: E402
    import torch                         # noqa: E402
    import torch.nn as nn                # noqa: E402
    import joblib                        # noqa: E402
    import hydra                         # noqa: E402
    from omegaconf import OmegaConf       # noqa: E402
    from easydict import EasyDict         # noqa: E402
    import legged_gym.envs                # noqa: E402,F401
    from legged_gym.utils import task_registry   # noqa: E402

    from hml_phys.g1e2e_data import PROPRIO_DIM, ACTION_DIM, TOKEN_DIM   # noqa: E402
    from hml_phys.g1e2e_flow import (action_channel_mask, generated_elements,   # noqa: E402
                                     observed_mask)
    from hml_phys.intent_model import IntentPolicy                        # noqa: E402
    from hml_phys.intent_vae import IntentVAE                             # noqa: E402
    from hml_phys.intent_flow import (intent_hidden, sample_actions,      # noqa: E402
                                      sample_latent)
    from hml_phys.g1e2e_residual import Residual                          # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ck = torch.load(policy_path, map_location="cpu")
    ta = ck["args"]
    gen_hz, H, F = ta["gen_hz"], ta["H"], ta["F"]
    hold = CONTROL_HZ // gen_hz
    n_fut_prop = {"none": 0, "first": 1, "all": F}[ta.get("obs_future", "first")]
    print(f"base policy: gen {gen_hz} Hz, H {H}, F {F}, hold {hold}; residual acts every control step",
          flush=True)

    # Termination on reference distance is ON here, unlike evaluation: a policy that stops following is
    # ended, not merely scored down. `terminate_by_1time_motion` stays ON so the env itself ends and
    # restarts a clip that runs out -- see the module docstring on why emulating it in Python was wrong.
    # `resample_motions_for_envs` is pinned OFF: it fires at common_step_counter 50000 and would hand
    # every env a new clip while `text`, `pooled`, `dur_s` and `n_env` here still point at the old ones.
    # `penalty_curriculum` is pinned OFF so the reward is stationary over the run.
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_h = hydra.compose(config_name="config_eval", overrides=[
            f"motion.motion_file={refs}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            "headless=True", "use_wandb=False",
            f"asset.terminate_by_ref_motion_distance={args.ref_dist > 0}",
            f"asset.termination_scales.max_ref_motion_distance={max(args.ref_dist, 0.01)}",
            "asset.terminate_by_1time_motion=True",
            "motion.resample_motions_for_envs=False",
            "rewards.penalty_curriculum=False",
            f"rewards.penalty_scale={args.penalty_scale}",
            *overrides])
    cfg = EasyDict(OmegaConf.to_container(cfg_h, resolve=True))
    cfg.physics_engine = gymapi.SIM_PHYSX
    assert cfg.asset.terminate_by_ref_motion_distance is (args.ref_dist > 0)
    assert cfg.asset.terminate_by_1time_motion is True, \
        "the env must end a clip itself; see the docstring on the frozen-reference corruption"
    assert cfg.motion.resample_motions_for_envs is False
    assert cfg.rewards.penalty_curriculum is False
    n_obs = int(cfg.env.num_observations)
    n_hist_block = (int(cfg.env.short_history_length) * (int(cfg.extra.dof_num) * 3 + 6)
                    if cfg.env.add_short_history else 0)
    assert n_obs == 48 + REF_DIM + ACTION_DIM + n_hist_block, (
        f"observation layout moved: num_observations={n_obs}; REF_SLICE={REF_SLICE} assumes "
        f"{48 + REF_DIM + ACTION_DIM + n_hist_block}. Re-derive it from legged_robot.py.")
    assert cfg.motion.teleop_obs_version == "v-teleop-extend-vr-max-nolinvel", \
        cfg.motion.teleop_obs_version
    leash = f"{args.ref_dist} m ON" if args.ref_dist > 0 else ("OFF -- the only early end is a real "
                                                              "fall, exactly as in evaluation")
    print(f"termination: contacts {cfg.asset.terminate_after_contacts_on}, "
          f"gravity {cfg.asset.terminate_by_gravity}, reference distance {leash}, "
          f"clip end ON (flagged as time-out, no -250)", flush=True)

    env, _ = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg, env_cfg=cfg)
    dev = env.device
    B = env.num_envs
    # The reward is the env's own; record which terms are live so the log says what was optimised.
    live = {k: float(v) for k, v in env.reward_scales.items() if abs(float(v)) > 0}
    print(f"env reward: {len(live)} live terms, termination {live.get('termination')}, "
          f"penalty_scale {cfg.rewards.penalty_scale}", flush=True)
    assert "termination" in live, "the teacher reward set lost its termination penalty"

    # ---- frozen base policy -----------------------------------------------------------------------
    stats = np.load(policy_path.parent / "stats.npz")
    mean = torch.tensor(stats["mean"], device=dev)
    std = torch.tensor(stats["std"], device=dev)
    vae_path = Path(ta["vae"])
    if not vae_path.is_absolute():
        vae_path = REPO / vae_path
    vck = torch.load(vae_path, map_location="cpu")
    va = vck["args"]
    vae = IntentVAE(input_dim=vck["input_dim"], width=va["width"], down_t=va["down_t"], stride_t=2,
                    depth=va["depth"], dilation_growth_rate=va["dilation"], latent_dim=va["latent"]).to(dev)
    vae.load_state_dict(vck["model"])
    vae.eval().requires_grad_(False)
    base = IntentPolicy(ck["policy_kw"], ta["intent_dim"], ta["intent_heads"], ta["intent_depth"],
                        ta.get("intent_mlp", 4.0), text_token_dim=768).to(dev)
    base.load_state_dict(ck["model"])
    base.eval().requires_grad_(False)
    n_base = sum(p.numel() for p in base.parameters())
    print(f"frozen base: {n_base / 1e6:.1f} M params, step {ck.get('step')}", flush=True)

    tok = joblib.load(text_cache / "tokens.pkl")
    lens = joblib.load(text_cache / "lengths.pkl")
    pool = joblib.load(text_cache / "pooled.pkl")
    env.cfg.env.test = True
    env.begin_seq_motion_samples()
    lib = env._motion_lib
    ids = lib._curr_motion_ids.clone()
    keys = [str(k) for k in lib._motion_data_keys[ids.cpu().numpy()]]
    assert all(k in tok for k in keys), "some env clips have no cached caption"
    text = torch.stack([torch.tensor(tok[k][0], dtype=torch.float32) for k in keys]).to(dev)
    pooled = torch.stack([torch.tensor(pool[k][0], dtype=torch.float32) for k in keys]).to(dev)
    tlen = torch.tensor([int(lens[k][0]) for k in keys], device=dev)
    text_u = torch.tensor(tok["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(text)
    pooled_u = torch.tensor(pool["__uncond__"][0], dtype=torch.float32, device=dev)[None].expand_as(pooled)
    tlen_u = torch.full_like(tlen, int(lens["__uncond__"][0]))
    mem_c, mv_c = base.adapter(text, tlen)
    mem_u, mv_u = base.adapter(text_u, tlen_u)
    s_read = float(ta["cond_aug_test"])
    lat_st = np.load(policy_path.parent / "intent_latent_stats.npz")
    lat_mean = torch.tensor(lat_st["mean"], device=dev)
    lat_std = torch.tensor(lat_st["std"], device=dev)

    secs = lib.get_motion_length().clone()
    assert secs.shape[0] == B, f"{secs.shape[0]} motion lengths for {B} envs"
    dur_s = secs.to(dev).float()
    n_env = (dur_s * CONTROL_HZ).ceil().long().clamp_min(1)
    print(f"clips {B}: {float(dur_s.min()):.2f}-{float(dur_s.max()):.2f} s "
          f"(mean {float(dur_s.mean()):.2f})", flush=True)

    act_mask = action_channel_mask(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    hist = torch.zeros(B, H, TOKEN_DIM, device=dev)
    holi = [None, None]
    ep_step = torch.zeros(B, dtype=torch.long, device=dev)

    def read_prop():
        return torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                          env.dof_pos, env.dof_vel], dim=-1)

    def hold_action():
        """The raw action that holds the robot where it is -- what a reset env's history starts from."""
        return (env.dof_pos - env.default_dof_pos) / float(cfg.control.action_scale)

    def push_hist(prop, action_raw):
        nonlocal hist
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1, :PROPRIO_DIM] = (prop - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
        hist[:, -1, PROPRIO_DIM:] = (action_raw - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:]

    def sample_holistic(idx=None):
        """Draw the holistic intent ONCE per episode, for the given envs only.

        The first version called this for all B envs whenever ANY env reset, which with 512 envs is
        ~89% of control steps. That is the pre-fix bug `--holi-per-plan` exists to measure: it moves
        the long-horizon conditioning to a different mode mid-episode and makes the immediate intent
        incoherent with the chunk before it (STATUS.md §5.7).
        """
        nonlocal holi
        with torch.inference_mode():
            if idx is None:
                n, sl = B, slice(None)
            else:
                n, sl = int(idx.numel()), idx
                if n == 0:
                    return
            I_H = sample_latent(base.hip, n, mem_c[sl], mv_c[sl], mem_u[sl], mv_u[sl],
                                num_steps=args.num_steps, cfg_scale=args.cfg_scale, generator=gen,
                                device=dev)
            hc = intent_hidden(base.hip, I_H, s_read, gen, mem=mem_c[sl], mem_valid=mv_c[sl])
            hu = intent_hidden(base.hip, I_H, s_read, gen, mem=mem_u[sl], mem_valid=mv_u[sl])
        if idx is None:
            holi = [hc.clone(), hu.clone()]
        else:
            holi[0][idx] = hc.clone()
            holi[1][idx] = hu.clone()

    @torch.inference_mode()
    def base_action():
        """One planning pass of the frozen policy -> the first future action row, in RAW units."""
        x_obs = torch.zeros(B, H + F, TOKEN_DIM, device=dev)
        x_obs[:, :H] = hist
        obs_m = observed_mask(B, H, H + F, dev)
        g = generated_elements(obs_m, None, act_mask)
        if n_fut_prop > 0:
            x_obs[:, H, :PROPRIO_DIM] = (read_prop() - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM]
        _, mu_hist, _ = vae.encode(hist[:, :, :PROPRIO_DIM])
        lat_hist = (mu_hist - lat_mean) / lat_std
        scal = torch.stack([(ep_step.float() / n_env.float()).clamp(max=1.0), dur_s / 10.0], -1)
        hH_c, hH_u = holi
        I_I = sample_latent(base.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                            cfg_scale=args.cfg_scale, generator=gen, prefix=lat_hist, scalars=scal,
                            extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = intent_hidden(base.iip, I_I, s_read, gen, mem=mem_c, mem_valid=mv_c,
                             prefix_latent=lat_hist, scalars=scal, mem_extra=hH_c)
        toks, _ = base.intent_tokens(hH_c, hI_c, torch.ones(B, dtype=torch.bool, device=dev))
        x0 = sample_actions(base.policy, x_obs, obs_m, g, (text, pooled, tlen),
                            (text_u, pooled_u, tlen_u), scal, toks, num_steps=args.num_steps,
                            cfg_scale=args.cfg_action, generator=gen)
        a_n = x0[:, H, PROPRIO_DIM:]
        return a_n * std[PROPRIO_DIM:] + mean[PROPRIO_DIM:]

    # ---- residual policy --------------------------------------------------------------------------
    # The reference block is IN the input. Without it neither the actor nor the critic can observe what
    # the reward measures, the value function cannot explain the return's dominant per-clip variance,
    # and the residual could only ever learn a reference-agnostic stabiliser.
    RES_IN = PROPRIO_DIM + ACTION_DIM + REF_DIM + 2

    res = Residual(RES_IN, ACTION_DIM, args.residual_scale, args.init_log_std,
                   args.min_log_std).to(dev)
    opt = torch.optim.Adam(res.parameters(), lr=args.lr)
    it0 = 0
    if args.resume and (out / "latest.pt").exists():
        rck = torch.load(out / "latest.pt", map_location=dev)
        res.load_state_dict(rck["model"])
        if "opt" in rck:
            opt.load_state_dict(rck["opt"])
        it0 = int(rck.get("iter", 0))
        print(f"resumed from {out / 'latest.pt'} at iteration {it0}", flush=True)
    n_res = sum(p.numel() for p in res.parameters())
    print(f"residual: {n_res / 1e6:.2f} M params, input {RES_IN} "
          f"(proprio {PROPRIO_DIM} + base action {ACTION_DIM} + reference {REF_DIM} + 2), "
          f"scale {args.residual_scale} raw action units, initialised to a no-op", flush=True)

    def res_input(prop, a_base, obs_now):
        return torch.cat([(prop - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM],
                          (a_base - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:],
                          obs_now[:, REF_SLICE],
                          (ep_step.float() / n_env.float()).clamp(max=1.0)[:, None],
                          (dur_s / 10.0)[:, None]], dim=-1)

    def extra_reward(obs_now, dof_vel):
        """The optional hand-rolled shaping. All three weights default to 0; see the docstring."""
        ref = obs_now[:, REF_SLICE]
        d = ref[:, REF_DIFF].view(B, 3, 3).norm(dim=-1)
        r = torch.full((B,), args.w_alive, device=dev)
        if args.w_track:
            r = r + args.w_track * torch.exp(-args.track_k * d.mean(dim=-1))
        if args.w_rest:
            ref_speed = ref[:, REF_VEL].view(B, 3, 3).norm(dim=-1).mean(dim=-1)
            r = r - args.w_rest * (ref_speed < args.rest_vel_thresh).float() * dof_vel.abs().mean(-1)
        return r

    # ---- rollout and update -----------------------------------------------------------------------
    obs, _ = env.reset()
    hist[:, :, :PROPRIO_DIM] = ((read_prop() - mean[:PROPRIO_DIM]) / std[:PROPRIO_DIM])[:, None, :]
    hist[:, :, PROPRIO_DIM:] = ((hold_action() - mean[PROPRIO_DIM:]) / std[PROPRIO_DIM:])[:, None, :]
    sample_holistic()
    a_base = base_action().clone()
    # The history pair is (the state an action was applied IN, THAT action) -- the recorder's own
    # convention. The first version pushed the BASE action, which is not what reached the robot.
    prop_in = read_prop()
    applied_in = a_base.clone()
    T = args.rollout_steps
    hist_json, t0 = [], time.time()
    ep_ret = torch.zeros(B, device=dev)
    ep_len = torch.zeros(B, device=dev)
    done_ret, done_len, done_fell = [], [], []

    for it in range(it0 + 1, args.iters + 1):
        bx = torch.zeros(T, B, RES_IN, device=dev)
        bu = torch.zeros(T, B, ACTION_DIM, device=dev)
        blp = torch.zeros(T, B, device=dev)
        bv = torch.zeros(T, B, device=dev)
        br = torch.zeros(T, B, device=dev)
        bd = torch.zeros(T, B, device=dev)
        acc = torch.zeros(4, device=dev)       # env reward, point error, |residual|, n resets

        for s in range(T):
            prop_now = read_prop()
            x = res_input(prop_now, a_base, obs)
            with torch.no_grad():
                dist = res.dist(x)
                u = dist.sample()
                blp[s] = dist.log_prob(u).sum(-1)
                bv[s] = res.value(x)
            bx[s], bu[s] = x, u
            delta = res.squash(u)
            action = a_base + delta
            boundary = (s + 1) % hold == 0
            if s % hold == 0:
                prop_in, applied_in = prop_now, action.detach().clone()

            # `rew` is computed by the env at legged_robot.py:478, BEFORE reset_idx at :481, so it is
            # the reward of the state the robot actually reached -- not of a freshly reset episode.
            obs, _, rew, dones, _ = env.step(action.detach())
            ep_step += 1
            r = rew.detach() * args.rew_scale
            if args.w_track or args.w_alive or args.w_rest:
                r = r + extra_reward(obs, env.dof_vel)

            timeout = env.time_out_buf.clone().bool()       # the clip simply ran out
            fell = dones.bool() & ~timeout                  # contact, tilt, or lost the reference
            br[s] = r
            bd[s] = fell.float()     # only a real termination cuts the bootstrap; see the docstring
            ep_ret += r
            ep_len += 1
            acc[0] += r.mean()
            acc[1] += x[:, PROPRIO_DIM + ACTION_DIM:PROPRIO_DIM + ACTION_DIM + 9] \
                .view(B, 3, 3).norm(dim=-1).mean()
            acc[2] += delta.abs().mean()

            if boundary:
                push_hist(prop_in, applied_in)

            done = dones.bool()
            if done.any():
                idx = done.nonzero(as_tuple=False).flatten()
                acc[3] += idx.numel()
                done_ret += ep_ret[idx].tolist()
                done_len += ep_len[idx].tolist()
                done_fell += fell[idx].float().tolist()
                ep_ret[idx] = 0.0
                ep_len[idx] = 0.0
                ep_step[idx] = 0
                # The env has already reset these envs. Rebuild their history from the new state --
                # AFTER push_hist above, so the pre-reset pair cannot survive into the new episode --
                # and redraw only their holistic intent.
                p, ha = read_prop(), hold_action()
                hist[idx] = 0.0
                hist[idx, :, :PROPRIO_DIM] = ((p[idx] - mean[:PROPRIO_DIM])
                                              / std[:PROPRIO_DIM])[:, None, :]
                hist[idx, :, PROPRIO_DIM:] = ((ha[idx] - mean[PROPRIO_DIM:])
                                              / std[PROPRIO_DIM:])[:, None, :]
                # A reset lands mid-hold for most envs, and the plan in hand was made from the state
                # before the reset. Holding the new pose for the rest of the hold is in-distribution
                # and safe; the next boundary replans from the new state.
                a_base[idx] = ha[idx]
                prop_in[idx], applied_in[idx] = p[idx], ha[idx]
                sample_holistic(idx)

            if boundary:
                a_base = base_action().clone()

        with torch.no_grad():
            last_v = res.value(res_input(read_prop(), a_base, obs))
        adv = torch.zeros_like(br)
        gae = torch.zeros(B, device=dev)
        for s in reversed(range(T)):
            nv = last_v if s == T - 1 else bv[s + 1]
            nonterm = 1.0 - bd[s]
            delta_t = br[s] + args.gamma * nv * nonterm - bv[s]
            gae = delta_t + args.gamma * args.lam * nonterm * gae
            adv[s] = gae
        ret = adv + bv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        fx = bx.reshape(-1, RES_IN)
        fu = bu.reshape(-1, ACTION_DIM)
        flp = blp.reshape(-1)
        fadv = adv.reshape(-1)
        fret = ret.reshape(-1)
        n = fx.shape[0]
        mb = n // args.minibatches
        pl = vl = el = kl = 0.0
        k = 0
        stop = False
        for _ in range(args.epochs):
            if stop:
                break
            perm = torch.randperm(n, device=dev)
            for i in range(args.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                dist = res.dist(fx[j])
                lp = dist.log_prob(fu[j]).sum(-1)
                ratio = (lp - flp[j]).exp()
                p1 = ratio * fadv[j]
                p2 = ratio.clamp(1 - args.clip, 1 + args.clip) * fadv[j]
                loss_pi = -torch.min(p1, p2).mean()
                loss_v = (res.value(fx[j]) - fret[j]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = loss_pi + args.vf_coef * loss_v - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(res.parameters(), args.max_grad_norm)
                opt.step()
                mb_kl = float((flp[j] - lp).mean())
                pl += float(loss_pi); vl += float(loss_v); el += float(ent); kl += mb_kl
                k += 1
                if args.target_kl and mb_kl > args.target_kl:
                    stop = True
                    break
        with torch.no_grad():
            res.log_std.clamp_(min=args.min_log_std)

        if it % args.log_every == 0:
            a = (acc / T).tolist()
            m = dict(iter=it, env_steps=it * T * B,
                     env_rew=a[0], point_err_m=a[1], residual_abs=a[2], resets_per_step=a[3],
                     ep_ret=float(np.mean(done_ret[-256:])) if done_ret else float("nan"),
                     ep_len=float(np.mean(done_len[-256:])) if done_len else float("nan"),
                     fall_frac=float(np.mean(done_fell[-256:])) if done_fell else float("nan"),
                     n_episodes=len(done_ret),
                     loss_pi=pl / max(k, 1), loss_v=vl / max(k, 1), entropy=el / max(k, 1),
                     kl=kl / max(k, 1), updates=k, early_stop=stop,
                     log_std=float(res.log_std.mean()), minutes=(time.time() - t0) / 60)
            m["ep_ret_per_step"] = m["ep_ret"] / m["ep_len"] if m["ep_len"] else float("nan")
            hist_json.append(m)
            print(f"it {it} steps {m['env_steps']} rew {m['env_rew']:.3f} "
                  f"err {m['point_err_m']:.3f}m |d| {m['residual_abs']:.4f} "
                  f"ret {m['ep_ret']:.1f} r/s {m['ep_ret_per_step']:.3f} len {m['ep_len']:.0f} "
                  f"fall {m['fall_frac']:.3f} kl {m['kl']:.4f} ls {m['log_std']:.2f} "
                  f"{m['minutes']:.1f}min", flush=True)
            (out / "history.json").write_text(json.dumps(hist_json, indent=1))

        timed_out = args.max_hours and (time.time() - t0) / 3600 >= args.max_hours
        if it % args.save_every == 0 or it == args.iters or timed_out:
            torch.save(dict(model=res.state_dict(), opt=opt.state_dict(), args=vars(args), iter=it,
                            base_policy=str(policy_path), res_in=RES_IN), out / "latest.pt")
        if timed_out:
            print(f"stopping at iteration {it}: --max-hours {args.max_hours} reached", flush=True)
            break

    print(f"done, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
