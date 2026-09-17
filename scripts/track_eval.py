"""Evaluate the pretrained TextOp tracker on a set of motions for a fixed number of steps.
Derived from TextOpTracker/scripts/rsl_rl/play.py. Prints tracking metrics and termination counts.
Run inside the Isaac Lab container from the TextOpTracker directory.
"""
import argparse, sys, os, glob, time
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TextOp", "TextOpTracker", "scripts", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-ProjGravObs-MNMLP-v0")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--resume_path", type=str, required=True)
parser.add_argument("--motion_glob", type=str, required=True, help="glob of motion.npz files")
parser.add_argument("--no_randomize", action="store_true", help="zero the reset pose/velocity/joint noise ranges")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections import defaultdict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import textop_tracker.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    motion_files = sorted(glob.glob(args_cli.motion_glob))
    assert motion_files, f"no motions match {args_cli.motion_glob}"
    env_cfg.commands.motion.motion_files = motion_files
    if args_cli.no_randomize:
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    print(f"[eval] {len(motion_files)} motions, {args_cli.num_envs} envs, {args_cli.steps} steps")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    print("[eval] obs shape", tuple(obs.shape), " action dim", env.num_actions)
    cmd = env.unwrapped.command_manager.get_term("motion")
    tm = env.unwrapped.termination_manager
    n_motion = cmd.motion.num_files
    lengths = cmd.motion.file_lengths
    names = [Path(f).parent.name for f in motion_files]
    succ = torch.zeros(n_motion); fail = torch.zeros(n_motion)
    err_sum = torch.zeros(n_motion); err_cnt = torch.zeros(n_motion)
    prev_idx = cmd.motion_idx.clone(); prev_t = cmd.time_steps.clone()
    first_err = []
    t0 = time.time()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
        idx = cmd.motion_idx.clone(); t = cmd.time_steps.clone()
        e = cmd.metrics["error_joint_pos"].detach().cpu()
        err_sum.index_add_(0, prev_idx.cpu(), e); err_cnt.index_add_(0, prev_idx.cpu(), torch.ones_like(e))
        terminated = tm.terminated.cpu()
        # a run ends either by termination (failure) or by reaching the end of its motion (success)
        reached_end = (prev_t.cpu() >= (lengths[prev_idx].cpu() - 2)) & ~terminated
        for i in range(args_cli.num_envs):
            if terminated[i]: fail[prev_idx[i]] += 1
            elif reached_end[i] and t[i] < prev_t[i]: succ[prev_idx[i]] += 1
        if step < 5: first_err.append(round(float(e.mean()), 3))
        prev_idx, prev_t = idx, t
    dt = time.time() - t0
    print(f"[eval] done {args_cli.steps} steps in {dt:.1f}s  ({args_cli.steps*args_cli.num_envs/dt:.0f} env-steps/s)")
    print("[eval] mean joint err first 5 steps:", first_err)
    print(f"[eval] {'motion':60s} {'len':>5s} {'succ':>4s} {'fail':>4s} {'jerr':>6s}")
    for m in range(n_motion):
        je = float(err_sum[m] / max(err_cnt[m], 1))
        print(f"  {names[m][:60]:60s} {int(lengths[m]):5d} {int(succ[m]):4d} {int(fail[m]):4d} {je:6.3f}")
    print(f"[eval] TOTAL succ={int(succ.sum())} fail={int(fail.sum())}  success_rate={float(succ.sum()/max(succ.sum()+fail.sum(),1)):.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
