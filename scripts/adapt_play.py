"""Closed-loop rollout of the ADAPT diffusion policy in the TextOp tracking env (reference ignored).
Reports fall rate for a text prompt. Run inside the Isaac Lab container from TextOpTracker/."""
import argparse, sys, os, glob, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TextOp", "TextOpTracker", "scripts", "rsl_rl"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-ProjGravObs-MNMLP-v0")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--prompt", default="stand", help="comma-separated prompts; switched randomly every --switch_s seconds")
parser.add_argument("--switch_s", default="5,10", help="min,max seconds between prompt switches (0,0 = never)")
parser.add_argument("--save_traj", default=None, help="optional npz path to save root pos/quat + prompts per step for env 0..3")
parser.add_argument("--text_dict", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/UniPhys/data/babel_state-action-text-pairs/text_embedding_dict_clip.pkl")
parser.add_argument("--motion_glob", required=True, help="any motion npz (needed by the env; reference is ignored)")
parser.add_argument("--ddim_steps", type=int, default=2); parser.add_argument("--guidance", type=float, default=2.5)
parser.add_argument("--exec_steps", type=int, default=1, help="execute this many predicted future actions before replanning")
parser.add_argument("--stab_level", type=int, default=0, help="present history at this noise level (needs a model trained with --hist_noise_k)")
parser.add_argument("--solver", default="euler", choices=["euler", "heun"], help="flow models only")
parser.add_argument("--replay_pkl", default=None, help="diagnostic: open-loop replay of recorded actions of --replay_motion instead of the policy")
parser.add_argument("--replay_motion", default=None)
parser.add_argument("--zero_action", action="store_true", help="diagnostic: constant zero action (PD hold default pose)")
parser.add_argument("--tracker_resume", default=None, help="diagnostic: run the TextOp tracker policy (uses env obs) instead of the diffusion policy")
parser.add_argument("--keep_terminations", action="store_true", help="diagnostic: keep reference-based failure terminations")
parser.add_argument("--compare_pkl", default=None, help="diagnostic: compare model actions with recorded actions of --replay_motion for the first 30 steps")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch, numpy as np, joblib
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import textop_tracker.tasks  # noqa
from adapt.policy import DiffusionPolicy, build_obs_torch


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = sorted(glob.glob(args_cli.motion_glob))
    env_cfg.commands.motion.start_from_zero_step = True
    env_cfg.commands.motion.enable_adaptive_sampling = False
    env_cfg.commands.motion.pose_range = {}; env_cfg.commands.motion.velocity_range = {}; env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 600.0
    # reference-based failure terminations off; we judge falls ourselves
    if not args_cli.keep_terminations:
        env_cfg.terminations.anchor_pos = None; env_cfg.terminations.anchor_ori = None; env_cfg.terminations.ee_body_pos = None
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    tracker = None
    if args_cli.tracker_resume:
        from rsl_rl.runners import OnPolicyRunner
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device); runner.load(args_cli.tracker_resume)
        tracker = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped; robot = uenv.scene["robot"]; cmd = uenv.command_manager.get_term("motion")
    env.reset(); cmd.time_steps -= 1; cmd._update_command()
    env_obs, _ = env.get_observations()
    N = args_cli.num_envs
    replay = None
    if args_cli.replay_pkl:
        rr = [r for r in joblib.load(args_cli.replay_pkl)["rollouts"] if r["motion"] == args_cli.replay_motion]
        assert rr, "replay motion not found"; replay = torch.as_tensor(rr[0]["action"], device=uenv.device)
        print(f"[play] REPLAY mode: {args_cli.replay_motion} T={replay.shape[0]} success={rr[0]['success']}")
    pol = DiffusionPolicy(args_cli.ckpt, device=str(uenv.device), steps=args_cli.ddim_steps, guidance=args_cli.guidance, exec_steps=args_cli.exec_steps, stab_level=args_cli.stab_level, solver=args_cli.solver)
    pol.reset(N)
    emb = joblib.load(args_cli.text_dict)
    prompts = [q.strip() for q in args_cli.prompt.split(",") if q.strip()]
    for q in prompts: assert q in emb, f"prompt '{q}' not in text dict"
    emb_t = torch.stack([torch.as_tensor(np.asarray(emb[q], dtype=np.float32)) for q in prompts]).to(uenv.device)  # (P,512)
    smin, smax = [float(v) for v in args_cli.switch_s.split(",")]
    fps = 50
    cur = torch.randint(0, len(prompts), (N,), device=uenv.device)
    next_switch = (torch.rand(N, device=uenv.device) * (smax - smin) + smin) * fps if smax > 0 else torch.full((N,), 1e9, device=uenv.device)
    n_switch = torch.zeros(N, device=uenv.device)
    fallen = torch.zeros(N, dtype=torch.bool, device=uenv.device); fall_step = torch.full((N,), -1, device=uenv.device)
    traj = {"root": [], "prompt": []}
    t0 = time.time(); infer_ms = []
    for step in range(args_cli.steps):
        if smax > 0:
            sw = (step >= next_switch) & ~fallen
            if sw.any():
                cur[sw] = (cur[sw] + torch.randint(1, len(prompts), (int(sw.sum()),), device=uenv.device)) % len(prompts)
                next_switch[sw] = step + (torch.rand(int(sw.sum()), device=uenv.device) * (smax - smin) + smin) * fps
                n_switch[sw] += 1
        text = emb_t[cur]
        d = robot.data
        obs = build_obs_torch(d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel, pol.prev_a)
        ts = time.time()
        if replay is not None:
            a = replay[min(step, replay.shape[0] - 1)][None].expand(N, -1).clone(); pol.prev_a = a.clone()
        elif args_cli.zero_action:
            a = torch.zeros(N, 29, device=uenv.device)
        elif tracker is not None:
            a = tracker(env_obs)
        else:
            a = pol.act(obs, text)
            if args_cli.compare_pkl and step < 30:
                if step == 0:
                    rr = [r for r in joblib.load(args_cli.compare_pkl)["rollouts"] if r["motion"] == args_cli.replay_motion]; ref_a = torch.as_tensor(rr[0]["action"], device=uenv.device)
                ra = ref_a[step]
                print(f"[cmp] step {step:2d} |model-rec| mean {float((a[0]-ra).abs().mean()):.3f} max {float((a[0]-ra).abs().max()):.3f} | |rec| mean {float(ra.abs().mean()):.3f} | |model| mean {float(a[0].abs().mean()):.3f} | env-spread {float(a.std(0).mean()):.3f}")
            if step in (0, 1, 5, 50):
                hn = (pol.buf - pol.mean) / pol.std
                print(f"[dbg] step {step} |norm hist| max {float(hn.abs().max()):.2f} mean {float(hn.abs().mean()):.2f} | action |max| {float(a.abs().max()):.2f} mean|a| {float(a.abs().mean()):.2f} | obs q[:5] {obs[0,9:14].cpu().numpy().round(2)} g {obs[0,6:9].cpu().numpy().round(2)}")
        torch.cuda.synchronize(); infer_ms.append((time.time() - ts) * 1000)
        env_obs, _, _, _ = env.step(a)
        z = robot.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]; gz = robot.data.projected_gravity_b[:, 2]
        newly = (~fallen) & ((z < 0.45) | (gz > -0.6))
        fall_step[newly] = step; fallen |= newly
        if args_cli.save_traj:
            traj["root"].append(torch.cat([robot.data.root_pos_w[:4] - uenv.scene.env_origins[:4], robot.data.root_quat_w[:4], robot.data.joint_pos[:4]], -1).cpu().numpy()); traj["prompt"].append(cur[:4].cpu().numpy())
        if step % 100 == 0:
            print(f"[play] step {step} fallen {int(fallen.sum())}/{N} pelvis z mean {float(z.mean()):.2f} infer {np.mean(infer_ms[-100:]):.1f} ms")
    alive = ~fallen
    print(f"[play] prompts={prompts} steps={args_cli.steps} switch_s={args_cli.switch_s} fall_rate={float(fallen.float().mean()):.3f} success_rate={float(alive.float().mean()):.3f} mean_fall_step={float(fall_step[fallen].float().mean()) if fallen.any() else -1:.0f} switches/env={float(n_switch.mean()):.1f} infer_ms={np.mean(infer_ms):.1f} total {time.time()-t0:.0f}s")
    if args_cli.save_traj:
        np.savez(args_cli.save_traj, root=np.stack(traj["root"]), prompt=np.stack(traj["prompt"]), prompts=np.array(prompts))
    env.close()

if __name__ == "__main__":
    main(); app.close()
