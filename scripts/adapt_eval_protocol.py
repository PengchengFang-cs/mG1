"""ADAPT Table-1 protocol evaluation of a diffusion policy in the TextOp G1 env.
- N rollouts of 20 s (1000 steps @50Hz); prompt switched every 5-10 s, sampled from --prompt_file
- fall = any body except ankle-roll / wrist-yaw links closer than --contact_z to the ground (proxy for "illegal torso contact")
- metrics: success (no fall), action smoothness mean||a_t-a_{t-1}||^2 (Eq. S10), transition smoothness (same, 1 s after switches),
  foot sliding: mean horizontal speed of ankle-roll links while in contact (z < --contact_z) (Eq. S11 proxy)
Run inside the Isaac Lab container from TextOpTracker/.
"""
import argparse, sys, os, glob, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TextOp", "TextOpTracker", "scripts", "rsl_rl"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-ProjGravObs-MNMLP-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rollouts", type=int, default=2048)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--prompt_file", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/adapt_eval_prompts.txt")
parser.add_argument("--text_dict", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/text_embedding_dict_clip_merged.pkl")
parser.add_argument("--motion_glob", default="artifacts/val_all/*/motion.npz")
parser.add_argument("--ddim_steps", type=int, default=2); parser.add_argument("--guidance", type=float, default=2.5)
parser.add_argument("--solver", default="euler"); parser.add_argument("--contact_z", type=float, default=0.06)
parser.add_argument("--out", required=True)
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
    N = args_cli.num_envs
    env_cfg.scene.num_envs = N
    env_cfg.commands.motion.motion_files = sorted(glob.glob(args_cli.motion_glob))
    env_cfg.commands.motion.start_from_zero_step = True
    env_cfg.commands.motion.enable_adaptive_sampling = False
    env_cfg.commands.motion.pose_range = {}; env_cfg.commands.motion.velocity_range = {}; env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 600.0
    env_cfg.terminations.anchor_pos = None; env_cfg.terminations.anchor_ori = None; env_cfg.terminations.ee_body_pos = None
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    uenv = env.unwrapped; robot = uenv.scene["robot"]; cmd = uenv.command_manager.get_term("motion")
    body_names = list(robot.body_names)
    # ADAPT: illegal contact = any body except ankle-roll and wrist-yaw links. We use a height proxy, so also exclude the
    # ankle-pitch links (they sit ~3 cm above ground at rest) and all wrist links (hand assembly touches objects/ground in
    # manipulation prompts). Knees/hips/torso/head/elbows/shoulders remain "bad".
    bad = torch.tensor([i for i, n in enumerate(body_names) if not ("ankle" in n or "wrist" in n)], device=uenv.device)
    feet = torch.tensor([i for i, n in enumerate(body_names) if "ankle_roll" in n], device=uenv.device)
    print(f"[eval] bodies {len(body_names)} bad-contact bodies {len(bad)} feet {len(feet)}")
    prompts = [l.strip() for l in open(args_cli.prompt_file) if l.strip()]
    emb = joblib.load(args_cli.text_dict)
    missing = [p for p in prompts if p not in emb]
    if missing: print(f"[eval] WARNING {len(missing)} prompts missing from text dict, dropped: {missing[:10]}")
    prompts = [p for p in prompts if p in emb]
    emb_t = torch.stack([torch.as_tensor(np.asarray(emb[p], dtype=np.float32)) for p in prompts]).to(uenv.device)
    print(f"[eval] {len(prompts)} prompts, {args_cli.rollouts} rollouts x {args_cli.steps} steps, {N} envs, ddim {args_cli.ddim_steps} g {args_cli.guidance}")
    fps = 50
    pol = DiffusionPolicy(args_cli.ckpt, device=str(uenv.device), steps=args_cli.ddim_steps, guidance=args_cli.guidance, solver=args_cli.solver)
    tot = {"n": 0, "success": 0, "smooth_sum": 0.0, "smooth_n": 0, "trans_sum": 0.0, "trans_n": 0, "slide_sum": 0.0, "slide_n": 0, "fall_steps": []}
    n_batches = int(np.ceil(args_cli.rollouts / N)); t0 = time.time(); infer = []
    for b in range(n_batches):
        env.reset(); cmd.time_steps -= 1; cmd._update_command()
        pol.reset(N)
        cur = torch.randint(0, len(prompts), (N,), device=uenv.device)
        next_sw = (torch.rand(N, device=uenv.device) * 5 + 5) * fps
        fallen = torch.zeros(N, dtype=torch.bool, device=uenv.device); fall_step = torch.full((N,), -1, device=uenv.device)
        prev_a = None; trans_win = torch.zeros(N, device=uenv.device)
        sm_sum = torch.zeros(N, device=uenv.device); sm_n = torch.zeros(N, device=uenv.device)
        tr_sum = torch.zeros(N, device=uenv.device); tr_n = torch.zeros(N, device=uenv.device)
        sl_sum = torch.zeros(N, device=uenv.device); sl_n = torch.zeros(N, device=uenv.device)
        for step in range(args_cli.steps):
            sw = (step >= next_sw) & ~fallen
            if sw.any():
                cur[sw] = (cur[sw] + torch.randint(1, len(prompts), (int(sw.sum()),), device=uenv.device)) % len(prompts)
                next_sw[sw] = step + (torch.rand(int(sw.sum()), device=uenv.device) * 5 + 5) * fps
                trans_win[sw] = fps  # 1 s window
            d = robot.data
            obs = build_obs_torch(d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel, pol.prev_a)
            ts = time.time(); a = pol.act(obs, emb_t[cur]); torch.cuda.synchronize(); infer.append((time.time() - ts) * 1000)
            if prev_a is not None:
                da = ((a - prev_a) ** 2).sum(-1)
                alive = ~fallen
                sm_sum += da * alive; sm_n += alive
                inwin = alive & (trans_win > 0)
                tr_sum += da * inwin; tr_n += inwin
            prev_a = a.clone(); trans_win = (trans_win - 1).clamp_min(0)
            env.step(a)
            bp = robot.data.body_pos_w; bv = robot.data.body_lin_vel_w
            z = bp[:, :, 2] - uenv.scene.env_origins[:, None, 2]
            newly = (~fallen) & ((z[:, bad] < args_cli.contact_z).any(-1) | (robot.data.projected_gravity_b[:, 2] > -0.5))
            fall_step[newly] = step; fallen |= newly
            contact = (z[:, feet] < args_cli.contact_z) & (~fallen)[:, None]
            sl_sum += (bv[:, feet, :2].norm(dim=-1) * contact).sum(-1); sl_n += contact.sum(-1)
            if step % 250 == 0:
                print(f"[eval] batch {b+1}/{n_batches} step {step} fallen {int(fallen.sum())}/{N} infer {np.mean(infer[-250:]):.1f} ms")
        take = min(N, args_cli.rollouts - tot["n"])
        tot["n"] += take; tot["success"] += int((~fallen[:take]).sum())
        tot["smooth_sum"] += float(sm_sum[:take].sum()); tot["smooth_n"] += float(sm_n[:take].sum())
        tot["trans_sum"] += float(tr_sum[:take].sum()); tot["trans_n"] += float(tr_n[:take].sum())
        tot["slide_sum"] += float(sl_sum[:take].sum()); tot["slide_n"] += float(sl_n[:take].sum())
        tot["fall_steps"] += fall_step[:take][fallen[:take]].tolist()
        print(f"[eval] batch {b+1} done: success so far {tot['success']}/{tot['n']} ({time.time()-t0:.0f}s)")
    res = {"ckpt": args_cli.ckpt, "rollouts": tot["n"], "success_rate": tot["success"] / tot["n"],
           "action_smoothness": tot["smooth_sum"] / max(tot["smooth_n"], 1), "transition_smoothness": tot["trans_sum"] / max(tot["trans_n"], 1),
           "foot_sliding_mps": tot["slide_sum"] / max(tot["slide_n"], 1), "mean_fall_time_s": (np.mean(tot["fall_steps"]) / fps) if tot["fall_steps"] else None,
           "infer_ms": float(np.mean(infer)), "prompts": len(prompts), "ddim_steps": args_cli.ddim_steps, "guidance": args_cli.guidance, "contact_z": args_cli.contact_z}
    print("[eval] RESULT " + json.dumps(res))
    json.dump(res, open(args_cli.out, "w"), indent=1)
    env.close()

if __name__ == "__main__":
    main(); app.close()
