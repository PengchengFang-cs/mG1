"""Roll out the pretrained TextOp tracker on reference motions and record (state, action, text) data
in the style of ADAPT / UniPhys: each rollout starts at frame 0 of a motion (no reset noise, no pushes),
runs until the motion ends (success) or a failure termination fires (failure).

Per step t we store, for the env that runs motion m:
  proprio[t]   (67,)  root_lin_vel_b(3) root_ang_vel_b(3) projected_gravity_b(3) joint_pos(29) joint_vel(29)   [clean, from sim]
  prev_action  (29,)  policy action applied at t-1 (zeros at t=0)
  action[t]    (29,)  raw policy output at t (normalized joint target; sim target = default + scale*action)
  joint_target (29,)  processed joint position target actually sent to the PD controller
  root_state   (13,)  world root pos(3) quat wxyz(4) lin_vel(3) ang_vel(3)
  ref_t        ()     reference frame index
Run inside the Isaac Lab container from the TextOpTracker directory.
"""
import argparse, sys, os, glob, time, json
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TextOp", "TextOpTracker", "scripts", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-ProjGravObs-MNMLP-v0")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--resume_path", type=str, required=True)
parser.add_argument("--motion_glob", type=str, required=True)
parser.add_argument("--meta_pkl", type=str, default=None, help="name->{feat_p, frame_ann,...} from make_motion_subset.py")
parser.add_argument("--out", type=str, required=True, help="output .pkl path")
parser.add_argument("--repeats", type=int, default=1, help="rollouts per motion")
parser.add_argument("--keep_pushes", action="store_true")
parser.add_argument("--clean_obs", type=int, default=1,
                    help="disable the tracker's observation corruption during recording; see the comment "
                         "where it is applied")
parser.add_argument("--no_dr", type=int, default=0,
                    help="disable every randomisation term, so the recorded dynamics are reproducible and a "
                         "recorded action sequence can be replayed exactly (scripts/adapt_replay_check.py)")
parser.add_argument("--dr_mode", default="startup", choices=["startup", "rollout"],
                    help="'startup' (the TextOp default) draws the randomisation ONCE per env at scene "
                         "creation, so a 6195-motion corpus collected with 256 envs contains only 256 "
                         "dynamics configurations and each env's draw is confounded with which motions it "
                         "happened to be handed. 'rollout' re-draws at every rollout boundary, which is what "
                         "'increase state coverage' (ADAPT Appendix A) needs.")
parser.add_argument("--actuator_dr", type=int, default=1,
                    help="add ADAPT Table S4's actuator-gain randomisation, which the TextOp EventCfg lacks")
parser.add_argument("--stiffness_range", type=float, nargs=2, default=[0.75, 1.25])
parser.add_argument("--damping_range", type=float, nargs=2, default=[0.75, 1.25])
parser.add_argument("--max_steps", type=int, default=200000)
parser.add_argument("--save_every", type=int, default=300, help="write a partial pkl every N finished rollouts (crash safety)")
# DAgger: execute the diffusion policy's action with probability --mix_prob per step, always record the tracker's
# action as the expert label (field expert_action); `action` is what was executed.
parser.add_argument("--policy_ckpt", default=None)
parser.add_argument("--mix_prob", type=float, default=0.0)
parser.add_argument("--policy_ddim", type=int, default=2); parser.add_argument("--policy_guidance", type=float, default=2.5); parser.add_argument("--policy_stab", type=int, default=0)
parser.add_argument("--text_dict", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/text_embedding_dict_clip_merged.pkl")
parser.add_argument("--policy_prompt_mode", default="ref", choices=["ref", "random"], help="random: the policy acts on a random prompt from --prompt_pool (resampled every 5-10 s) so states visited under prompt A get expert labels for the reference text B")
parser.add_argument("--prompt_pool", default="walk,stand,run,jog,sit,turn around,wave,jump,kick,walk backwards,stand still,tpose,dance,squat,step back")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch, numpy as np, joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
    names = [Path(f).parent.name for f in motion_files]
    env_cfg.commands.motion.motion_files = motion_files
    # deterministic, clean rollouts
    env_cfg.commands.motion.start_from_zero_step = True
    env_cfg.commands.motion.enable_adaptive_sampling = False
    env_cfg.commands.motion.random_static_prob = -1.0
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.episode_length_s = 600.0
    if args_cli.clean_obs:
        # `ProjGravObservationsCfg.PolicyCfg.__post_init__` sets enable_corruption=True (base_lin_vel +-0.5 m/s,
        # joint_vel +-0.5 rad/s, base_ang_vel +-0.2, projected_gravity +-0.07, motion_anchor_pos_b +-0.25).
        # With it on, the tracker acts on a NOISY observation while we store the CLEAN state, so every
        # behaviour-cloning label is a_t = pi(o_t + noise) recorded against o_t: irreducible label noise in
        # 100% of targets, which floors the denoising loss and smears the conditional action distribution.
        env_cfg.observations.policy.enable_corruption = False
        print("[rec] observation corruption OFF (labels match the stored state)")
    if not args_cli.keep_pushes:
        env_cfg.events.push_robot = None
    # base_com stays at startup: TextOp's randomize_rigid_body_com writes `coms[:, body_ids] += rand_samples`
    # with `coms` covering every env, so it only works when env_ids is all of them. Physically that is also
    # the most defensible split -- the torso centre of mass is a per-robot constant, while ground friction,
    # joint calibration drift and actuator gains are what should vary between episodes.
    if args_cli.no_dr:
        for _n in ("physics_material", "add_joint_default_pos", "base_com"):
            if getattr(env_cfg.events, _n, None) is not None:
                setattr(env_cfg.events, _n, None)
        args_cli.actuator_dr = 0
        print("[rec] ALL randomisation off (reproducible dynamics)")
    dr_terms = ["physics_material", "add_joint_default_pos"] + (["actuator_gains"] if args_cli.actuator_dr else [])
    if args_cli.dr_mode == "rollout":
        for _n in dr_terms:
            _t = getattr(env_cfg.events, _n, None)
            if _t is not None:
                _t.mode = "reset"
        print(f"[rec] domain randomisation re-drawn per rollout: {dr_terms}")
    if args_cli.actuator_dr:
        # ADAPT Table S4 randomises actuator dynamics during data collection; the TextOp EventCfg already
        # carries the other five quantities (friction, restitution, joint default offset, torso CoM) at the
        # same ranges, but has no actuator term at all. Added here so the collected data covers the
        # stiffness/damping variation the paper relies on -- it is the category that matters most for a
        # PD-controlled robot, where the policy emits joint targets and the gains decide the torque.
        from isaaclab.managers import EventTermCfg as _EventTerm, SceneEntityCfg
        from isaaclab.envs import mdp as _mdp
        env_cfg.events.actuator_gains = _EventTerm(
            func=_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": tuple(args_cli.stiffness_range),
                "damping_distribution_params": tuple(args_cli.damping_range),
                "operation": "scale",
            },
        )
        print(f"[rec] actuator DR on: stiffness x{tuple(args_cli.stiffness_range)} "
              f"damping x{tuple(args_cli.damping_range)}")
    meta = joblib.load(args_cli.meta_pkl) if args_cli.meta_pkl else {}

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("motion")
    tm = uenv.termination_manager
    robot = uenv.scene["robot"]
    act_term = uenv.action_manager.get_term("joint_pos")
    n_motion = cmd.motion.num_files
    lengths = cmd.motion.file_lengths.cpu()
    N = args_cli.num_envs

    # ---- motion queue: every motion `repeats` times, handed out round-robin by the patched sampler ----
    queue = [m for r in range(args_cli.repeats) for m in range(n_motion)]
    state = {"queue": queue, "idle": torch.zeros(N, dtype=torch.bool)}

    def queued_sampling(env_ids):
        out = torch.zeros(len(env_ids), dtype=torch.long, device=cmd.device)
        for j, e in enumerate(env_ids.tolist()):
            if state["queue"]:
                out[j] = state["queue"].pop(0)
            else:
                out[j] = 0
                state["idle"][e] = True
        return out
    cmd._uniform_sampling = queued_sampling  # type: ignore

    # initial assignment happened during env creation with the original sampler -> force a fresh reset now
    env.reset()  # calls _resample_command for all envs -> our queue
    # IsaacLab computes terminations BEFORE command_manager.compute() inside step(), so after an explicit
    # reset the command's relative body poses are stale and every env would be falsely terminated on step 1.
    # Refresh them without advancing the reference frame.
    cmd.time_steps -= 1
    cmd._update_command()
    obs, _ = env.get_observations()
    print(f"[rec] {n_motion} motions x {args_cli.repeats} repeats, {N} envs, joint order: {robot.joint_names}")

    # ---- optional DAgger policy ----
    dpol, emb_dict, cur_text = None, None, None
    if args_cli.policy_ckpt:
        # ADAPT's DAgger policy was deleted on 2026-10-01 (CLAUDE.md §5, STATUS.md §4); recording
        # tracker rollouts without --policy_ckpt is unaffected.
        raise SystemExit(
            "--policy_ckpt needed ADAPT's DiffusionPolicy, deleted 2026-10-01 "
            "(CLAUDE.md §5, STATUS.md §4). Record tracker rollouts without --policy_ckpt."
        )
        from hml_phys.babel_labels import labels_overlapping
        dpol = DiffusionPolicy(args_cli.policy_ckpt, device=str(uenv.device), steps=args_cli.policy_ddim, guidance=args_cli.policy_guidance, stab_level=args_cli.policy_stab)
        dpol.reset(N)
        emb_dict = joblib.load(args_cli.text_dict)
        zero_emb = torch.zeros(512, device=uenv.device)
        def text_for(i):
            m = names[int(cmd.motion_idx[i])]; ann = meta.get(m, {}).get("frame_ann", []); t = float(cmd.time_steps[i]) / 50.0
            labs = labels_overlapping(ann, t, t + 0.4)
            return torch.as_tensor(np.asarray(emb_dict[labs[0]], dtype=np.float32), device=uenv.device) if labs and labs[0] in emb_dict else zero_emb
        pool = [q.strip() for q in args_cli.prompt_pool.split(",") if q.strip() in emb_dict]
        pool_emb = torch.stack([torch.as_tensor(np.asarray(emb_dict[q], dtype=np.float32), device=uenv.device) for q in pool])
        rnd_prompt = torch.randint(0, len(pool), (N,), device=uenv.device)
        rnd_next = torch.randint(250, 500, (N,), device=uenv.device)
        print(f"[rec] DAgger mode: policy {args_cli.policy_ckpt} mix_prob {args_cli.mix_prob} prompt_mode {args_cli.policy_prompt_mode} pool {len(pool)}")
    rollouts, active = [], [None] * N
    def start(i):
        m = int(cmd.motion_idx[i]); active[i] = {"motion": m, "buf": {k: [] for k in ("proprio","prev_action","action","expert_action","joint_target","root_state","ref_t","by_policy")}}
        if dpol is not None: dpol.reset(N, torch.tensor([i], device=uenv.device))
    def finish(i, success):
        r = active[i]; b = r["buf"]
        if len(b["action"]) == 0:
            active[i] = None; return
        m = r["motion"]; nm = names[m]
        rec = {k: (np.asarray(v, dtype=np.int32) if k == "ref_t" else np.asarray(v, dtype=bool) if k == "by_policy" else np.stack(v).astype(np.float32)) for k, v in b.items()}
        rec.update(motion=nm, motion_len=int(lengths[m]), success=bool(success), feat_p=meta.get(nm, {}).get("feat_p", nm),
                   frame_ann=meta.get(nm, {}).get("frame_ann", []), fps=50)
        rollouts.append(rec); active[i] = None
        print(f"[rec] {'OK ' if success else 'FAIL'} {nm[:55]:55s} T={rec['action'].shape[0]:5d}/{int(lengths[m])}  done={len(rollouts)}")
        if args_cli.save_every > 0 and len(rollouts) % args_cli.save_every == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            joblib.dump({"rollouts": rollouts, "summary": {"partial": True, "n_rollouts": len(rollouts), "joint_names": list(robot.joint_names)}}, args_cli.out + ".partial")
            print(f"[rec] partial save {len(rollouts)} rollouts -> {args_cli.out}.partial")
    for i in range(N): start(i)

    prev_action = torch.zeros(N, robot.num_joints, device=uenv.device)
    boundary = torch.zeros(N, dtype=torch.bool, device=uenv.device)
    if args_cli.dr_mode == "rollout":
        uenv.event_manager.apply(mode="reset", env_ids=torch.arange(N, device=uenv.device),
                                 global_env_step_count=0)   # first draw
    t0 = time.time(); step = 0
    while step < args_cli.max_steps and not bool(state["idle"].all()):
        with torch.inference_mode():
            expert = policy(obs)
            actions = expert
            by_pol = torch.zeros(N, dtype=torch.bool, device=uenv.device)
            if dpol is not None:
                d = robot.data
                o96 = build_obs_torch(d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel, prev_action)
                if args_cli.policy_prompt_mode == "random":
                    sw = rnd_next <= 0
                    rnd_prompt[sw] = torch.randint(0, len(pool), (int(sw.sum()),), device=uenv.device); rnd_next[sw] = torch.randint(250, 500, (int(sw.sum()),), device=uenv.device)
                    rnd_next -= 1
                    text = pool_emb[rnd_prompt]
                else:
                    text = torch.stack([text_for(i) for i in range(N)])
                a_pol = dpol.act(o96, text)
                by_pol = torch.rand(N, device=uenv.device) < args_cli.mix_prob
                actions = torch.where(by_pol[:, None], a_pol, expert)
                dpol.prev_a = actions.clone()   # the policy must see the executed action as its previous action
            # snapshot state at time t (before stepping)
            d = robot.data
            proprio = torch.cat([d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b, d.joint_pos, d.joint_vel], dim=-1)
            root_state = torch.cat([d.root_pos_w - uenv.scene.env_origins, d.root_quat_w, d.root_lin_vel_w, d.root_ang_vel_w], dim=-1)
            ref_t = cmd.time_steps.clone(); mlen = cmd.motion_length.clone()
            obs, rew, dones, extras = env.step(actions)
            tgt = act_term.processed_actions.clone()
        terminated = tm.terminated.cpu(); ref_t_c = ref_t.cpu()
        for i in range(N):
            if active[i] is None: continue
            b = active[i]["buf"]
            b["proprio"].append(proprio[i].cpu().numpy()); b["prev_action"].append(prev_action[i].cpu().numpy())
            b["action"].append(actions[i].cpu().numpy()); b["expert_action"].append(expert[i].cpu().numpy()); b["by_policy"].append(bool(by_pol[i]))
            b["joint_target"].append(tgt[i].cpu().numpy())
            b["root_state"].append(root_state[i].cpu().numpy()); b["ref_t"].append(int(ref_t_c[i]))
            # completion wins over termination: a rollout that executes its last reference frame AND trips a
            # termination on the same step did finish the motion, and scoring it as a failure silently threw
            # those motions out of an only_success dataset
            if ref_t_c[i] >= int(mlen[i]) - 1:     # last reference frame executed -> motion complete
                finish(i, True)
            elif terminated[i]:
                if len(b["action"]) <= 2:
                    fired = [n for n in tm.active_terms if bool(tm.get_term(n)[i])]
                    print(f"[rec] early termination env {i} T={len(b['action'])} ref_t={int(ref_t_c[i])} terms={fired}")
                finish(i, False)
            if active[i] is None and not state["idle"][i]:
                start(i); boundary[i] = True
        if args_cli.dr_mode == "rollout" and bool(boundary.any()):
            # a completed motion resamples the reference INSIDE command_manager.compute() with no env reset,
            # so reset-mode events would never fire at that boundary; apply them by hand for those envs
            uenv.event_manager.apply(mode="reset", env_ids=torch.where(boundary)[0],
                                     global_env_step_count=int(uenv.common_step_counter))
        prev_action = actions.clone()
        # Clear prev_action at EVERY rollout boundary, not just on failure. On motion completion the command
        # term resamples and teleports the robot inside command_manager.compute() with no env reset, so the
        # old action survived into the new rollout: measured 80% of recorded rollouts began with a non-zero
        # prev_action (median |max| 2.24) paired with a freshly teleported pose -- a state the closed-loop
        # policy, which resets its buffer to zero, never reproduces.
        prev_action[terminated.to(uenv.device)] = 0.0
        prev_action[boundary] = 0.0
        boundary[:] = False
        step += 1
        if step % 500 == 0:
            print(f"[rec] step {step}  rollouts {len(rollouts)}  queue {len(state['queue'])}  {time.time()-t0:.0f}s")
    for i in range(N):
        if active[i] is not None and not state["idle"][i]: finish(i, False)

    n_ok = sum(r["success"] for r in rollouts)
    summary = {"n_rollouts": len(rollouts), "n_success": n_ok, "success_rate": n_ok / max(len(rollouts), 1),
               "frames_total": int(sum(r["action"].shape[0] for r in rollouts)),
               "frames_success": int(sum(r["action"].shape[0] for r in rollouts if r["success"])),
               "joint_names": list(robot.joint_names), "action_scale": (act_term._scale[0] if act_term._scale.dim() == 2 else act_term._scale).cpu().numpy().tolist() if torch.is_tensor(act_term._scale) else act_term._scale,
               "default_joint_pos": robot.data.default_joint_pos[0].cpu().numpy().tolist(),
               "proprio_layout": "root_lin_vel_b(3) root_ang_vel_b(3) projected_gravity_b(3) joint_pos(29) joint_vel(29)"}
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    joblib.dump({"rollouts": rollouts, "summary": summary}, args_cli.out)
    with open(args_cli.out + ".summary.json", "w") as f: json.dump(summary, f, indent=1)
    print(f"[rec] saved {len(rollouts)} rollouts ({n_ok} success, {summary['frames_success']} success frames) -> {args_cli.out}  in {time.time()-t0:.0f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
