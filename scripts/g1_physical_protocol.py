"""ADAPT Table-1 protocol evaluation of a diffusion policy in the TextOp G1 env (docs/05 §5).

Faithful to the paper's definitions, not proxies:
- 2,048 rollouts of 20 s (1000 steps @50 Hz), prompt switched every 5-10 s, drawn from a pool of 130 commands
  covering locomotion / exercises / upper-body gestures (§4.1)
- **fall = illegal torso contact** (Appendix C), read from the scene's ContactSensor -- not a height proxy.
  The height proxy scored the tracker 0.761 where the contact criterion scores it 0.629; they are not the
  same measurement.
- action smoothness = mean ||a_t - a_{t-1}||^2 (Eq. S10); transition smoothness = the same inside a 1 s window
  after each prompt switch
- foot sliding = Eq. S11: horizontal speed of the ankle-roll links summed over both feet, gated by
  **peak contact force > 1 N** over the sensor's history window
- R-Precision is NOT computed here: the paper uses a trained TMR retrieval model over all N candidates
  (Eqs. S12-S13), which is a separate scoring pass over the recorded motions.

The prompt pool and the reference-motion whitelist both come from `scripts/hml_phys/g1_prompt_pool.py`.
The whitelist matters: BABEL also labels sitting, kneeling and crawling, where torso contact is the correct
behaviour, so a contact-based fall criterion is only meaningful once those motions are out.

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
parser.add_argument("--source", default="policy", choices=["policy", "tracker"],
                    help="'policy' drives our diffusion policy from text. 'tracker' runs the pretrained "
                         "TextOp tracker on a pre-generated reference, which is how a two-stage baseline "
                         "(generate a kinematic motion offline, then track it) is evaluated on this protocol "
                         "-- FRoM-W1 and ADAPT's own 'Offline TextOp' / DART rows are of that kind.")
parser.add_argument("--ckpt", default="", help="our policy checkpoint; required for --source policy")
parser.add_argument("--resume_path", default="logs/rsl_rl/Pretrained/checkpoints/model_75000.pt",
                    help="the pretrained TextOp tracker, for --source tracker")
parser.add_argument("--prompt_file", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/g1_prompt_pool_130.txt")
parser.add_argument("--motion_list", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/g1_motion_whitelist_val_all.txt",
                    help="pass 'none' to skip; only the val_all reference set needs screening")
parser.add_argument("--fall_mode", default="contact", choices=["contact", "height"])
parser.add_argument("--fall_bodies", default="pelvis,torso_link",
                    help="'illegal torso contact' (Appendix C). waist_yaw_link and waist_roll_link are rigid "
                         "bodies with NO collision geometry in the G1 URDF, so they can never report a force; "
                         "listing them only overstates the criterion's coverage. A collapse onto knees and "
                         "elbows is therefore not a fall under the paper's literal definition.")
parser.add_argument("--fall_force", type=float, default=1.0)
parser.add_argument("--record_body_pos", default="", help="also dump per-episode link positions here, for TMR scoring")
parser.add_argument("--text_dict", default="/iridisfs/scratch/pf2m24/projects/motion_rebot/data/text_embedding_dict_clip_merged.pkl")
parser.add_argument("--motion_set", default="val_all",
                    help="artifacts subdirectory; the wildcard is built inside the script because a glob on "
                         "the command line is expanded by the shell into several hydra overrides and fails "
                         "with \"mismatched input 'motion.npz' expecting ID\"")
parser.add_argument("--ddim_steps", type=int, default=2); parser.add_argument("--guidance", type=float, default=2.5)
parser.add_argument("--solver", default="euler"); parser.add_argument("--contact_z", type=float, default=0.06)
parser.add_argument("--init_pose", default="reference", choices=["reference", "default"],
                    help="'reference' starts each episode at frame 0 of whatever reference motion the env "
                         "drew, which can be mid-stride or crouched and carries that frame's velocities. "
                         "'default' starts from the robot's neutral stance at rest. The paper does not say "
                         "which it uses, and our policy's mean fall time (3.5 s) is shorter than the first "
                         "prompt segment, so the start matters.")
parser.add_argument("--fixed_noise", type=int, default=0,
                    help="hold the sampler's initial Gaussian fixed across control steps (diagnostic)")
parser.add_argument("--seed", type=int, default=0)
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
# ADAPT's DiffusionPolicy / build_obs_torch were deleted on 2026-10-01 with the rest of the ADAPT
# reproduction (CLAUDE.md §5, STATUS.md §4). Only the tracker path of this script survives; our own
# end-to-end G1 policy is evaluated by scripts/g1_eval_rollout.py, which uses hml_phys/g1_model.py.
def _adapt_policy_deleted(*_a, **_k):
    raise SystemExit(
        "this path needed ADAPT's DiffusionPolicy / build_obs_torch, deleted 2026-10-01 "
        "(CLAUDE.md §5, STATUS.md §4). Use --source tracker here; for our own end-to-end G1 "
        "policy use scripts/g1_eval_rollout.py."
    )


DiffusionPolicy = build_obs_torch = _adapt_policy_deleted


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    torch.manual_seed(args_cli.seed); np.random.seed(args_cli.seed)
    assert args_cli.source != "policy" or args_cli.ckpt, "--source policy needs --ckpt"
    N = args_cli.num_envs
    env_cfg.scene.num_envs = N
    motion_files = sorted(glob.glob(f"artifacts/{args_cli.motion_set}/*/motion.npz"))
    assert motion_files, f"no motions under artifacts/{args_cli.motion_set}"
    if args_cli.motion_list and args_cli.motion_list.lower() not in ("", "none"):
        keep = {l.strip() for l in open(args_cli.motion_list) if l.strip()}
        n0 = len(motion_files)
        motion_files = [f for f in motion_files if os.path.basename(os.path.dirname(f)) in keep]
        print(f"[eval] motion whitelist: {len(motion_files)}/{n0}")
        assert motion_files, "the whitelist matched no motion"
    env_cfg.commands.motion.motion_files = motion_files
    env_cfg.commands.motion.start_from_zero_step = True
    env_cfg.commands.motion.enable_adaptive_sampling = False
    env_cfg.commands.motion.pose_range = {}; env_cfg.commands.motion.velocity_range = {}; env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 600.0
    env_cfg.terminations.anchor_pos = None; env_cfg.terminations.anchor_ori = None; env_cfg.terminations.ee_body_pos = None
    # The scene's ContactSensor reports NET force per body, so with self-collisions enabled an arm resting on
    # the torso is indistinguishable from the ground. Filtering against the ground prim gives force_matrix_w,
    # which is ground contact only -- the quantity "illegal torso contact" actually means.
    if args_cli.fall_mode == "contact":
        # the collider is the Plane prim, not the Xform: filtering against "/World/ground" matches a
        # prim with no CollisionAPI, so force_matrix_w is allocated but stays identically zero --
        # which silently disables the fall criterion AND foot-sliding detection (measured: a
        # 300-step model scored success 1.0 and foot sliding 0.0)
        env_cfg.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/GroundPlane/CollisionPlane"]
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    uenv = env.unwrapped; robot = uenv.scene["robot"]; cmd = uenv.command_manager.get_term("motion")

    # ---- stop the motion command from teleporting the robot when its reference clip runs out.
    # `MotionCommandMulti._update_command` increments `time_steps` every step and calls `_resample_command`
    # on any env past the end of its clip, which writes a fresh joint and root state straight into the sim.
    # Our reference clips have a median length of 6.7 s against a 20 s episode, so ~79% of rollouts were
    # being teleported at least once, mid-flight; `episode_length_s = 600` suppresses the episode reset, not
    # this one. Every physical metric was measuring those discontinuities. The policy observes nothing from
    # the command term, and the reference buffers are read through a clamped index (commands_multi.py:371+),
    # so holding the reference at its last frame is safe and changes nothing the policy can see.
    _frozen = {"on": False}
    _orig_resample, _orig_update = cmd._resample_command, cmd._update_command

    def _resample(env_ids):
        if _frozen["on"]:
            return                      # never re-place the robot during a rollout
        _orig_resample(env_ids)

    def _update():
        if _frozen["on"]:
            cmd.time_steps.clamp_(max=cmd.motion_length - 2)     # so the +1 below lands on the last frame
        _orig_update()
        if _frozen["on"]:
            cmd.time_steps.clamp_(max=cmd.motion_length - 1)

    cmd._resample_command, cmd._update_command = _resample, _update
    body_names = list(robot.body_names)
    # ADAPT: illegal contact = any body except ankle-roll and wrist-yaw links. We use a height proxy, so also exclude the
    # ankle-pitch links (they sit ~3 cm above ground at rest) and all wrist links (hand assembly touches objects/ground in
    # manipulation prompts). Knees/hips/torso/head/elbows/shoulders remain "bad".
    bad = torch.tensor([i for i, n in enumerate(body_names) if not ("ankle" in n or "wrist" in n)], device=uenv.device)
    feet = torch.tensor([i for i, n in enumerate(body_names) if "ankle_roll" in n], device=uenv.device)
    csensor = uenv.scene.sensors["contact_forces"] if "contact_forces" in uenv.scene.sensors else None
    fall_names = [n.strip() for n in args_cli.fall_bodies.split(",") if n.strip()]
    if args_cli.fall_mode == "contact":
        assert csensor is not None, "no 'contact_forces' sensor in the scene; use --fall_mode height"
        sensor_names = list(csensor.body_names)      # unconditional: a silent fallback here mispairs indices
        missing = [n for n in fall_names if n not in sensor_names]
        assert not missing, f"unknown bodies in --fall_bodies: {missing}"
        s_fall = torch.tensor([sensor_names.index(n) for n in fall_names], device=uenv.device)
        # same order as `feet`, or one foot's contact flag would gate the other foot's velocity
        s_feet = torch.tensor([sensor_names.index(body_names[i]) for i in feet.tolist()], device=uenv.device)
    else:
        s_fall = s_feet = None

    def contact_peak(ids):
        """peak |ground contact force| over the sensor's history window, per body.

        `force_matrix_w` [N, bodies, filters, 3] exists because we filtered against the ground prim; it
        excludes self-collision, which `net_forces_w` does not."""
        fm = getattr(csensor.data, "force_matrix_w", None)
        assert fm is not None, "contact filtering did not take effect; force_matrix_w is absent"
        return fm[:, ids].sum(dim=2).norm(dim=-1)

    print(f"[eval] {len(body_names)} bodies; fall={args_cli.fall_mode} "
          f"({fall_names if args_cli.fall_mode == 'contact' else f'{len(bad)} bodies below {args_cli.contact_z} m'})")
    prompts = [l.strip() for l in open(args_cli.prompt_file) if l.strip()]
    emb = joblib.load(args_cli.text_dict)
    missing = [p for p in prompts if p not in emb]
    if missing: print(f"[eval] WARNING {len(missing)} prompts missing from text dict, dropped: {missing[:10]}")
    prompts = [p for p in prompts if p in emb]
    emb_t = torch.stack([torch.as_tensor(np.asarray(emb[p], dtype=np.float32)) for p in prompts]).to(uenv.device)
    print(f"[eval] {len(prompts)} prompts, {args_cli.rollouts} rollouts x {args_cli.steps} steps, {N} envs, ddim {args_cli.ddim_steps} g {args_cli.guidance}")
    fps = 50
    pol = tracker = None
    if args_cli.source == "policy":
        pol = DiffusionPolicy(args_cli.ckpt, device=str(uenv.device), steps=args_cli.ddim_steps,
                              guidance=args_cli.guidance, solver=args_cli.solver)
        pol.fixed_noise = bool(args_cli.fixed_noise)
    else:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.resume_path)
        tracker = runner.get_inference_policy(device=uenv.device)
        # The reference must ADVANCE for a tracker -- freezing it, which is right for a policy that ignores
        # the command term, would make it hold the first frame forever. The schedules are built to exactly
        # --steps frames, so the clip never runs out and no resample (i.e. no teleport) can fire either way.
        short = int((cmd.motion_length < args_cli.steps).sum())
        assert short == 0, (f"{short} reference motions are shorter than {args_cli.steps} steps; they would "
                            f"be resampled mid-rollout, which teleports the robot")
        print(f"[eval] tracker {args_cli.resume_path}; reference advances, "
              f"min motion length {int(cmd.motion_length.min())} frames")
    tot = {"n": 0, "success": 0, "sm": 0.0, "sm_k": 0, "tr": 0.0, "tr_k": 0, "sl": 0.0, "sl_k": 0, "fall_steps": []}
    if args_cli.fall_mode == "contact":
        # a filter that matches no collider yields an all-zero force matrix, i.e. "nothing ever touches the
        # ground" -- silent and catastrophic. Step a few times and insist the feet report something.
        for _ in range(8):
            env.step(torch.zeros(N, robot.num_joints, device=uenv.device))
        assert float(contact_peak(s_feet).max()) > 1.0, (
            "the ground-filtered contact forces are all zero after 8 steps; check "
            "env_cfg.scene.contact_forces.filter_prim_paths_expr against the actual collider prim")
        print(f"[eval] ground contact check OK (peak foot force {float(contact_peak(s_feet).max()):.0f} N)")
    n_batches = int(np.ceil(args_cli.rollouts / N)); t0 = time.time(); infer = []; episodes = []
    for b in range(n_batches):
        _frozen["on"] = False                 # the reset must still hand out fresh reference motions
        env.reset(); cmd.time_steps -= 1; cmd._update_command()
        _frozen["on"] = args_cli.source == "policy"   # a tracker needs the reference to keep advancing
        if args_cli.init_pose == "default":
            ids = torch.arange(N, device=uenv.device)
            root = robot.data.default_root_state.clone()
            root[:, :3] += uenv.scene.env_origins
            root[:, 7:] = 0.0                                   # zero linear and angular velocity
            robot.write_root_state_to_sim(root, env_ids=ids)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                           torch.zeros_like(robot.data.default_joint_vel), env_ids=ids)
            uenv.sim.step(render=False); uenv.scene.update(uenv.physics_dt)
        if pol is not None:
            pol.reset(N)
        obs_t, _ = env.get_observations()
        cur = torch.randint(0, len(prompts), (N,), device=uenv.device)
        next_sw = (torch.rand(N, device=uenv.device) * 5 + 5) * fps
        fallen = torch.zeros(N, dtype=torch.bool, device=uenv.device); fall_step = torch.full((N,), -1, device=uenv.device)
        prev_a = None; trans_win = torch.zeros(N, device=uenv.device)
        rec_bp = [] if args_cli.record_body_pos else None
        seg_start = torch.zeros(N, dtype=torch.long, device=uenv.device); segments = [[] for _ in range(N)]
        sm_sum = torch.zeros(N, device=uenv.device); sm_n = torch.zeros(N, device=uenv.device)
        tr_sum = torch.zeros(N, device=uenv.device); tr_n = torch.zeros(N, device=uenv.device)
        sl_sum = torch.zeros(N, device=uenv.device); sl_n = torch.zeros(N, device=uenv.device)
        for step in range(args_cli.steps):
            sw = (step >= next_sw) & ~fallen & (len(prompts) > 1)   # a one-command pool never switches
            if sw.any():
                for i in torch.where(sw)[0].tolist():
                    segments[i].append((int(seg_start[i]), step, prompts[int(cur[i])]))
                seg_start[sw] = step
                cur[sw] = (cur[sw] + torch.randint(1, len(prompts), (int(sw.sum()),), device=uenv.device)) % len(prompts)
                next_sw[sw] = step + (torch.rand(int(sw.sum()), device=uenv.device) * 5 + 5) * fps
                trans_win[sw] = fps  # 1 s window
            d = robot.data
            ts = time.time()
            if tracker is not None:
                a = tracker(obs_t)
            else:
                obs = build_obs_torch(d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b,
                                      d.joint_pos, d.joint_vel, pol.prev_a)
                a = pol.act(obs, emb_t[cur])
            torch.cuda.synchronize(); infer.append((time.time() - ts) * 1000)
            if prev_a is not None:
                da = ((a - prev_a) ** 2).sum(-1)
                alive = ~fallen
                sm_sum += da * alive; sm_n += alive
                inwin = alive & (trans_win > 0)
                tr_sum += da * inwin; tr_n += inwin
            prev_a = a.clone(); trans_win = (trans_win - 1).clamp_min(0)
            obs_t, _, _, _ = env.step(a)
            bp = robot.data.body_pos_w; bv = robot.data.body_lin_vel_w
            if rec_bp is not None:
                rec_bp.append((bp - uenv.scene.env_origins[:, None]).to(torch.float16).cpu().numpy())
            if args_cli.fall_mode == "contact":
                newly = (~fallen) & (contact_peak(s_fall) > args_cli.fall_force).any(-1)
                contact = (contact_peak(s_feet) > 1.0) & (~fallen)[:, None]        # Eq. S11
            else:
                z = bp[:, :, 2] - uenv.scene.env_origins[:, None, 2]
                newly = (~fallen) & ((z[:, bad] < args_cli.contact_z).any(-1) | (robot.data.projected_gravity_b[:, 2] > -0.5))
                contact = (z[:, feet] < args_cli.contact_z) & (~fallen)[:, None]
            fall_step[newly] = step; fallen |= newly
            # Eq. S11: E_slide = (1/T) * sum_t sum_{b in feet} 1[peak|f|>1N] * ||v_xy||. The denominator is T,
            # the number of timesteps, NOT the number of foot-contact frames -- dividing by contacts rescales
            # the metric by how much of the time the robot had feet down, so a hopping policy and a standing
            # one would be measured on different bases.
            sl_sum += (bv[:, feet, :2].norm(dim=-1) * contact).sum(-1); sl_n += (~fallen).float()
            if step % 250 == 0:
                print(f"[eval] batch {b+1}/{n_batches} step {step} fallen {int(fallen.sum())}/{N} infer {np.mean(infer[-250:]):.1f} ms")
        for i in range(N):
            segments[i].append((int(seg_start[i]), args_cli.steps, prompts[int(cur[i])]))
        take = min(N, args_cli.rollouts - tot["n"])
        # Paper 4.1: "semantic alignment and motion-quality metrics are computed only on rollouts that did not
        # fall". Pooling numerators over all rollouts instead makes every quality metric a function of the fall
        # rate -- a worse policy that falls early has its bad frames down-weighted -- and two rows with
        # different success rates stop being comparable. Average per rollout, over survivors only.
        ok = (~fallen[:take])
        per_sm = (sm_sum[:take] / sm_n[:take].clamp_min(1))[ok]
        per_tr = (tr_sum[:take] / tr_n[:take].clamp_min(1))[ok & (tr_n[:take] > 0)]
        per_sl = (sl_sum[:take] / sl_n[:take].clamp_min(1))[ok]
        tot["sm"] += per_sm.sum().item(); tot["sm_k"] += per_sm.numel()
        tot["tr"] += per_tr.sum().item(); tot["tr_k"] += per_tr.numel()
        tot["sl"] += per_sl.sum().item(); tot["sl_k"] += per_sl.numel()
        if rec_bp is not None:
            body_pos = np.stack(rec_bp)
            for i in range(take):
                L = int(fall_step[i]) if bool(fallen[i]) else args_cli.steps
                episodes.append(dict(rollout=tot["n"] + i, fell=bool(fallen[i]), fall_step=int(fall_step[i]),
                                     length=L, body_pos=body_pos[:L, i],
                                     segments=[(a0, min(b0, L), c) for a0, b0, c in segments[i] if a0 < L]))
        tot["n"] += take; tot["success"] += int(ok.sum())
        tot["fall_steps"] += fall_step[:take][fallen[:take]].tolist()
        print(f"[eval] batch {b+1} done: success so far {tot['success']}/{tot['n']} ({time.time()-t0:.0f}s)")
    res = {"ckpt": args_cli.ckpt, "rollouts": tot["n"], "success_rate": tot["success"] / tot["n"],
           "action_smoothness": tot["sm"] / max(tot["sm_k"], 1), "transition_smoothness": tot["tr"] / max(tot["tr_k"], 1),
           "foot_sliding_mps": tot["sl"] / max(tot["sl_k"], 1), "quality_rollouts": tot["sm_k"],
           "mean_fall_time_s": (float(np.mean(tot["fall_steps"])) / fps) if tot["fall_steps"] else None,
           "infer_ms": float(np.mean(infer)), "prompts": len(prompts), "ddim_steps": args_cli.ddim_steps, "guidance": args_cli.guidance, "contact_z": args_cli.contact_z}
    res["fall_mode"] = args_cli.fall_mode; res["fall_bodies"] = args_cli.fall_bodies
    res["fixed_noise"] = bool(args_cli.fixed_noise); res["init_pose"] = args_cli.init_pose
    res["source"] = args_cli.source
    if args_cli.source == "tracker":
        res["resume_path"] = args_cli.resume_path
    res["motion_list"] = args_cli.motion_list
    print("[eval] RESULT " + json.dumps(res))
    json.dump(res, open(args_cli.out, "w"), indent=1)
    if args_cli.record_body_pos:
        joblib.dump(dict(episodes=episodes, physical=res, body_names=body_names, fps=fps),
                    args_cli.record_body_pos, compress=3)
        print(f"[eval] wrote {args_cli.record_body_pos} ({os.path.getsize(args_cli.record_body_pos)/1e6:.0f} MB)")
    env.close()

if __name__ == "__main__":
    main(); app.close()
