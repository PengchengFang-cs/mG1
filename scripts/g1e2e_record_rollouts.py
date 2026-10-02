"""Record (proprio, action, caption) rollouts of FRoM-W1's released G1 student tracking our references.

Step 2 of the end-to-end G1 line. The tracker is the teacher: it sees a reference trajectory and outputs
joint targets; we keep what it saw of its own body and what it did, and pair that with the clip's
HumanML3D captions. A policy trained on this has to infer from the text alone what the tracker inferred
from the reference -- which is exactly the ambiguity MIND's intent VAE exists to model.

Only successful episodes are kept, following MIND ("PHC 跟踪 HumanML3D，只留成功", docs/01 §一). What
"successful" means has to be said precisely, because the env changes the criterion under the flags this
config sets: with `env.test`/`env.im_eval` True, `legged_robot.py:618-630` switches the reference check
from "ANY body farther than the limit" to "the MEAN over bodies farther than the limit", and `env.test`
is required here (it is what gives `motion_start_times = 0`). The threshold is therefore the only free
part, and it is set to 0.5 m -- the authors' own success criterion, which `OnPolicyRunner.eval()`
substitutes for scoring -- rather than the 1.5 m `config_eval.yaml` ships or the 5.0 m of the training
config. At 1.5 m an episode survives with a 1.49 m AVERAGE body error, i.e. the robot stayed upright while
doing something other than what its caption says, and that clip then trains the policy under that caption.
The per-clip mean tracking error the env computes into `extras["mpjpe"]` is recorded alongside each clip
so the filter can be tightened afterwards without re-running the simulation.

Three different rates are involved and only one of them is a choice:

  reference keyframes  30 fps   how finely the motion library stores the reference. `_calc_frame_blend`
                               interpolates between keyframes at continuous time, so the tracker effectively
                               sees a 50 Hz reference; 30 fps affects interpolation accuracy, nothing else.
                               It comes from PHC's `target_fr = 30`, not from any FRoM-W1 model.
  control              50 Hz   sim dt 0.005 x decimation 4 = 0.02 s. The rate the tracker actually acts at
                               and the robot is driven at. Fixed by the released policy.
  generation           ours    at what rate our end-to-end policy emits actions -- a training-time choice,
                               not fixed here.

Recording is at the CONTROL rate, 50 Hz, because that is where the tracker's actions live: anything coarser
throws away commands that were really issued, and downsampling later is free. A policy that generates at
20 or 25 Hz and interpolates up to 50 (the setpoints are smooth PD targets, so this is sound, and it is what
works on the real robot) trains from exactly this data -- and gets more out of a fixed history budget:
MIND's 16 history frames cover 0.32 s at 50 Hz but 0.8 s at 20 Hz, which is the difference between seeing
a twitch and seeing a motion. What a lower decision rate may cost is the stabilising high-frequency content
of the tracker's own feedback, which is why the rate is a parameter to measure rather than a decision baked
into the data.

Token layout for this line (21 DoF, from the env's own buffers):
    proprio 51 = base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | dof_pos(21) | dof_vel(21)
    action  21   the tracker's RAW output, before the env touches it. The robot is actually driven by
                 target = default_dof_pos + 0.25 * (0.8 * a_t + 0.2 * a_{t-1}): legged_robot.py:299 clips
                 to +/-10, :315 applies an unconditional 0.8/0.2 action EMA (control.action_filt is False
                 and ctrl-delay randomisation is off, so this is the only filter), and :1731 scales by
                 0.25. The raw action is stored because that is what a policy trained on this data should
                 emit -- the env applies the same filter at evaluation time -- but any consumer that
                 reconstructs joint targets, or deploys outside this env, must apply the EMA itself.
                 A consequence worth knowing: the (proprio_t -> a_t) map is only Markov up to a_{t-1},
                 which the 51-d proprio does not contain.
    token   72 = proprio | action
Nothing is scaled or blanked here. A consumer that wants ADAPT-style scaling or a blanked linear velocity
can do it at training time; the recording stays raw so that choice is not baked in.

`--mix-prob` and `--policy-ckpt` are the on-policy hook: execute our own policy's action with that
probability while still recording the TRACKER's action as the label, so DAgger-style relabelling needs no
new recorder. Left at 0 the recording is pure teacher data.

Run on a compute node:
  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/g1e2e_record_rollouts.py \
      --refs data/g1_e2e/refs_train.pkl --text data/g1_e2e/refs_train.text.json \
      --out data/g1_e2e/rollouts_train.pkl'
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
EXPERIMENT = "robot:teleop"
LOG_ROOT = LEGGED_GYM / "logs" / EXPERIMENT

PROPRIO_DIM = 51
ACTION_DIM = 21
TOKEN_DIM = PROPRIO_DIM + ACTION_DIM
CONTROL_HZ = 50        # sim dt 0.005 x decimation 4; asserted against the env below
REF_FPS = 30           # the motion library's keyframe rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="reference library from g1e2e_build_references.py")
    ap.add_argument("--text", required=True, help="the matching .text.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-run", default="25_12_11_18-16-37_OmniH2O_STUDENT",
                    help="G1-Full; the G1-Clean student is 25_12_11_18-18-10_OmniH2O_STUDENT_FILTER")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--ref-dist", type=float, default=0.5,
                    help="mean-body reference-distance limit; 0.5 is the authors' own success criterion")
    ap.add_argument("--keep-failed", action="store_true",
                    help="keep terminated episodes too (truncated at termination); off by default, as MIND does")
    ap.add_argument("--policy-ckpt", default="", help="on-policy hook: our own policy, for DAgger")
    ap.add_argument("--mix-prob", type=float, default=0.0, help="probability of executing our action instead")
    ap.add_argument("--device", default="cuda:0")
    args, overrides = ap.parse_known_args()

    refs = Path(args.refs).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert refs.exists(), refs
    text_index = json.loads(Path(args.text).resolve().read_text())
    run_dir = LOG_ROOT / args.load_run
    assert run_dir.exists(), (
        f"{run_dir} not found -- run `python scripts/fromw1_eval_g1_policy.py --link` first")
    assert not args.policy_ckpt and args.mix_prob == 0.0, (
        "the on-policy path is wired but not implemented yet; leave --policy-ckpt empty and --mix-prob 0. "
        "Accepting a non-zero --mix-prob and ignoring it would silently produce pure-teacher data.")

    # cwd must be legged_gym/: cfg_g1/asset/asset_teleop.yaml holds a plain relative asset path with no
    # {LEGGED_GYM_ROOT_DIR} placeholder, and the repo has two resources trees -- only this one has the
    # 21-DoF asset (docs/09 §6.14).
    assert (LEGGED_GYM / "resources/robots/g1/urdf/g1_21dof.urdf").exists(), "21-DoF asset missing"
    os.chdir(LEGGED_GYM)
    sys.path.insert(0, str(H2H))

    from isaacgym import gymapi          # noqa: E402  (before torch, on purpose)
    import torch                         # noqa: E402
    import numpy as np                   # noqa: E402
    import joblib                        # noqa: E402
    import hydra                         # noqa: E402
    from omegaconf import OmegaConf       # noqa: E402
    from easydict import EasyDict         # noqa: E402
    import legged_gym.envs                # noqa: E402,F401  (import registers the tasks)
    from legged_gym.utils import task_registry   # noqa: E402

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_hydra = hydra.compose(config_name="config_eval", overrides=[
            f"motion.motion_file={refs}", f"num_envs={args.num_envs}", f"sim_device={args.device}",
            f"load_run={args.load_run}", "headless=True", "use_wandb=False",
            f"asset.termination_scales.max_ref_motion_distance={args.ref_dist}", *overrides])
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg.physics_engine = gymapi.SIM_PHYSX

    env, _ = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=cfg.task, args=cfg, train_cfg=cfg.train)
    policy = runner.get_inference_policy(device=env.device)
    lib = env._motion_lib
    n_motions = int(lib._num_unique_motions)
    # 0.1 Hz, not an exact match: decimation * sim_dt is 0.020000000000000004 in floating point. The
    # mistake worth catching is a 30-vs-50 mix-up, which any tolerance under 20 Hz catches.
    assert abs(1.0 / env.dt - CONTROL_HZ) < 0.1, (
        f"env control rate is {1.0 / env.dt:.3f} Hz, not the {CONTROL_HZ} Hz this recorder labels its data "
        f"with; fix CONTROL_HZ or the config, do not let the two disagree")
    print(f"motions {n_motions}   envs {args.num_envs}   "
          f"action_scale {cfg.control.action_scale}   ref-distance limit "
          f"{cfg.asset.termination_scales.max_ref_motion_distance} m")

    # Walk the library sequentially, one motion per env, exactly as OnPolicyRunner.eval() does.
    env.cfg.env.test = True
    env.begin_seq_motion_samples()
    obs, _ = env.reset()

    data, n_fail, n_done = {}, 0, 0
    t0 = time.time()
    while n_done < n_motions:
        ids = lib._curr_motion_ids.clone()
        keys = [str(k) for k in lib._motion_data_keys[ids.cpu().numpy()]]
        # get_motion_num_steps() counts in 30 Hz units (num_frames * 30 / motion_fps) while the env steps
        # at 1/env.dt = 50 Hz, so using it directly as a horizon -- which OnPolicyRunner.eval() does --
        # plays only 30/50 = 60% of each motion. Work in seconds and convert with the env's own dt.
        # No motion_ids here, deliberately. The library loads only num_envs motions at a time (its
        # "Loaded N motions" line), so its per-motion tensors are LOCAL to the current batch and indexed
        # by env, while _curr_motion_ids holds GLOBAL ids into the full library. Passing the global ids in
        # indexes a 512-entry tensor with 512..1023 and triggers a device-side assert in the CUDA index
        # kernel -- which surfaces asynchronously, at whatever line happens to synchronise next.
        # OnPolicyRunner.eval() never hits this because it always calls these accessors with no ids.
        # _motion_data_keys is the exception: it IS the full list, so the global ids are correct there.
        secs = lib.get_motion_length().clone()
        assert secs.shape[0] == env.num_envs, (
            f"library returned {secs.shape[0]} motion lengths for {env.num_envs} envs; these tensors are "
            f"per-env and local to the loaded batch")
        steps = (secs / env.dt).ceil().long()
        horizon = int(steps[: min(len(steps), args.num_envs)].max().item())

        mpjpe_sum = torch.zeros(env.num_envs, dtype=torch.float64)
        mpjpe_n = torch.zeros(env.num_envs, dtype=torch.float64)
        prop = torch.zeros((horizon, env.num_envs, PROPRIO_DIM), dtype=torch.float32)
        act = torch.zeros((horizon, env.num_envs, ACTION_DIM), dtype=torch.float32)
        length = torch.zeros(env.num_envs, dtype=torch.long)
        failed = torch.zeros(env.num_envs, dtype=torch.bool)
        # `horizon` is the LONGEST motion in the batch, so a short clip's env keeps stepping after its own
        # reference has run out -- the motion library clamps the time, the reference freezes on its last
        # frame and the robot just stands there. Those trailing frames are not the motion and must not be
        # recorded, so each env stops contributing at its own step count.
        active = torch.ones(env.num_envs, dtype=torch.bool)
        steps_cpu = steps[: env.num_envs].cpu()

        for t in range(horizon):
            # proprio BEFORE the step: the state the action is applied in, which is the pairing a policy
            # trained on this has to reproduce.
            prop[t] = torch.cat([env.base_lin_vel, env.base_ang_vel, env.projected_gravity,
                                 env.dof_pos, env.dof_vel], dim=-1).detach().cpu()
            with torch.inference_mode():
                a = policy(obs.detach())
            act[t] = a.detach().cpu()
            obs, _, _, dones, infos = env.step(a.detach())
            if isinstance(infos, dict) and "mpjpe" in infos:
                m = infos["mpjpe"].detach().cpu().double()
                mpjpe_sum += m * active.double()
                mpjpe_n += active.double()
            length[active] = t + 1
            # Reaching the end of its own motion is not a failure; terminating before that is. The env
            # already distinguishes the two: `terminate_by_1time_motion` is True for g1
            # (cfg_g1/asset/asset_teleop.yaml), so `time_out_buf` IS `time > motion_len`
            # (legged_robot.py:643-646). Using it is exact, where the earlier +/-3-step slack both
            # deleted the last 2-3 frames of EVERY clip (measured: exactly 3 on 1539 of 1560) and hid a
            # fall occurring inside that window by recording it as a success.
            timed_out = env.time_out_buf.detach().cpu().bool()
            d = dones.detach().cpu().bool()
            newly_failed = d & active & ~timed_out
            failed |= newly_failed
            active &= ~((d & timed_out) | newly_failed)
            if not bool(active.any()):
                break

        for i, key in enumerate(keys):
            if key in data or key not in text_index:
                continue
            L = min(int(length[i]), int(steps_cpu[i]))
            if L < 20:
                continue
            if bool(failed[i]) and not args.keep_failed:
                n_fail += 1
                continue
            n_fail += int(bool(failed[i]))
            data[key] = dict(
                proprio=prop[:L, i].numpy().astype(np.float32),
                action=act[:L, i].numpy().astype(np.float32),
                failed=bool(failed[i]),
                mpjpe_mean_m=float(mpjpe_sum[i] / mpjpe_n[i]) if float(mpjpe_n[i]) > 0 else float("nan"),
                n_frames=L,
                fps=CONTROL_HZ,
                captions=text_index[key]["captions"],
                split=text_index[key]["split"],
                source_file=text_index[key]["source_file"],
            )
        n_done += min(env.num_envs, n_motions - n_done)
        print(f"  {n_done}/{n_motions} motions, kept {len(data)}, failed {n_fail}, "
              f"{time.time() - t0:.0f}s")
        if n_done < n_motions:
            env.forward_motion_samples()
            obs, _ = env.reset()

    n_short = n_motions - len(data) - n_fail
    frames = sum(v["n_frames"] for v in data.values())
    print(f"\nkept {len(data)}/{n_motions} clips ({n_fail} failed, {n_short} too short/dup), {frames} frames "
          f"({frames / CONTROL_HZ / 60:.1f} min) in {time.time() - t0:.0f}s")
    print(f"token layout: proprio {PROPRIO_DIM} | action {ACTION_DIM} = {TOKEN_DIM}")
    joblib.dump(data, out)
    meta = dict(n_clips=len(data), n_motions=n_motions, n_failed=n_fail, n_dropped_other=n_short,
                frames=frames,
                minutes=frames / CONTROL_HZ / 60, proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM,
                token_dim=TOKEN_DIM, action_scale=float(cfg.control.action_scale),
                control_hz=CONTROL_HZ, ref_fps=REF_FPS,
                tracker=args.load_run, refs=str(refs),
                ref_distance_limit=float(cfg.asset.termination_scales.max_ref_motion_distance),
                ref_distance_is_mean_over_bodies=True, action_is_pre_ema=True,
                keep_failed=bool(args.keep_failed))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
