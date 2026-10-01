"""Evaluate FRoM-W1's released G1 tracking policy on an AMASS motion library, with their own metrics.

Why this exists instead of `train_hydra.py --config-name=config_eval`: their eval runs inside the
training loop (`on_policy_runner.learn`), which calls `self.eval()` and then `self.alg.update(...,
dagger_only=True)`. `config_eval.yaml` sets `dagger.load_run_dagger` to
`25_12_05_23-30-46_OmniH2O_TEACHER_G1`, and the teacher is not released ("Teacher (TBD)"), so the
training step would fail. This builds the same env and runner from the same config and calls `eval()`
directly, so the unreleased teacher is never touched.

Everything measured is theirs: `OnPolicyRunner.eval()` sets `test`/`im_eval`, overrides the success
threshold to 0.5 m (NOT the 1.5 m in the config -- that one is the training terminator), walks the
motion library one motion per env via `begin_seq_motion_samples` / `forward_motion_samples`, and scores
with `phc.smpllib.smpl_eval.compute_metrics_lite`.

Checkpoint discovery follows legged_gym's convention, `<LEGGED_GYM_ROOT_DIR>/logs/<experiment_name>/
<load_run>/model_*.pt`, so the released weights are symlinked into place rather than copied; --link
does that and exits.

Run on a compute node:
  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/fromw1_eval_g1_policy.py --link
    python scripts/fromw1_eval_g1_policy.py \
      --motion-file /abs/path/amass_sub500.pkl --num-envs 500 \
      --load-run 25_12_11_18-16-37_OmniH2O_STUDENT \
      --out outputs/fromw1/eval_g1_full_sub500.json'
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H2H = REPO / "external/FRoM-W1/H-ACT/human2humanoid"
WEIGHTS = REPO / "external/fromw1_weights/hact/g1"
LEGGED_GYM = H2H / "legged_gym"
CFG_DIR = H2H / "legged_gym/legged_gym/cfg/cfg_g1"
# train.runner.experiment_name in the released env_cfg.json
EXPERIMENT = "robot:teleop"
LOG_ROOT = H2H / "legged_gym/logs" / EXPERIMENT


def link_weights():
    """Expose the released runs where legged_gym's `get_load_path` looks for them."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for run in sorted(p for p in WEIGHTS.iterdir() if p.is_dir()):
        dst = LOG_ROOT / run.name
        if dst.is_symlink() or dst.exists():
            print(f"[keep] {dst.relative_to(H2H)} -> {os.readlink(dst) if dst.is_symlink() else '(real dir)'}")
            continue
        dst.symlink_to(run)
        print(f"[link] {dst.relative_to(H2H)} -> {run}")
    print("\nruns visible to legged_gym:")
    for p in sorted(LOG_ROOT.iterdir()):
        models = sorted(q.name for q in p.iterdir() if "model" in q.name)
        print(f"  {p.name}: {models}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true", help="symlink the released runs, then exit")
    ap.add_argument("--motion-file", help="motion library .pkl (absolute path)")
    ap.add_argument("--load-run", default="25_12_11_18-16-37_OmniH2O_STUDENT")
    ap.add_argument("--num-envs", type=int, default=500,
                    help="one env per motion gives a single pass; their config_eval uses 406, which "
                         "presumably matched their own undisclosed eval set size")
    ap.add_argument("--out", required=False, help="where to write the metrics JSON")
    ap.add_argument("--device", default="cuda:0")
    args, overrides = ap.parse_known_args()

    if args.link:
        link_weights()
        return

    assert args.motion_file and args.out, "--motion-file and --out are required unless --link"
    motion_file = Path(args.motion_file).resolve()
    assert motion_file.exists(), motion_file
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_dir = LOG_ROOT / args.load_run
    assert run_dir.exists(), (
        f"{run_dir} not found -- run with --link first so legged_gym can see the released weights"
    )

    # cwd must be legged_gym/, NOT human2humanoid/. cfg_g1/asset/asset_teleop.yaml holds a plain
    # relative path, `resources/robots/g1/urdf/g1_21dof.urdf`, with no {LEGGED_GYM_ROOT_DIR} placeholder
    # to expand -- and the repo has two resources trees: human2humanoid/resources/robots/g1/urdf has
    # only g1_23dof.urdf, while legged_gym/resources/robots/g1/urdf has the 21-DoF asset the policy
    # needs. Running from human2humanoid/ (what their README's command implies) makes Isaac Gym report
    # "Failed to parse URDF file 'g1_21dof.urdf'" and then hand back a 0-DoF asset, which surfaces far
    # away as `could not broadcast input array from shape (21,) into shape (0,)`.
    assert (LEGGED_GYM / "resources/robots/g1/urdf/g1_21dof.urdf").exists(), "21-DoF asset missing"
    os.chdir(LEGGED_GYM)
    sys.path.insert(0, str(H2H))

    from isaacgym import gymapi      # noqa: E402  (before torch, on purpose)
    import torch                     # noqa: E402,F401
    import hydra                     # noqa: E402
    from omegaconf import OmegaConf   # noqa: E402
    from easydict import EasyDict     # noqa: E402
    import legged_gym.envs            # noqa: E402,F401  (import registers the tasks)
    from legged_gym.utils import task_registry  # noqa: E402

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg_hydra = hydra.compose(
            config_name="config_eval",
            overrides=[
                f"motion.motion_file={motion_file}",
                f"num_envs={args.num_envs}",
                f"sim_device={args.device}",
                f"load_run={args.load_run}",
                "headless=True",
                "use_wandb=False",
                *overrides,
            ],
        )
    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg.physics_engine = gymapi.SIM_PHYSX

    print("=== eval configuration ===")
    print(f"  motion_file          : {motion_file}")
    print(f"  load_run             : {args.load_run}")
    print(f"  num_envs             : {cfg.num_envs}")
    print(f"  teleop_obs_version   : {cfg.motion.teleop_obs_version}")
    print(f"  num_observations     : {cfg.env.num_observations}")
    print(f"  short_history_length : {cfg.env.short_history_length}")
    print(f"  train-time terminator: {cfg.asset.termination_scales.max_ref_motion_distance} m")
    print("  eval success thresh  : 0.5 m (hardcoded in OnPolicyRunner.eval)")

    env, env_cfg = task_registry.make_env_hydra(name=cfg.task, hydra_cfg=cfg, env_cfg=cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=cfg.task, args=cfg, train_cfg=cfg.train
    )
    print(f"  loaded policy        : {getattr(runner, 'loaded_policy_path', '(unknown)')}")
    print(f"  motions in library   : {env._motion_lib._num_unique_motions}")

    with torch.inference_mode():
        info = runner.eval()

    eval_info = dict(info.get("eval_info", {}))
    result = {
        "motion_file": str(motion_file),
        "n_motions": int(env._motion_lib._num_unique_motions),
        "load_run": args.load_run,
        "checkpoint": str(getattr(runner, "loaded_policy_path", "")),
        "num_envs": int(cfg.num_envs),
        "success_threshold_m": 0.5,
        "metrics": {k: float(v) for k, v in eval_info.items()},
        "n_failed": int(len(info.get("failed_keys", []))),
        "failed_keys": [str(k) for k in info.get("failed_keys", [])],
        "note": (
            "Single evaluation pass, single metric computation (project CLAUDE.md §4). "
            "Metrics from phc.smpllib.smpl_eval.compute_metrics_lite via OnPolicyRunner.eval(); "
            "units mm except success rate. accel_dist / vel_dist / mpjpe_pa are reported by their "
            "code over SUCCESSFUL motions only."
        ),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    for k, v in result["metrics"].items():
        print(f"  {k:22s} {v:.4f}")


if __name__ == "__main__":
    main()
