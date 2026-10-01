"""Build the G1 21-DoF AMASS motion library the released FRoM-W1 tracking policy expects.

The policy's `env_cfg.json` points at `resources/motions/g1/amass_all_21dof.pkl`, and neither that file
nor a script that produces it is in the release: `resources/motions/` does not exist, human2humanoid's
README says the preprocessed G1/H1 datasets are "(TODO: Add download links)", and the only grad-fit
driver shipped, `scripts/data_process/grad_fit_h1.py`, is H1-only (19 DoF hardcoded, H1 joint names,
`torch_h1_humanoid_batch`, `data/h1/shape_optimized_v1.pkl`) -- and ends in `ipdb.set_trace()` without
ever saving `data_dump`, so it is a debug leftover rather than the production path.

So this script stands in for the missing G1 driver. Everything load-bearing is theirs:

  * the gradient fit (losses, optimiser, schedule) is transcribed from
    `retarget/body_retarget/grad_fit_robot.py:process_data`, FRoM-W1's own generic retargeter;
  * the forward kinematics is their `retarget/body_retarget/robot.py:Humanoid_Batch`;
  * the SMPL body shape is their released `assets/beta/shape_optimized_g1.pkl` -- body proportions do
    not change between the 21/23/29-DoF variants, which are one robot with joints locked;
  * the 30 Hz resampling and the occlusion filter are transcribed from
    `scripts/data_process/process_amass_db.py:175-195`;
  * the robot config comes from `fromw1_g1_21dof_config.py`, parsed out of their `g1_21dof.xml`.

Four places where theirs could not be used as-is, each a deliberate, recorded choice (docs/09 §6.6):

1. **Fit at 21 DoF, not 29-then-drop.** Their `G1Config` is 29 DoF. They never say how
   `amass_all_21dof.pkl` was made. Fitting with the wrists and waist free and then discarding those
   eight joints would not give the best 21-DoF reference, so the fit is constrained to 21 from the
   start. OUR CHOICE, not their specification.
2. **Ground the root per clip; do not pin it.** Their `G1Config.FIX_BASE_HEIGHT = True` holds the root
   at a constant 0.75 m, which suits the deployment path but would erase every crouch, sit and jump
   from a tracking reference. We use the grounding `grad_fit_h1.py:219` applies instead.
3. **No `[2,0,1]` axis permutation.** `grad_fit_robot.py` permutes the dumped root translation for the
   deployment consumer; the motion library is in the robot's own z-up frame (`grad_fit_h1.py:225`).
4. **Resample off the real mocap framerate.** `grad_fit_robot.load_amass_data` hardcodes `"fps": 30`
   with the real framerate commented out, so `skip = fps // 30` is always 1 -- feeding it raw 120 Hz
   AMASS yields 4x slow motion. `process_amass_db.py:175` does it correctly and is what we follow.
   (`process_amass_db.py` itself cannot be run as shipped: `ipdb.set_trace()` sits inside its loop and
   it imports the uninstalled `uhc` package, so its logic is transcribed rather than called.)

Also skipped: `fix_height_smpl_vanilla`, which `process_amass_db.py` applies to the SMPL trans. It
lives in `uhc`, which is not installed and not a dependency of this release. It shifts both the fit
target and the shared root by the same constant, and the robot is grounded afterwards anyway, so the
fit is unaffected up to that offset.

Run on a compute node:
  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/fromw1_amass_to_g1_21dof.py --amass data/amass --out data/g1_21dof/amass_all_21dof.pkl'
"""
import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
RETARGET = REPO / "external/FRoM-W1/H-ACT/retarget"

# Hyperparameters exactly as grad_fit_robot.py:51-55 sets them.
SMOOTH_WEIGHT = 1e-3
MAX_ITER_PATIENCE = 100
MIN_LR = 1e-5
INIT_LR = 1e-1
WEIGHT_DECAY = 1e-5
MAX_ITER = 1000
TARGET_FR = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", required=True, help="directory of extracted AMASS dataset folders")
    ap.add_argument("--out", required=True, help="output .pkl motion library")
    ap.add_argument("--xml", default=None, help="g1_21dof.xml (default: the policy's own asset)")
    ap.add_argument("--occlusion", default=None, help="amass_copycat_occlusion_v3.pkl")
    ap.add_argument("--splits", default="train", help="comma-separated: train,vald,test")
    ap.add_argument("--min-frames", type=int, default=10, help="as process_amass_db.py:196")
    ap.add_argument("--limit", type=int, default=0, help="stop after N sequences (smoke test)")
    ap.add_argument("--sample", type=int, default=0,
                    help="draw a fixed random subset of N sequences (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="split the sequence list across GPUs")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    amass_dir = Path(args.amass).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml = Path(
        args.xml
        or REPO / "external/FRoM-W1/H-ACT/human2humanoid/legged_gym/resources/robots/g1/xml/g1_21dof.xml"
    ).resolve()
    occl_path = Path(
        args.occlusion or REPO / "UniPhys/sample_data/amass_copycat_occlusion_v3.pkl"
    ).resolve()

    sys.path.insert(0, str(REPO / "scripts"))
    # The retarget package resolves "assets/...", "models/smpl" relative to its own root, and importing
    # body_retarget loads the SMPL parser and the G1 betas at module level, so chdir first.
    os.chdir(RETARGET)
    sys.path.insert(0, str(RETARGET))

    from body_retarget.robot import Humanoid_Batch                      # noqa: E402
    from body_retarget.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES  # noqa: E402
    from body_retarget.grad_fit_robot import get_joint_global_rot        # noqa: E402
    from fromw1_g1_21dof_config import G121DOFConfig                    # noqa: E402

    device = torch.device(args.device)
    cfg = G121DOFConfig(str(xml), SMPL_BONE_ORDER_NAMES)
    robot = Humanoid_Batch(cfg=cfg, device=device)

    # joints_range from the MJCF has one row per JOINT (21); the dof variable has one row per non-root
    # BODY (23). Swap in the body-aligned table so clamp_ lines up and the two locked rows stay at 0.
    robot.joints_range = cfg.joints_range_expanded.to(device)

    shape_new, scale = joblib.load("assets/beta/shape_optimized_g1.pkl")
    shape_new, scale = shape_new.to(device), scale.to(device)
    smpl_parser_n = SMPL_Parser(model_path="models/smpl", gender="neutral")
    smpl_parser_n.to(device)

    print(f"robot      : {xml.name}  bodies={len(cfg.body_names)}  axis_rows={cfg.JOINT_NUM}")
    print(f"actuated   : {len(cfg.actuated_joint_names)} joints, order verified against the policy")
    print(f"locked     : {[cfg.body_names[1:][i] for i in range(cfg.JOINT_NUM) if i not in cfg.actuated_row_idx]}")
    print(f"extend     : {cfg.Extend.extend_link_name} -> parents {cfg.Extend.extend_parent_idx}")

    occlusion = joblib.load(occl_path) if occl_path.exists() else {}
    print(f"occlusion  : {len(occlusion)} entries from {occl_path.name}")

    # process_amass_db.py:242-261 -- the folder names each split matches on.
    SPLITS = {
        "vald": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
        "test": ["Transitions_mocap", "SSM_synced"],
        "train": [
            "CMU", "MPI_Limits", "TotalCapture", "Eyes_Japan_Dataset", "KIT", "BML", "EKUT",
            "TCD_handMocap", "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab",
            "Eyes", "DFaust",
        ],
    }
    wanted = set()
    for s in args.splits.split(","):
        wanted.update(SPLITS[s.strip()])

    # Extracted folder names do not always equal the split names (DFaust -> DFaust_67,
    # BioMotionLab_NTroje for BML, ...), so match by prefix on the folder name.
    present = sorted(p for p in amass_dir.iterdir() if p.is_dir())
    selected = [p for p in present if any(p.name.startswith(w) or w.startswith(p.name) for w in wanted)]
    print(f"datasets   : {len(selected)}/{len(present)} selected -> {[p.name for p in selected]}")
    assert selected, f"no AMASS dataset folder under {amass_dir} matched split(s) {args.splits}"

    seqs = []
    for d in selected:
        for npz in sorted(d.rglob("*.npz")):
            if npz.name.endswith("shape.npz"):
                continue
            seqs.append((d.name, npz))
    print(f"sequences  : {len(seqs)}")
    # Drawn before sharding so every shard sees the same pool and the union is exactly the sample.
    # The chosen sequences are written next to the output so the subset is reproducible and auditable.
    if args.sample and args.sample < len(seqs):
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(len(seqs), size=args.sample, replace=False).tolist())
        seqs = [seqs[i] for i in pick]
        manifest = out_path.with_name(out_path.stem + ".manifest.txt")
        manifest.write_text("\n".join(str(n) for _, n in seqs) + "\n")
        print(f"             sampled {len(seqs)} (seed {args.seed}) -> {manifest.name}")
    if args.limit:
        seqs = seqs[: args.limit]
        print(f"             limited to {len(seqs)} for this run")
    if args.nshards > 1:
        seqs = seqs[args.shard :: args.nshards]
        print(f"             shard {args.shard}/{args.nshards} -> {len(seqs)} sequences")

    data_dump, skipped = {}, {}
    t0 = time.time()
    for ds_name, npz in tqdm(seqs, desc="retarget"):
        key = "0-" + f"{ds_name}_{npz.relative_to(npz.parents[1]).with_suffix('')}".replace("/", "_")
        try:
            raw = dict(np.load(open(npz, "rb"), allow_pickle=True))
        except Exception as e:
            skipped[key] = f"unreadable: {e}"
            continue

        fr = raw.get("mocap_framerate", raw.get("mocap_frame_rate"))
        if fr is None or "poses" not in raw or "trans" not in raw:
            skipped[key] = f"missing keys: {sorted(raw)}"
            continue

        # process_amass_db.py:175-178
        skip = max(1, int(float(fr) / TARGET_FR))
        poses = raw["poses"][::skip]
        trans_np = raw["trans"][::skip]

        # process_amass_db.py:181-195 -- the occlusion filter, by the same rules.
        bound = poses.shape[0]
        if key in occlusion:
            issue = occlusion[key]["issue"]
            if issue in ("sitting", "airborne") and "idxes" in occlusion[key]:
                bound = occlusion[key]["idxes"][0]
                if bound < args.min_frames:
                    skipped[key] = f"occlusion bound too small ({bound})"
                    continue
            else:
                skipped[key] = f"occlusion irrecoverable: {issue}"
                continue
        poses, trans_np = poses[:bound], trans_np[:bound]
        if poses.shape[0] < args.min_frames:
            skipped[key] = f"too short ({poses.shape[0]})"
            continue

        N = poses.shape[0]
        # SMPL, not SMPL-H: keep the 22 body joints, zero the hands (process_amass_db.py:206).
        pose_aa_np = np.concatenate([poses[:, :66], np.zeros((N, 6))], axis=-1)
        # NO pre-rotation of the root here. The two code paths differ and they are not interchangeable:
        #   grad_fit_h1.py  (the motion-library producer): root stays R, gt_root_rot = R * Q^-1
        #   grad_fit_robot.py (the deployment path): load_amass_data sets root = Q*R first, so
        #                     gt_root_rot = (Q*R) * Q^-1 = Q R Q^-1, and the dump is then permuted
        #                     [2,0,1] for its consumer.
        # Copying the deployment pre-rotation while writing library-format output (no permutation) puts
        # the whole clip in a Q-rotated world -- Q = quat(0.5,0.5,0.5,0.5) is a 120 deg turn about
        # (1,1,1)/sqrt(3) -- so the robot spawns lying on its side and every episode terminates on the
        # first step. The fit error does NOT catch it: the SMPL target is rotated by the same Q, so the
        # fit stays self-consistent at ~5.7 cm. Measured with the env's own FK, head minus pelvis was
        # -0.09/-0.04/+0.04/-0.04 m with the pre-rotation and +0.44 m (the head extend offset) without.

        trans = torch.from_numpy(trans_np).float().to(device)
        pose_aa_walk = torch.from_numpy(pose_aa_np).float().to(device)

        # Zero-beta pass only to recover the root offset, as grad_fit_robot.py:178 does.
        _, joints0 = smpl_parser_n.get_joints_verts(
            pose_aa_walk, torch.zeros((1, 10)).to(device), trans
        )
        root_trans_offset = trans + (joints0[:, 0] - trans)

        gt_root_rot = (
            torch.from_numpy(
                (
                    sRot.from_rotvec(pose_aa_np[:, :3])
                    * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                ).as_rotvec()
            ).float().to(device)
        )

        dof_pos_new = Variable(
            torch.zeros((1, N, cfg.JOINT_NUM, 1), device=device), requires_grad=True
        )
        optimizer = torch.optim.Adam([dof_pos_new], lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5, min_lr=MIN_LR
        )
        axis = cfg.ROBOT_ROTATION_AXIS[None].to(device)
        n_extend = len(cfg.Extend.extend_parent_idx)
        zeros_extend = torch.zeros((1, N, n_extend, 3), device=device)

        # grad_fit_robot.py recomputes the SMPL forward pass (and the global rotations it needs for the
        # wrist term) on every iteration, but neither depends on dof_pos_new -- they are constant for
        # the clip. Hoisting them out is mathematically identical and is what makes a 14k-sequence run
        # finish in hours instead of days.
        with torch.no_grad():
            _, joints_fit = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints_fit[:, 0:1]
            joints_fit = (joints_fit - root_pos) * scale + root_pos
            target = joints_fit[:, cfg.smpl_joint_pick_idx]

        patience = 0
        for _ in range(MAX_ITER):
            pose_aa_robot = torch.cat(
                [gt_root_rot[None, :, None], axis * dof_pos_new, zeros_extend], axis=2
            )
            fk = robot.fk_batch(pose_aa_robot, root_trans_offset[None])
            diff = fk["global_translation_extend"][:, :, cfg.robot_joint_pick_idx] - target
            # cfg.wrist_pick_idx is empty at 21 DoF, so the geodesic term is dropped exactly as
            # grad_fit_robot.py:231 does when a robot has no wrist joints.
            loss = diff.norm(dim=-1).mean() + SMOOTH_WEIGHT * torch.norm(
                dof_pos_new[:, 1:] - dof_pos_new[:, :-1], p=2
            )

            scheduler.step(loss)
            if optimizer.param_groups[0]["lr"] <= MIN_LR:
                patience += 1
            if patience >= MAX_ITER_PATIENCE:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dof_pos_new.data.clamp_(
                robot.joints_range[:, 0, None], robot.joints_range[:, 1, None]
            )

        dof_pos_new.data.clamp_(robot.joints_range[:, 0, None], robot.joints_range[:, 1, None])
        pose_aa_robot = torch.cat(
            [gt_root_rot[None, :, None], axis * dof_pos_new, zeros_extend], axis=2
        )
        fk = robot.fk_batch(pose_aa_robot, root_trans_offset[None])

        # grad_fit_h1.py:219 -- drop the clip so its lowest body sits 8 cm above the floor. No axis
        # permutation: the motion library is in the robot's own z-up frame.
        root_dump = root_trans_offset.clone()
        root_dump[..., 2] -= fk["global_translation"][..., 2].min().item() - 0.08

        final_err = float(
            (fk["global_translation_extend"][:, :, cfg.robot_joint_pick_idx] - target)
            .norm(dim=-1).mean().item()
        )

        dof_full = dof_pos_new.squeeze(0).squeeze(-1)                  # (N, 23) body-aligned
        dof_act = dof_full[:, cfg.actuated_row_idx]                    # (N, 21) policy order

        data_dump[key] = {
            "root_trans_offset": root_dump.squeeze().cpu().numpy(),
            "pose_aa": pose_aa_robot.squeeze(0).detach().cpu().numpy(),
            "dof": dof_act.detach().cpu().numpy(),
            "dof_full": dof_full.detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            "fps": TARGET_FR,
            "body_names": robot.model_names,
            "dof_names": cfg.actuated_joint_names,
            "fit_err_m": final_err,
        }
        del dof_pos_new, fk, pose_aa_robot
        torch.cuda.empty_cache()

    print(f"\nretargeted {len(data_dump)} / {len(seqs)} in {time.time() - t0:.0f}s")
    if skipped:
        print(f"skipped {len(skipped)}:")
        from collections import Counter
        for reason, n in Counter(r.split(":")[0].split("(")[0].strip() for r in skipped.values()).most_common():
            print(f"  {n:6d}  {reason}")
    if data_dump:
        errs = np.array([v["fit_err_m"] for v in data_dump.values()])
        print(f"fit error (m): mean {errs.mean():.4f}  median {np.median(errs):.4f}  p95 {np.percentile(errs, 95):.4f}  max {errs.max():.4f}")

    joblib.dump(data_dump, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    joblib.dump(skipped, out_path.with_suffix(".skipped.pkl"))


if __name__ == "__main__":
    main()
