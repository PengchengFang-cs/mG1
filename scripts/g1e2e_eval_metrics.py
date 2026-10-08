"""Full-spectrum metrics for the end-to-end G1 line, computed on rollouts ALREADY TAKEN.

CLAUDE.md §13 was changed on 2026-10-08: survival alone is no longer the report. This script covers
the physical-plausibility and fidelity halves of the new §13. It reads the `*.bodypos.npz` that every
closed loop already writes, so it needs NO new rollout -- §4 holds (one rollout per checkpoint, each
metric computed once).

    survival     fall rate, duration completion                     -- recomputed, must match the eval log
    physics      Floating, Penetration, Foot-sliding, Jerk           -- hml_phys/phys_metrics.py
    fidelity     actual / reference root speed, pelvis height, path  -- against the reference library

The three semantic metrics (R@1/R@2/R@3, FID) are NOT here. They need the 22 G1 bodies mapped onto
SMPL's 22 joints, and `hml_phys/g1_to_smpl.py` cannot do it for this robot: its DIRECT map wants
`torso_link`, `left_wrist_yaw_link` and `right_wrist_yaw_link`, and legged_gym loads the asset with
`collapse_fixed_joints=True`, so the 21-DoF G1's rigid bodies are exactly `pelvis` plus the child link
of each actuated joint -- 22 bodies with no torso, no wrists and no head. That mapper was written for a
G1 that keeps them. Fixing it is a separate change.

WHAT IS AND IS NOT COMPARABLE ACROSS PAPERS (CLAUDE.md §2):
  Floating / Penetration / Foot-sliding  -- comparable.
  Jerk                                   -- NOT. It is a third difference per frame, so it scales with
                                            the frame rate; ours is 50 Hz, the SMPL-line default 30 Hz.
                                            The rate is reported alongside it for that reason.

The fallen convention is TRUNCATE (CLAUDE.md §2): every clip is kept, cut at its fall. Excluding fallen
clips scores only the surviving segments, which flatters a policy that falls more.

Run on a compute node:
    python scripts/g1e2e_eval_metrics.py \
      --npz outputs/g1e2e/eval_ppo_ppoA.bodypos.npz outputs/g1e2e/eval_G_cfgA1.0.bodypos.npz \
      --refs data/g1_e2e/refs_train_part1.pkl --out outputs/g1e2e/metrics_full.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hml_phys.phys_metrics import all_metrics_raw          # noqa: E402

CONTROL_HZ = 50.0
# The 21-DoF G1's rigid bodies, in the order isaacgym returns them with fixed joints collapsed:
# `pelvis` plus the child link of each actuated joint. Verified against
# gym.get_asset_rigid_body_names(g1_21dof.urdf, collapse_fixed_joints=True).
PELVIS = 0
FEET = [6, 12]            # left_ankle_roll_link, right_ankle_roll_link -- no toe bodies exist
N_BODIES = 22


def per_clip(bp, n):
    """bp [T,22,3] z-up metres, n valid frames -> the fidelity scalars for one clip."""
    p = bp[:n].astype(np.float64)
    root = p[:, PELVIS]
    if n > 1:
        v = np.diff(root, axis=0) * CONTROL_HZ
        speed = float(np.linalg.norm(v[:, :2], axis=-1).mean())
    else:
        speed = 0.0
    return dict(root_speed_xy=speed, pelvis_z_mean=float(root[:, 2].mean()),
                pelvis_z_min=float(root[:, 2].min()), n=int(n))


def ref_speed(v):
    """Mean horizontal root speed the reference asks for, m/s."""
    t = v["root_trans_offset"].astype(np.float64)
    fps = float(v["fps"])
    if t.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(t[:, :2], axis=0) * fps, axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", type=int, default=4,
                    help="clips shorter than this after truncation carry no usable difference")
    args = ap.parse_args()

    lib = joblib.load(args.refs)
    rspeed = {}
    out = {}
    for f in args.npz:
        d = np.load(f)
        if "body_pos" not in d.files:
            # The teacher-mode rollouts saved only what §5.8's failure attribution needed. Physics and
            # fidelity cannot be recovered from them, and re-running the teacher to capture positions
            # would be a third rollout of a configuration that has already had two (§4).
            print(f"SKIP {Path(f).name}: no body_pos (keys: {d.files})")
            out[Path(f).name] = dict(npz=f, skipped="no body_pos stored in this rollout")
            continue
        bp, fs, hz = d["body_pos"], d["fall_step"].astype(int), d["horizon"].astype(int)
        keys = [str(k) for k in d["keys"]]
        assert bp.shape[2] == N_BODIES, (
            f"{f} has {bp.shape[2]} bodies, not {N_BODIES}; FEET={FEET} assumes the 21-DoF "
            f"collapsed body list")
        fell = fs < hz
        valid = np.minimum(fs, hz)              # TRUNCATE: keep every clip, cut it at its fall

        pos_list, props, ratios = [], [], []
        short = 0
        for i, k in enumerate(keys):
            n = int(valid[i])
            if n < args.min_frames:
                short += 1
                continue
            pos_list.append(bp[i, :n].astype(np.float64))
            pr = per_clip(bp[i], n)
            if k not in rspeed:
                assert k in lib, f"clip {k} is absent from {args.refs}"
                rspeed[k] = ref_speed(lib[k])
            pr["ref_speed_xy"] = rspeed[k]
            # the ratio is only meaningful where the reference actually asks for motion
            if rspeed[k] > 0.02:
                ratios.append(pr["root_speed_xy"] / rspeed[k])
            pr["key"] = k
            props.append(pr)

        phys = all_metrics_raw(pos_list, feet=FEET, fps=CONTROL_HZ)
        rat = np.array(ratios) if ratios else np.array([np.nan])
        res = dict(
            npz=f, n_clips=len(keys), n_fell=int(fell.sum()), n_too_short=short,
            fall_rate=float(fell.mean()),
            duration_completion=float((valid / np.maximum(hz, 1)).mean()),
            floating_mm=phys["floating_mm"], penetration_mm=phys["penetration_mm"],
            foot_sliding_mm=phys["skating_mm"], jerk_mm_frame3=phys["jerk_mm_frame3"],
            jerk_fps=phys["fps"],
            root_speed_xy=float(np.mean([p["root_speed_xy"] for p in props])),
            ref_speed_xy=float(np.mean([p["ref_speed_xy"] for p in props])),
            speed_ratio_median=float(np.nanmedian(rat)),
            speed_ratio_mean=float(np.nanmean(rat)), n_speed_ratio=int(len(ratios)),
            pelvis_z_mean=float(np.mean([p["pelvis_z_mean"] for p in props])),
            pelvis_z_min=float(np.min([p["pelvis_z_min"] for p in props])),
        )
        out[Path(f).name] = res

    out = {k: v for k, v in out.items() if "skipped" not in v} or out
    name_w = max(len(n) for n in out) + 2
    cols = [("fall", "fall_rate", "{:.4f}"), ("duration", "duration_completion", "{:.4f}"),
            ("Float mm", "floating_mm", "{:.2f}"), ("Penet mm", "penetration_mm", "{:.2f}"),
            ("Slide mm", "foot_sliding_mm", "{:.2f}"), ("Jerk mm", "jerk_mm_frame3", "{:.2f}"),
            ("v act", "root_speed_xy", "{:.3f}"), ("v ref", "ref_speed_xy", "{:.3f}"),
            ("v ratio", "speed_ratio_median", "{:.2f}"), ("pelvis z", "pelvis_z_mean", "{:.3f}")]
    print(f"{'rollout':<{name_w}}" + "".join(f"{h:>10}" for h, _, _ in cols))
    for n, r in out.items():
        print(f"{n:<{name_w}}" + "".join(f"{fmt.format(r[k]):>10}" for _, k, fmt in cols))
    print("\ntruncate-fallen convention; Jerk is frame-rate dependent "
          f"({CONTROL_HZ:.0f} Hz here) and is NOT comparable across papers; "
          "Floating / Penetration / Foot-sliding are.")
    print("v ratio is the median of actual/reference root speed over clips whose reference moves "
          "faster than 0.02 m/s. 1.0 is correct; above 1.0 means moving more than asked.")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
