"""The kinematic reference ceiling: score the reference motions themselves through the G1 path.

`--source tracker` in `scripts/g1_eval_rollout.py` measures what a physically simulated, on-label motion
scores.  This measures what the SAME motion scores with the physics taken out entirely -- the reference the
tracker is trying to follow, straight out of `artifacts/<set>/<name>/motion.npz`.  The gap between the two is
the tracker's physical error; whatever is missing below this line is the joint mapping, the caption format and
the evaluator's domain gap, not anything our policy can fix.

Writes a pkl in exactly the shape `scripts/hml_phys/g1_eval_metrics.py` consumes, so the two rows go through
one identical scoring path.
"""
import argparse, glob, os, sys

import joblib
import numpy as np

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")

ap = argparse.ArgumentParser()
ap.add_argument("--motion_glob", default="TextOp/TextOpTracker/artifacts/val_all/*/motion.npz")
ap.add_argument("--meta_pkl", default="data/g1_motions/val_all_meta.pkl")
ap.add_argument("--max_motions", type=int, default=2048)
ap.add_argument("--max_s", type=float, default=20.0, help="same 20 s horizon as the rollout protocol")
ap.add_argument("--body_names", default="", help="pkl written by g1_eval_rollout.py, to copy its body order")
ap.add_argument("--out", required=True)
args = ap.parse_args()

meta = joblib.load(args.meta_pkl)
files = sorted(glob.glob(args.motion_glob))[: args.max_motions]
assert files, args.motion_glob
if args.body_names:
    body_names = joblib.load(args.body_names)["body_names"]
else:                       # the sim's own order, as printed by the rollout script
    body_names = None

episodes, skipped = [], 0
for f in files:
    name = os.path.basename(os.path.dirname(f))
    z = np.load(f)
    fps = float(z["fps"][0]) if z["fps"].shape else float(z["fps"])
    bp = z["body_pos_w"].astype(np.float32)
    L = min(len(bp), int(args.max_s * fps))
    ann = meta.get(name, {}).get("frame_ann", [])
    segs = []
    for a, b, lab, *_ in sorted(ann):
        lab = str(lab)
        if lab == "transition":
            continue
        s0, s1 = max(0, int(round(float(a) * fps))), min(L, int(round(float(b) * fps)))
        if s1 > s0:
            segs.append((s0, s1, lab))
    if not segs:
        skipped += 1; continue
    # the reference is kinematic: z is already the world height, but the pelvis offset is not env-relative
    episodes.append(dict(rollout=len(episodes), fell=False, fall_step=-1, length=L,
                         body_pos=bp[:L], segments=segs))
print(f"{len(episodes)} reference motions ({skipped} with no usable label), "
      f"{sum(len(e['segments']) for e in episodes)} segments")
if body_names is None:
    raise SystemExit("pass --body_names <a rollout pkl> so the joint mapping uses the simulator's body order")
joblib.dump(dict(episodes=episodes, body_names=body_names, fps=50.0,
                 physical=dict(source="reference_kinematic", rollouts=len(episodes), success_rate=1.0,
                               note="no physics: the reference motion the tracker follows")),
            args.out, compress=3)
print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB)")
