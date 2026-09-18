"""Paper-aligned physics metrics (SCRIPT Appendix D convention) on raw 30 Hz simulator output.

Usage: 08_paper_metrics.py --rollouts a.pkl b.pkl [--physgt]
Prints Floating / Penetration / Skating / Jerk / Duration for each file, plus the physics ground truth.
Everything is recomputed from stored body positions; no simulation, no GPU.
"""
import argparse, json, os, sys
import numpy as np, joblib
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys import phys_metrics as pm

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", nargs="*", default=[])
ap.add_argument("--physgt", action="store_true")
ap.add_argument("--out", default="data/humanml3d_phys/paper_metrics.json")
args = ap.parse_args()
out = {}

if args.physgt:
    d = joblib.load("data/humanml3d_phys/hml_phys_test.pkl")
    bp = [b for b in d["body_pos"] if len(b) >= 8]
    out["phys_gt"] = dict(pm.all_metrics_raw(bp), n=len(bp), duration=1.0)
    print("phys_gt", json.dumps(out["phys_gt"]))

for p in args.rollouts:
    e = joblib.load(p)["episodes"]
    keep = [x for x in e if len(x["body_pos"]) >= 8]
    ok = [x["body_pos"] for x in keep if not x["fell"]]
    allp = [x["body_pos"] for x in keep]
    valid = np.array([len(x["body_pos"]) for x in e], float)
    ref = np.array([x["target_frames"] for x in e], float)
    name = os.path.basename(p).replace("rollouts_", "").replace(".pkl", "")
    out[name] = dict(non_fallen=pm.all_metrics_raw(ok), all_episodes=pm.all_metrics_raw(allp),
                     duration=pm.duration_frame_weighted(valid, ref),
                     never_fell=float(1 - np.mean([x["fell"] for x in e])),
                     n_non_fallen=len(ok), n=len(e))
    m = out[name]["non_fallen"]
    print("%-28s Float %.2f  Pen %.3f  Skate %.2f  Jerk %.3f | Duration %.3f  NeverFell %.3f  n %d/%d"
          % (name, m["floating_mm"], m["penetration_mm"], m["skating_mm"], m["jerk_mm_frame3"],
             out[name]["duration"], out[name]["never_fell"], len(ok), len(e)))
json.dump(out, open(args.out, "w"), indent=1)
print("saved", args.out)
