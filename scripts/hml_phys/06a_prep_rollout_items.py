"""Step 4a: caption list for closed-loop rollouts on the HumanML3D test split.

One rollout per evaluator item (same items as build_gt_items: whole-clip items with all captions, sub-clip
items with their own caption). Caption choice: --caption first|random (seeded). target_frames = sim frames
(30 fps) needed to cover the GT length at 20 fps plus a small margin.
"""
import argparse, json, math, sys
import numpy as np
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import build_gt_items

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--caption", default="random", choices=["first", "random"])
ap.add_argument("--n_per_item", type=int, default=1, help=">1: repeat the SAME caption n times (rep index) for MultiModality")
ap.add_argument("--fixed_target", type=int, default=0, help=">0: run every episode for this many sim frames (CLoSD-style fixed length) instead of ceil(1.5*L)+margin; must be <= phc.env.episode_length-2 (2 warm-up steps)")
ap.add_argument("--margin_frames", type=int, default=6, help="extra 30-fps frames beyond the GT length")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/humanml3d_phys/rollout_items_test.json")
args = ap.parse_args()
rng = np.random.RandomState(args.seed)
items, dropped = build_gt_items(args.split)
out = []
for it in items:
    t = it["texts"][0] if args.caption == "first" else it["texts"][rng.randint(len(it["texts"]))]
    for k in range(args.n_per_item):
        target = args.fixed_target if args.fixed_target > 0 else int(math.ceil(it["length"] * 30 / 20)) + args.margin_frames
        out.append(dict(key=it["key"], base=it["base"], caption=t["caption"], tokens=" ".join(t["tokens"]),
                        gt_length=int(it["length"]), target_frames=target, rep=k))
json.dump(out, open(args.out, "w"))
print(f"{len(out)} rollout items from {len(items)} evaluator items ({dropped} ids dropped) -> {args.out}; "
      f"max target {max(o['target_frames'] for o in out)} frames")
