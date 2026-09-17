"""Step 3: physics ground-truth check.

PULSE-tracked HumanML3D test clips (hml_phys_test.pkl) -> 22-joint 20 fps -> official 263-d -> evaluator,
using exactly the GT evaluation protocol (random caption, official length quantisation / crop). If the tracked
data is as good as the kinematic data, R-Precision/FID should be close to GT-vs-GT (R@1 ~0.51, FID ~0.00x).
--src_fps nominal: treat physics frames as 30 fps (what the simulator actually replayed; KIT/EKUT clips are
    11 % slower than real time); eff: use fps_eff (restores original timing).
Also reports the physics metrics of the tracked data (the Phys-GT reference row for Floating / Jerk / ...).
"""
import argparse, json, os, sys, time, collections
import numpy as np, joblib
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import HMLEvaluator, build_gt_items, format_summary, MIN_MOTION_LEN
from hml_phys.sim2hml import isaac_body_pos_to_hml263, isaac_body_pos_to_hml_joints

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="data/humanml3d_phys/hml_phys_test.pkl")
ap.add_argument("--split", default="test")
ap.add_argument("--src_fps", default="eff", choices=["nominal", "eff"])
ap.add_argument("--replications", type=int, default=20)
ap.add_argument("--out", default="data/humanml3d_phys/eval_physgt_test.json")
ap.add_argument("--max_align_err", type=float, default=0.0, help="debug: keep only clips whose file alignment error <= this (0 = all)")
ap.add_argument("--debug_use_gt_features", action="store_true", help="debug: substitute the official GT 263-d features (same item logic) to isolate the converter")
ap.add_argument("--debug_block_from_gt", default="", help="debug: comma list of blocks {root,ric,rot,vel,contact} copied from GT features (whole-clip items with matching length only)")
args = ap.parse_args()

t0 = time.time()
d = joblib.load(args.pkl)
phys = {n: i for i, n in enumerate(d["name"])}
gt_items, _ = build_gt_items(args.split)
feat_cache = {}; raw_joint_cache = {}
def feats(name):
    if name not in feat_cache:
        i = phys[name]
        fps = 30.0 if args.src_fps == "nominal" else float(d["fps_eff"][i])
        f, _ = isaac_body_pos_to_hml263(d["body_pos"][i], src_fps=fps)
        feat_cache[name] = f
        raw_joint_cache[name] = isaac_body_pos_to_hml_joints(d["body_pos"][i], src_fps=fps)
    return feat_cache[name]
BLOCKS = {"root": (0, 4), "ric": (4, 67), "rot": (67, 193), "vel": (193, 259), "contact": (259, 263)}
HML = "/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D"
gen_items, stats = [], collections.Counter()
for it in gt_items:
    b = it["base"]
    if b not in phys:
        stats["no_physics_clip"] += 1; continue
    if args.max_align_err > 0 and not (d["file_align_err_m"][phys[b]] <= args.max_align_err):
        stats["align_err_filtered"] += 1; continue
    f = feats(b)
    if args.debug_use_gt_features:
        f = np.load(os.path.join(HML, "new_joint_vecs", b + ".npy")).astype(np.float32)
    elif args.debug_block_from_gt:
        g = np.load(os.path.join(HML, "new_joint_vecs", b + ".npy")).astype(np.float32)
        n = min(len(f), len(g)); f = f[:n].copy()
        for blk in args.debug_block_from_gt.split(","):
            a, c = BLOCKS[blk]; f[:, a:c] = g[:n, a:c]
    if "_" in it["key"]:  # sub-clip item: locate its caption's time tags
        # the evaluator item stores only caption/tokens; recover tags from the physics record's texts
        tags = [(t["f_tag"], t["to_tag"]) for t in d["texts"][phys[b]] if t["caption"] == it["texts"][0]["caption"]]
        if not tags:
            stats["subclip_tag_missing"] += 1; continue
        f0, t1 = tags[0]
        stretch = 1.0 if args.src_fps == "eff" else float(d["time_stretch"][phys[b]])  # nominal: features are on a slowed timeline
        f = f[int(f0 * 20 * stretch): int(t1 * 20 * stretch)]
    L = min(len(f), it["length"])
    if L < MIN_MOTION_LEN:
        stats["too_short"] += 1; continue
    if abs(len(f) - it["length"]) > 0.15 * it["length"]:
        stats["length_mismatch_gt15pct"] += 1
    if not np.isfinite(f[:L]).all():
        stats["non_finite_features"] += 1; continue
    gen_items.append(dict(key=it["key"], base=b, motion=f[:L], length=L, texts=it["texts"],
                          raw_joints=raw_joint_cache[b][:L] if "_" not in it["key"] else None))
    stats["kept"] += 1
print(f"phys-GT items: {dict(stats)} of {len(gt_items)} evaluator items, convert {time.time()-t0:.0f}s")
ev = HMLEvaluator("cuda")
summary, per_rep = ev.evaluate(gt_items, gen_items, replications=args.replications, physics=True, gen_official_crop=True)
# kinematic GT restricted to the same clips (subset-matched reference row)
sub_ids = sorted({g["base"] for g in gen_items})
gt_sub, _ = build_gt_items(args.split, id_list=sub_ids)
sub_summary, _ = ev.evaluate(gt_sub, None, replications=min(args.replications, 5), physics=False)
summary["kinematic_gt_same_subset"] = {k: v for k, v in sub_summary.items() if isinstance(v, dict) and "mean" in v}
# selection-effect decomposition: the kinematic GT of the same clips, evaluated as if generated, against the full GT
sub_as_gen = [dict(it, key=it["key"], base=it["base"]) for it in gt_sub]
dec_summary, _ = ev.evaluate(gt_items, sub_as_gen, replications=min(args.replications, 5), physics=False, gen_official_crop=True)
summary["kinematic_gt_same_subset_vs_full_gt"] = {k: v for k, v in dec_summary.items() if isinstance(v, dict) and "mean" in v and not k.startswith("real_")}
summary["physics_gen_raw_sim_floor"] = ev.physics_raw([g["raw_joints"] for g in gen_items if g["raw_joints"] is not None])
summary["coverage"] = dict(stats); summary["src_fps"] = args.src_fps
print(format_summary(summary))
json.dump(dict(summary=summary, per_rep=per_rep, args=vars(args)), open(args.out, "w"), indent=1)
print("saved", args.out)
