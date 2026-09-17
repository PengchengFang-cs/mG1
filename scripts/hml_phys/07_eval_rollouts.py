"""Step 4b: evaluate closed-loop rollouts (Isaac body_pos episodes) on the HumanML3D protocol.

rollouts pkl: dict(episodes=[{key, caption, body_pos [L,24,3], target_frames, fell, fall_step, gt_length}])
Each episode -> 22-joint 20 fps -> official 263-d -> evaluator with the caption it was generated for.
--fallen exclude (CLoSD: keep only episodes that reached their target length) | truncate (use frames before the fall)
Reports R-Precision/FID/MM-Dist/Diversity vs the kinematic GT of the same split, physics metrics on the
rollouts, and Duration (fraction of episodes that reached the target length).
"""
import argparse, json, os, sys, time
import numpy as np, joblib
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import HMLEvaluator, build_gt_items, format_summary, MIN_MOTION_LEN
from hml_phys.sim2hml import isaac_body_pos_to_hml263, isaac_body_pos_to_hml_joints

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--fallen", default="exclude", choices=["exclude", "truncate"])
ap.add_argument("--replications", type=int, default=20)
ap.add_argument("--drop_prefix_20fps", type=int, default=0, help="frames dropped at the start (CLoSD drops 16 sim frames = ~11 @20fps)")
ap.add_argument("--mm_min_reps", type=int, default=0, help=">0: compute MultiModality from captions having at least this many repeated episodes (mm_times=10)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

episodes = []
for p in args.rollouts:
    episodes += joblib.load(p)["episodes"]
n_total = len(episodes)
n_fell = sum(e["fell"] for e in episodes)
if args.fallen == "exclude":
    episodes = [e for e in episodes if not e["fell"]]
gt_items, _ = build_gt_items(args.split)
gt_len = {it["key"]: it["length"] for it in gt_items}
gen_items, too_short = [], 0
t0 = time.time()
for e in episodes:
    L_gt = gt_len.get(e["key"], e.get("gt_length", -1))
    if e["body_pos"].shape[0] < 4:
        too_short += 1; continue
    feat, _ = isaac_body_pos_to_hml263(e["body_pos"])
    raw_j = isaac_body_pos_to_hml_joints(e["body_pos"])[args.drop_prefix_20fps:]  # y-up, true sim floor
    feat = feat[args.drop_prefix_20fps:]
    L = min(len(feat), L_gt)
    if L < MIN_MOTION_LEN and args.fallen == "truncate":
        too_short += 1; continue
    if L < 4:
        too_short += 1; continue
    if not np.isfinite(feat[:L]).all():
        too_short += 1; continue
    gen_items.append(dict(key=e["key"], base=e["key"], motion=feat[:L], length=L, rep=int(e.get("rep", 0)), raw_joints=raw_j[:L],
                          texts=[dict(caption=e["caption"], tokens=e.get("tokens", "").split(" ") if e.get("tokens") else None)]))
# tokens: rollouts may not carry POS tokens -> take them from the GT item texts by caption
cap2tok = {}
for it in gt_items:
    for t in it["texts"]:
        cap2tok[(it["key"], t["caption"])] = t["tokens"]
missing_tok = 0
for g in gen_items:
    if g["texts"][0]["tokens"] is None:
        tok = cap2tok.get((g["key"], g["texts"][0]["caption"]))
        if tok is None:
            missing_tok += 1
        g["texts"][0]["tokens"] = tok or ["unk/OTHER"]
print(f"episodes {n_total}, fell {n_fell} ({100*n_fell/max(1,n_total):.1f}%), evaluated {len(gen_items)}, "
      f"too short {too_short}, missing tokens {missing_tok}, convert {time.time()-t0:.0f}s")
mm_items = None
if args.mm_min_reps > 0:
    groups = {}
    for g in gen_items:
        groups.setdefault((g["key"], g["texts"][0]["caption"]), []).append(g)
    mm_items = [dict(texts=v[0]["texts"], motions=[x["motion"] for x in v], lengths=[x["length"] for x in v])
                for v in groups.values() if len(v) >= args.mm_min_reps]
    print(f"MultiModality over {len(mm_items)} captions with >= {args.mm_min_reps} episodes")
ev = HMLEvaluator("cuda")
summary, per_rep = ev.evaluate(gt_items, gen_items, replications=args.replications, physics=True, gen_official_crop=False,
                               mm_items=mm_items if mm_items else None)
summary["physics_gen_raw_sim_floor"] = ev.physics_raw([g["raw_joints"] for g in gen_items])
summary["duration_completion"] = 1.0 - n_fell / max(1, n_total)
summary["n_episodes"] = n_total; summary["n_fell"] = n_fell; summary["n_evaluated"] = len(gen_items)
summary["fallen_policy"] = args.fallen
print(format_summary(summary)); print("duration (reached target):", f"{summary['duration_completion']:.4f}"); print("physics (raw sim floor):", summary["physics_gen_raw_sim_floor"])
json.dump(dict(summary=summary, per_rep=per_rep, args=vars(args)), open(args.out, "w"), indent=1)
print("saved", args.out)
