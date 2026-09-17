"""Evaluator sanity check (no simulation): HumanML3D test split kinematic GT.

1) Converter check: new_joints/<id>.npy -> our joints_to_hml263 -> compare with new_joint_vecs/<id>.npy.
2) Evaluator check: GT-vs-GT with the official protocol; expected (Guo et al. 2022, test split)
   R@1 0.511, R@2 0.703, R@3 0.797, FID 0.002, MM-Dist 2.974, Diversity 9.503.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import HMLEvaluator, build_gt_items, format_summary, HML_ROOT
from hml_phys.sim2hml import joints_to_hml263, hml263_to_joints

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--replications", type=int, default=20)
ap.add_argument("--n_convert_check", type=int, default=0, help="legacy check against the local new_joints/ directory (that directory is NOT consistent with new_joint_vecs/; keep 0)")
ap.add_argument("--out", default="data/humanml3d_phys/eval_gt_kinematic.json")
ap.add_argument("--gt_via_converter", action="store_true", help="rebuild GT 263-d from new_joints through our converter (converter validation)")
ap.add_argument("--joints_source", default="recover", choices=["recover", "new_joints"], help="recover: joints = recover_from_ric(new_joint_vecs) (official positions); new_joints: local new_joints files")
args = ap.parse_args()

t0 = time.time()
items, dropped = build_gt_items(args.split)
print(f"{args.split}: {len(items)} items ({dropped} ids dropped by length filter) in {time.time()-t0:.0f}s")

# 1) converter check on the first N base ids
errs = []
seen = set()
for it in items:
    b = it["base"]
    if b in seen or "_" in it["key"] or not os.path.exists(os.path.join(HML_ROOT, "new_joints", b + ".npy")):
        continue
    seen.add(b)
    j = np.load(os.path.join(HML_ROOT, "new_joints", b + ".npy"))
    ref = np.load(os.path.join(HML_ROOT, "new_joint_vecs", b + ".npy"))
    ours, _ = joints_to_hml263(j)
    n = min(len(ours), len(ref))
    errs.append(float(np.abs(ours[:n] - ref[:n]).max()))
    if len(seen) >= args.n_convert_check:
        break
if errs:
    print(f"converter check on {len(errs)} clips: max abs err {max(errs):.2e}, median {np.median(errs):.2e}")

# 2) GT-vs-GT
ev = HMLEvaluator("cuda")
gen_items = None
if args.gt_via_converter:
    # generated set = GT joints pushed through our converter; should match GT-vs-GT numbers if the converter is right
    gen_items = []
    cache = {}
    n_missing = 0
    for it in items:
        b = it["base"]
        if b not in cache:
            pj = os.path.join(HML_ROOT, "new_joints", b + ".npy")
            if args.joints_source == "recover":
                src = hml263_to_joints(np.load(os.path.join(HML_ROOT, "new_joint_vecs", b + ".npy")))
                cache[b], _ = joints_to_hml263(src)
            elif not os.path.exists(pj):
                cache[b] = None; n_missing += 1
            else:
                cache[b], _ = joints_to_hml263(np.load(pj))
        conv = cache[b]
        if conv is None:
            continue
        L = min(len(conv), it["length"])
        gen_items.append(dict(key=it["key"], base=b, motion=conv[:L], length=L, texts=it["texts"]))
    print("built converter GT items", len(gen_items), "skipped (no new_joints file):", n_missing)
summary, per_rep = ev.evaluate(items, gen_items, replications=args.replications, physics=True, gen_official_crop=True)
print(format_summary(summary))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(dict(summary=summary, per_rep=per_rep, converter_max_err=(max(errs) if errs else None), n_items=len(items)),
          open(args.out, "w"), indent=1)
print("saved", args.out)
