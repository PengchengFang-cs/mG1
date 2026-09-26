"""Score CLoSD's released motions with OUR evaluator under OUR protocol (docs/06 §2.2b).

CLoSD publishes its generated motions already in the 263-d HumanML3D feature space
(huggingface guytevet/CLoSD :: evaluation/closd/CloSD.pkl), so no conversion is needed and their
simulator never has to be installed. Their own eval.log used `--do_unique`, which changes the
retrieval pool and puts their reported numbers (their GT R@1 0.4616 / MM-Dist 3.2408) off the
canonical Guo scale that we reproduce (0.515 / 2.972 vs Guo's 0.511 / 2.974). Re-scoring their
motions here is the only way to get a CLoSD number on our scale.

ONE pass, ONE metric computation (project CLAUDE.md §4).
"""
import argparse, json, pickle, sys, time
import numpy as np
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import HMLEvaluator, build_gt_items, format_summary

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="external/closd/CloSD.pkl")
ap.add_argument("--split", default="test")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="data/humanml3d_phys/eval_closd_external.json")
ap.add_argument("--denorm", type=int, default=1,
                help="CLoSD stores motions ALREADY normalised by the HumanML3D mean/std (measured: per-channel "
                     "std 0.957 vs 0.177 for raw features). Our evaluator normalises internally, so they must be "
                     "de-normalised first or everything is normalised twice (that gave FID 54.4).")
args = ap.parse_args()
assert args.split != "val", "the val split is banned (CLAUDE.md §1)"

d = pickle.load(open(args.pkl, "rb"))
mot, cap, ln, tok, key = d["motion"], d["caption"], d["length"], d["tokens"], d["db_key"]
mot = mot.numpy() if hasattr(mot, "numpy") else np.asarray(mot)
gen_items = []
for i in range(len(cap)):
    L = int(ln[i])
    gen_items.append(dict(key=str(key[i]), length=L, motion=mot[i, :L].astype(np.float32),
                          texts=[dict(caption=cap[i], tokens=tok[i].split("_"))]))
print(f"CLoSD: {len(gen_items)} motions, lengths {int(ln.min())}-{int(ln.max())}, feature dim {mot.shape[-1]}")

gt_items, dropped = build_gt_items(args.split)   # returns (items, n_dropped)
print(f"our GT items ({args.split}): {len(gt_items)} (dropped {dropped} by the official length filter)")
ev = HMLEvaluator(args.device)
if args.denorm:
    for g in gen_items:
        g["motion"] = (g["motion"] * ev.std + ev.mean).astype(np.float32)
    chk = np.concatenate([g["motion"] for g in gen_items[:200]])
    ref = np.concatenate([gt_items[i]["motion"] for i in range(200)])
    print(f"de-normalised: per-channel std median {np.median(chk.std(0)):.4f} (our GT {np.median(ref.std(0)):.4f})")
t0 = time.time()
summary, per_rep = ev.evaluate(gt_items, gen_items, replications=1, gen_official_crop=False, physics=False)
summary["n_motions"] = len(gen_items)
summary["source"] = args.pkl
summary["protocol"] = ("our standard Guo-evaluator protocol, single pass, single metric computation; "
                       "CLoSD's own eval used --do_unique and is NOT on this scale")
print(format_summary(summary))
json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summary.items()},
          open(args.out, "w"), indent=1, default=float)
print(f"wrote {args.out} in {time.time()-t0:.0f}s")
