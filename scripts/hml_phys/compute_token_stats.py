"""Per-dim normalisation statistics for the physics tokens and for the local (velocity) root.

Windows are built exactly as the training dataset builds them (v3: sparse distant + dense recent history,
canonicalised on the newest history frame), so the statistics match the distribution the model actually
sees, including the non-contiguous rows of the sparse history.

Writes an npz with root_mean/root_std/body_mean/body_std and local_root_mean/local_root_std.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import PhysWindowDataset, TokenStats, ROOT
from hml_phys import tokens as tk

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--F", type=int, default=32)
ap.add_argument("--H_sparse", type=int, default=16); ap.add_argument("--L_max", type=int, default=154)
ap.add_argument("--alpha", type=float, default=3.0)
ap.add_argument("--whole_sequence", action="store_true")
ap.add_argument("--out", default=os.path.join(ROOT, "token_stats_v3.npz"))
ap.add_argument("--eps", type=float, default=1e-5)
args = ap.parse_args()

ds = PhysWindowDataset("train", H=args.H, F=args.F, train=False, whole_sequence=args.whole_sequence,
                       H_sparse=args.H_sparse, L_max=args.L_max, alpha=args.alpha, randomize_history=False)
rng = np.random.RandomState(0)
sel = rng.choice(len(ds), min(args.n, len(ds)), replace=False)
roots, bodies, locals_ = [], [], []
for j, i in enumerate(sel):
    clip, s = map(int, ds.windows[i])
    wrng = np.random.RandomState(int(i))
    bp, dof, rs, ac, rows, fidx, n_hist, n_used = ds.raw_window(clip, s, "normal", args.H_sparse, args.alpha, wrng)
    origin = int(rows[n_hist - 1])
    r_full, b_full = tk.window_tokens(bp, dof, rs, ac, origin=origin)
    r, b = r_full[rows], b_full[rows]
    roots.append(r); bodies.append(b)
    locals_.append(tk.root_to_local_root(r, frame_index=fidx))
    if j % 5000 == 0:
        print(j, flush=True)
st = TokenStats.fit(roots, bodies, eps=args.eps)
lr = np.concatenate(locals_, 0).astype(np.float64)
st["local_root_mean"] = lr.mean(0).astype(np.float32)
st["local_root_std"] = np.sqrt(lr.var(0) + args.eps).astype(np.float32)
np.savez(args.out, **st)
print("saved", args.out, "| windows total", len(ds), "| short clips skipped", ds.n_clips_short)
print("root std ", np.round(st["root_std"], 3))
print("local root mean", np.round(st["local_root_mean"], 3), "std", np.round(st["local_root_std"], 3))
print("body std range", st["body_std"].min(), st["body_std"].max())
