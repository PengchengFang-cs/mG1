"""Per-dim normalisation statistics of the physics tokens from random training windows."""
import argparse, os, sys, numpy as np
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import PhysWindowDataset, TokenStats, ROOT
from hml_phys import tokens as tk
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--H", type=int, default=16); ap.add_argument("--F", type=int, default=32)
ap.add_argument("--out", default=os.path.join(ROOT, "token_stats.npz")); ap.add_argument("--whole_sequence", action="store_true"); args = ap.parse_args()
ds = PhysWindowDataset("train", H=args.H, F=args.F, train=False, whole_sequence=args.whole_sequence)
rng = np.random.RandomState(0); sel = rng.choice(len(ds), min(args.n, len(ds)), replace=False)
roots, bodies = [], []
for j, i in enumerate(sel):
    clip, s = map(int, ds.windows[i]); bp, d, rs, ac = ds.raw_window(clip, s, "normal")
    r, b = tk.window_tokens(bp, d, rs, ac); roots.append(r); bodies.append(b)
    if j % 5000 == 0: print(j, flush=True)
st = TokenStats.fit(roots, bodies)
np.savez(args.out, **st)
print("saved", args.out, "windows total", len(ds), "short clips skipped", ds.n_clips_short)
print("root std range", st["root_std"].min(), st["root_std"].max(), "body std range", st["body_std"].min(), st["body_std"].max())
