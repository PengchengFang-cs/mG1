"""Fit per-channel normalisation statistics for the G1 token (docs/04 line A).

The G1 token is 96-d and has nothing to do with the SMPL side's 435-d one:

    proprio 67 = root_lin_vel_b 3 | root_ang_vel_b 3 | projected_gravity_b 3 | joint_pos 29 | joint_vel 29
    action  29 = one PD target per joint, executed as  target = default_joint_pos + action_scale * action

The dataset ships `BABEL-AMASS-ROBOT-.../meanstd.pkl`, but that is a 57-d (mean, std) pair for the tracker's
REFERENCE motion, not for our 96 channels, so it cannot be reused -- hence this script.

Statistics are fitted on the frames we will actually train on: successful rollouts of the TRAIN split
(project CLAUDE.md §1: the G1 dataset has only train/val, so train trains and val evaluates).  Accumulation is
streaming (sum / sum of squares per channel) so the 2.4 GB pickles never have to be resident together.
"""
import argparse, json, os, sys

import joblib
import numpy as np

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")

G1_DIR = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/g1_rollouts"
PROPRIO_DIM, ACTION_DIM = 67, 29
TOKEN_DIM = PROPRIO_DIM + ACTION_DIM
# proprio channel layout, from the rollout summaries' `proprio_layout`
PROPRIO_SLICES = dict(root_lin_vel_b=(0, 3), root_ang_vel_b=(3, 6), projected_gravity_b=(6, 9),
                      joint_pos=(9, 38), joint_vel=(38, 67))

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", default=["train_all_x1.pkl", "dagger4_trainall_v4_mix05_randprompt.pkl"],
                help="TRAIN-split rollout pickles under data/g1_rollouts")
ap.add_argument("--success_only", type=int, default=1, help="fit on successful rollouts only (what we train on)")
ap.add_argument("--min_frames", type=int, default=40)
ap.add_argument("--eps", type=float, default=1e-4, help="floor on std, so dead channels do not blow up")
ap.add_argument("--out", default=os.path.join(G1_DIR, "g1_token_stats.npz"))
args = ap.parse_args()

n = 0
s1 = np.zeros(TOKEN_DIM, np.float64)
s2 = np.zeros(TOKEN_DIM, np.float64)
lo = np.full(TOKEN_DIM, np.inf)
hi = np.full(TOKEN_DIM, -np.inf)
per_file = []

for fn in args.files:
    p = os.path.join(G1_DIR, fn)
    if not os.path.exists(p):
        print(f"MISSING {p}"); continue
    print(f"reading {fn} ...", flush=True)
    obj = joblib.load(p)
    rolls = obj["rollouts"] if isinstance(obj, dict) else obj
    kept = 0
    for r in rolls:
        if args.success_only and not bool(r.get("success", False)):
            continue
        pro, act = r["proprio"], r["action"]
        if pro.shape[0] < args.min_frames or pro.shape[0] != act.shape[0]:
            continue
        assert pro.shape[1] == PROPRIO_DIM and act.shape[1] == ACTION_DIM, (pro.shape, act.shape)
        x = np.concatenate([pro, act], -1).astype(np.float64)
        if not np.isfinite(x).all():                       # a diverged rollout can carry inf/nan
            continue
        n += x.shape[0]; s1 += x.sum(0); s2 += (x * x).sum(0)
        lo = np.minimum(lo, x.min(0)); hi = np.maximum(hi, x.max(0))
        kept += 1
    per_file.append(dict(file=fn, n_rollouts=len(rolls), kept=kept))
    print(f"  kept {kept}/{len(rolls)} rollouts, running frames {n}", flush=True)
    del obj, rolls

assert n > 0, "no frames accumulated"
mean = (s1 / n).astype(np.float32)
var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
std = np.maximum(np.sqrt(var), args.eps).astype(np.float32)

print(f"\n{n} frames over {sum(f['kept'] for f in per_file)} rollouts")
print(f"{'channel group':<22}{'|mean| max':>12}{'std min':>10}{'std max':>10}")
groups = {**{k: v for k, v in PROPRIO_SLICES.items()}, "action": (PROPRIO_DIM, TOKEN_DIM)}
for name, (a, b) in groups.items():
    print(f"{name:<22}{np.abs(mean[a:b]).max():>12.4f}{std[a:b].min():>10.4f}{std[a:b].max():>10.4f}")
dead = int((std <= args.eps * 1.001).sum())
if dead:
    print(f"WARNING: {dead} channels have std <= eps (constant in the data); they normalise to 0")

np.savez(args.out, mean=mean, std=std, lo=lo.astype(np.float32), hi=hi.astype(np.float32),
         n_frames=np.int64(n), proprio_dim=np.int64(PROPRIO_DIM), action_dim=np.int64(ACTION_DIM),
         meta=json.dumps(dict(files=per_file, success_only=bool(args.success_only),
                              min_frames=args.min_frames, eps=args.eps,
                              proprio_slices=PROPRIO_SLICES)))
print(f"wrote {args.out}")
