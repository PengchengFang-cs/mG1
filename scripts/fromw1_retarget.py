"""Retarget H-GPT's 623-d motions onto the G1, body only.

Not using `H-ACT/retarget/main.py`: it calls `retarget_from_rotvec` unconditionally, which needs the MANO
hand models, and its `SMPLX_OUTPUT_PATH` is an empty string so the script cannot run as shipped. The G1 here
is the 29-DoF version whose end effectors are the wrist links -- there are no dexterous hands to retarget --
so only `body_retarget.process_data` is needed.

The retarget module resolves its assets relatively ("assets/meta/mean.npy", "models/smpl", the robot MJCF),
so this chdirs into it; the assets themselves are symlinks into external/fromw1_data and UniPhys/data/smpl.
"""
import argparse, glob, os, sys, time

import joblib
import numpy as np
import torch

R = "/iridisfs/scratch/pf2m24/projects/motion_rebot"
RT = f"{R}/external/FRoM-W1/H-ACT/retarget"

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True, help="an H-GPT samples_* directory")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--robot", default="G1")
args = ap.parse_args()
# resolve before the chdir below, or a relative --samples/--out silently resolves inside the retarget tree
args.samples = os.path.abspath(args.samples)
args.out = os.path.abspath(args.out)

sys.path.insert(0, RT)
os.chdir(RT)
from body_retarget import load_amass_data, process_data          # noqa: E402
from utils import feats2joints, pos2smpl                          # noqa: E402

feats = sorted(glob.glob(os.path.join(args.samples, "*_feats_out.npy")),
               key=lambda p: int(os.path.basename(p).split("_")[0]))
if args.limit:
    feats = feats[: args.limit]
assert feats, f"no *_feats_out.npy under {args.samples}"
os.makedirs(args.out, exist_ok=True)
tmp = os.path.join(args.out, "_smplx_tmp.npz")
print(f"[rt] {len(feats)} clips -> {args.out}", flush=True)

done, t0 = 0, time.time()
for p in feats:
    i = int(os.path.basename(p).split("_")[0])
    dst = os.path.join(args.out, f"{i}.pkl")
    if os.path.exists(dst):
        done += 1; continue
    x = np.load(p)
    assert x.ndim == 3 and x.shape[-1] == 623, f"{p}: expected [1,T,623], got {x.shape}"
    joints = feats2joints(torch.from_numpy(x[0]))                 # [T, 52, 3]
    # pos2smpl's docstring says np.ndarray but it calls torch.zeros_like on the input, so keep it a tensor
    np.savez(tmp, **pos2smpl(joints))
    robot_data = process_data(load_amass_data(tmp), robot=args.robot)
    # carry the prompt through so the downstream protocol knows what this clip was asked to be
    txt = os.path.join(args.samples, f"{i}_text_in.txt")
    if os.path.exists(txt):
        robot_data["prompt"] = open(txt).read().strip()
    joblib.dump(robot_data, dst)
    done += 1
    print(f"[rt] {done}/{len(feats)}  clip {i}  T={x.shape[1]}  {time.time()-t0:.0f}s", flush=True)

k = joblib.load(os.path.join(args.out, f"{int(os.path.basename(feats[0]).split('_')[0])}.pkl"))
print("[rt] output keys:", {kk: (np.shape(vv) if hasattr(vv, '__len__') else vv) for kk, vv in k.items()})
print(f"[rt] done {done} clips in {time.time()-t0:.0f}s")
