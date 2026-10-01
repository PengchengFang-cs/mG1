"""Build 20 s reference schedules from FRoM-W1's per-prompt clips, for the ADAPT Table-1 protocol.

FRoM-W1 is a two-stage method: generate a kinematic motion offline, then track it. The protocol switches the
prompt every 5-10 s over 20 s, so a schedule is a concatenation of per-prompt clips -- which is exactly how
ADAPT describes the two-stage baselines it compares against ("evaluated offline with full lookahead over the
motion schedule").

Two properties of the generated clips force explicit handling, both recorded in the output so the write-up
can state them:
  * 23 of 130 clips ran to the 1000-frame generation cap without the model emitting an end token, so they are
    truncated to --max_clip_s.
  * 37 of 130 are shorter than the shortest protocol segment (5 s), so they are looped to fill it.

Boundary alignment: consecutive clips are generated in their own frames, so each segment is yaw-rotated and
XY-translated to continue from the previous segment's last frame. Height is left alone -- the retarget already
put each clip on the floor. No blending is applied: the pose discontinuity at a switch is a real property of
the two-stage approach and smoothing it would flatter the baseline.

Axis convention: the retarget writes `root_trans_offset` permuted [2,0,1] after correcting the floor on the
pre-permutation axis 1, so in the output XY is horizontal and index 2 is height (z-up), matching our env.
"""
import argparse, glob, json, os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True, help="directory of retargeted per-prompt pkls")
ap.add_argument("--episodes", type=int, default=2048)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--switch_lo", type=float, default=5.0)
ap.add_argument("--switch_hi", type=float, default=10.0)
ap.add_argument("--max_clip_s", type=float, default=10.0, help="cap for clips that hit the generation limit")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

paths = sorted(glob.glob(os.path.join(args.clips, "*.pkl")),
               key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
paths = [p for p in paths if os.path.basename(p)[0].isdigit()]
clips = []
for p in paths:
    d = joblib.load(p)
    fps = float(d["fps"])
    cap = int(args.max_clip_s * fps)
    n = len(d["dof"])
    clips.append(dict(prompt=d.get("prompt", os.path.basename(p)), fps=fps,
                      dof=np.asarray(d["dof"], np.float32)[:cap],
                      pos=np.asarray(d["root_trans_offset"], np.float32)[:cap],
                      rot=np.asarray(d["root_rot"], np.float32)[:cap],
                      n_gen=n, capped=n > cap))
fps = clips[0]["fps"]
assert all(abs(c["fps"] - fps) < 1e-6 for c in clips), "clips disagree on fps"
print(f"[sched] {len(clips)} clips @ {fps:g} fps, {sum(c['capped'] for c in clips)} truncated at "
      f"{args.max_clip_s:g} s, {sum(len(c['dof']) < args.switch_lo * fps for c in clips)} shorter than "
      f"{args.switch_lo:g} s (will be looped)")


def yaw_of(q):
    """heading angle about the vertical axis from an XYZW quaternion"""
    return Rot.from_quat(q).as_euler("zyx")[0]


def take(c, n):
    """n frames from clip c, looping if it is shorter, with the loop seam root-aligned like a boundary"""
    dof, pos, rot = c["dof"], c["pos"], c["rot"]
    if len(dof) >= n:
        return dof[:n].copy(), pos[:n].copy(), rot[:n].copy()
    D, P, R = [dof], [pos], [rot]
    while sum(len(x) for x in D) < n:
        D.append(dof); P.append(pos); R.append(rot)
    D, P, R = np.concatenate(D)[:n], np.concatenate(P)[:n], np.concatenate(R)[:n]
    return D, P, R


rng = np.random.RandomState(args.seed)
total = int(round(args.seconds * fps))
episodes, meta = {}, []
for e in range(args.episodes):
    dofs, poss, rots, segs = [], [], [], []
    t = 0
    anchor_xy = np.zeros(2, np.float32)
    anchor_yaw = 0.0
    while t < total:
        k = int(rng.randint(len(clips)))
        seg = min(int(round(rng.uniform(args.switch_lo, args.switch_hi) * fps)), total - t)
        if seg < 2:
            break
        d, p, q = take(clips[k], seg)
        # rotate this segment about the vertical axis so its heading continues from the previous one,
        # then slide it so its first frame starts at the previous last frame's XY
        dyaw = anchor_yaw - yaw_of(q[0])
        Rz = Rot.from_euler("z", dyaw)
        _p_orig = p.copy()
        p_rel = p - p[0]
        p = np.asarray(Rz.apply(p_rel), np.float32)
        p[:, :2] += anchor_xy
        # a yaw rotation cannot change height, so the clip's own floor-corrected height is kept as generated
        p[:, 2] = np.asarray(_p_orig[:, 2], np.float32)
        q = (Rz * Rot.from_quat(q)).as_quat().astype(np.float32)
        dofs.append(d); poss.append(p); rots.append(q)
        segs.append(dict(start=t, end=t + seg, prompt=clips[k]["prompt"], clip=k,
                         looped=bool(len(clips[k]["dof"]) < seg)))
        anchor_xy = p[-1, :2].copy(); anchor_yaw = yaw_of(q[-1])
        t += seg
    name = f"fromw1_ep{e:05d}"
    episodes[name] = dict(dof=np.concatenate(dofs), root_trans_offset=np.concatenate(poss),
                          root_rot=np.concatenate(rots), fps=fps)
    meta.append(dict(name=name, segments=segs))

n = len(next(iter(episodes.values()))["dof"])
print(f"[sched] {len(episodes)} episodes x {n} frames ({n / fps:.1f} s), "
      f"{np.mean([len(m['segments']) for m in meta]):.2f} segments each, "
      f"{100 * np.mean([s['looped'] for m in meta for s in m['segments']]):.1f}% of segments looped")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
joblib.dump(episodes, args.out, compress=3)
json.dump(dict(meta=meta, args=vars(args), fps=fps,
               clip_prompts=[c["prompt"] for c in clips],
               clips_capped=[i for i, c in enumerate(clips) if c["capped"]]),
          open(os.path.splitext(args.out)[0] + "_meta.json", "w"))
print(f"[sched] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) and its _meta.json")
