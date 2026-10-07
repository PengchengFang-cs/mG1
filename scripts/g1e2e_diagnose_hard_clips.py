"""Why do the clips BOTH the teacher and our policy fail on fail?

Measured on 2026-10-07 (STATUS.md §5.8): of 512 clips, 33 fail for us and 30 for the teacher, but only
14 fail for both -- and on those 14 we survive 20.7% of the clip against the teacher's 20.3%. A clip the
teacher cannot do either has no correction to offer, so no amount of residual RL or DAgger closes it:
the reward would be pointing at a reference the robot cannot realise. That makes those 14 a REFERENCE
problem, and this script asks which property of the reference separates them from the 463 both-succeed
clips.

Everything here comes from the reference library itself plus the two per-clip outcome dumps -- no
simulation, no FK. The properties are the ones the reference-layer failures in
arXiv 2610.03196 point at (infeasible references, flight phases, extreme speed) expressed in quantities
the library already carries:

    fit_err_m       how well the 21-DoF fit reproduced the SMPL target at all
    root speed      horizontal speed of the retargeted root, m/s
    root height     mean and min of the root's z, against the 0.08 m grounding offset
    vertical speed  |dz/dt| of the root -- jumps and drops
    dof range       per-clip peak |dof| against the joint limits, i.e. how close to the stops
    dof rate        peak |d dof/dt|, how fast the reference asks the joints to move
    duration

Usage (on a compute node):
    python scripts/g1e2e_diagnose_hard_clips.py \
      --refs data/g1_e2e/refs_train_part1.pkl \
      --ours outputs/g1e2e/eval_G_cfgA1.0.bodypos.npz \
      --teacher outputs/g1e2e/teacher_part1_keys.bodypos.npz \
      --out outputs/g1e2e/hard_clips.json
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def clip_props(v):
    """Scalar properties of one retargeted reference, all from the stored arrays."""
    fps = float(v["fps"])
    t = v["root_trans_offset"].astype(np.float64)          # [T,3] world, z up, grounded at 0.08
    dof = v["dof_picked"].astype(np.float64)               # [T,21] actuated joint angles
    n = t.shape[0]
    dt = 1.0 / fps
    vel = np.diff(t, axis=0) / dt if n > 1 else np.zeros((1, 3))
    drate = np.diff(dof, axis=0) / dt if n > 1 else np.zeros((1, dof.shape[1]))
    return dict(
        n_frames=int(n),
        seconds=float(n / fps),
        fps=fps,
        fit_err_m=float(v["fit_err_m"]),
        root_speed_xy=float(np.linalg.norm(vel[:, :2], axis=-1).mean()),
        root_speed_xy_max=float(np.linalg.norm(vel[:, :2], axis=-1).max()),
        root_z_mean=float(t[:, 2].mean()),
        root_z_min=float(t[:, 2].min()),
        root_z_max=float(t[:, 2].max()),
        root_vz_absmax=float(np.abs(vel[:, 2]).max()),
        dof_absmax=float(np.abs(dof).max()),
        dof_rate_absmax=float(np.abs(drate).max()),
        dof_rate_p95=float(np.percentile(np.abs(drate), 95)),
    )


def summarise(name, props, fields):
    if not props:
        return {k: None for k in fields} | {"n": 0}
    out = {"n": len(props)}
    for f in fields:
        a = np.array([p[f] for p in props], dtype=np.float64)
        out[f] = dict(mean=float(a.mean()), median=float(np.median(a)),
                      p90=float(np.percentile(a, 90)), max=float(a.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--ours", required=True, help="our policy's .bodypos.npz")
    ap.add_argument("--teacher", required=True, help="the teacher's .bodypos.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    o = np.load(args.ours)
    t = np.load(args.teacher)
    ko = [str(k) for k in o["keys"]]
    kt = [str(k) for k in t["keys"]]
    assert ko == kt, "the two runs scored different clips, or in a different order"

    fo = o["fall_step"].astype(np.float64) < o["horizon"].astype(np.float64)
    ft = t["fall_step"].astype(np.float64) < t["horizon"].astype(np.float64)
    ours_frac = o["fall_step"].astype(np.float64) / o["horizon"].astype(np.float64)
    teach_frac = t["fall_step"].astype(np.float64) / t["horizon"].astype(np.float64)

    group = {}
    for i, k in enumerate(ko):
        if fo[i] and ft[i]:
            group[k] = "both_fail"
        elif fo[i]:
            group[k] = "ours_only"
        elif ft[i]:
            group[k] = "teacher_only"
        else:
            group[k] = "both_ok"

    lib = joblib.load(args.refs)
    missing = [k for k in ko if k not in lib]
    assert not missing, f"{len(missing)} scored clips are absent from the library, e.g. {missing[:3]}"

    fields = ["seconds", "fit_err_m", "root_speed_xy", "root_speed_xy_max", "root_z_mean",
              "root_z_min", "root_vz_absmax", "dof_absmax", "dof_rate_absmax", "dof_rate_p95"]
    buckets = {g: [] for g in ("both_fail", "ours_only", "teacher_only", "both_ok")}
    per_clip = {}
    for i, k in enumerate(ko):
        p = clip_props(lib[k])
        p["group"] = group[k]
        p["ours_survived_frac"] = float(ours_frac[i])
        p["teacher_survived_frac"] = float(teach_frac[i])
        per_clip[k] = p
        buckets[group[k]].append(p)

    res = {g: summarise(g, b, fields) for g, b in buckets.items()}

    print(f"{'group':<14}{'n':>5}", end="")
    for f in fields:
        print(f"{f[:13]:>15}", end="")
    print()
    for g in ("both_ok", "teacher_only", "ours_only", "both_fail"):
        s = res[g]
        print(f"{g:<14}{s['n']:>5}", end="")
        for f in fields:
            v = s[f]["median"] if s["n"] else float("nan")
            print(f"{v:>15.4f}", end="")
        print()
    print("\n(medians; the full mean/median/p90/max per group is in the JSON)")

    # Which single property separates both_fail from both_ok most strongly, by median ratio.
    if res["both_fail"]["n"] and res["both_ok"]["n"]:
        print("\nmedian(both_fail) / median(both_ok):")
        rows = []
        for f in fields:
            a = res["both_fail"][f]["median"]
            b = res["both_ok"][f]["median"]
            if abs(b) > 1e-9:
                rows.append((abs(a / b), f, a, b))
        for r, f, a, b in sorted(rows, reverse=True):
            flag = "  <-- separates" if r > 1.5 or r < 0.67 else ""
            print(f"  {f:<20} {a:>10.4f} / {b:>10.4f} = {r:>6.2f}x{flag}")

    both_fail_keys = sorted(k for k, g in group.items() if g == "both_fail")
    ours_only_keys = sorted(k for k, g in group.items() if g == "ours_only")
    print(f"\nboth_fail ({len(both_fail_keys)}): {both_fail_keys}")
    print(f"ours_only ({len(ours_only_keys)}): {ours_only_keys}")

    Path(args.out).write_text(json.dumps(
        dict(groups=res, both_fail=both_fail_keys, ours_only=ours_only_keys,
             teacher_only=sorted(k for k, g in group.items() if g == "teacher_only"),
             per_clip=per_clip), indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
