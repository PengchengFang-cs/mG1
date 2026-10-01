"""Merge the per-GPU retarget shards into one motion library and sanity-check it.

The shards are disjoint by construction (`seqs[shard::nshards]` off a pool both shards sampled with the
same seed), so the union is exactly the requested sample. The checks below are the ones that would have
caught the mistakes this pipeline is prone to: a frame-rate that never got resampled, a root height
left in the deployment frame, a dof width that does not match the policy, and clips whose fit diverged.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-dof", type=int, default=21)
    ap.add_argument("--expect-fps", type=int, default=30)
    args = ap.parse_args()

    merged, per_shard = {}, []
    for s in args.shards:
        d = joblib.load(s)
        dup = set(d) & set(merged)
        assert not dup, f"shards overlap on {len(dup)} keys, e.g. {sorted(dup)[:3]}"
        merged.update(d)
        per_shard.append((Path(s).name, len(d)))

    print("=== shards ===")
    for name, n in per_shard:
        print(f"  {name}: {n}")
    print(f"  merged: {len(merged)}")

    n_frames, errs, heights, durations = [], [], [], []
    bad = []
    for k, v in merged.items():
        dof, root = v["dof"], v["root_trans_offset"]
        if dof.shape[1] != args.expect_dof:
            bad.append((k, f"dof width {dof.shape[1]} != {args.expect_dof}"))
        if v["fps"] != args.expect_fps:
            bad.append((k, f"fps {v['fps']} != {args.expect_fps}"))
        if dof.shape[0] != root.shape[0] or dof.shape[0] != v["pose_aa"].shape[0]:
            bad.append((k, f"length mismatch dof {dof.shape[0]} root {root.shape[0]} pose {v['pose_aa'].shape[0]}"))
        n_frames.append(dof.shape[0])
        errs.append(v["fit_err_m"])
        heights.append(float(root[:, 2].min()))
        durations.append(dof.shape[0] / v["fps"])

    n_frames, errs, heights, durations = map(np.array, (n_frames, errs, heights, durations))
    print("\n=== library ===")
    print(f"  clips            : {len(merged)}")
    print(f"  frames           : {n_frames.sum()}  ({durations.sum() / 60:.1f} min at {args.expect_fps} fps)")
    print(f"  clip length (s)  : min {durations.min():.1f}  median {np.median(durations):.1f}  max {durations.max():.1f}")
    print(f"  fit error (m)    : mean {errs.mean():.4f}  median {np.median(errs):.4f}  p95 {np.percentile(errs, 95):.4f}  max {errs.max():.4f}")
    # Grounding puts the lowest body point at 0.08 m, so the root's own minimum height should sit
    # comfortably above zero and well under standing height. A value near 0 or negative would mean the
    # deployment-frame axis permutation leaked in.
    print(f"  root min z (m)   : min {heights.min():.3f}  median {np.median(heights):.3f}  max {heights.max():.3f}")
    sample = merged[next(iter(merged))]
    print(f"  keys per clip    : {sorted(sample)}")
    print(f"  dof names        : {len(sample['dof_names'])} -> {sample['dof_names'][:3]} ... {sample['dof_names'][-2:]}")
    print(f"  body names       : {len(sample['body_names'])}")

    if bad:
        print(f"\n!!! {len(bad)} clips failed structural checks:")
        for k, why in bad[:10]:
            print(f"  {k}: {why}")
        for why, n in Counter(w for _, w in bad).most_common():
            print(f"  {n:5d}  {why}")
        raise SystemExit(1)
    print("\nstructural checks: all clips OK")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    (out.with_suffix(".summary.json")).write_text(json.dumps({
        "n_clips": len(merged),
        "n_frames": int(n_frames.sum()),
        "minutes": float(durations.sum() / 60),
        "fit_err_m": {"mean": float(errs.mean()), "median": float(np.median(errs)),
                      "p95": float(np.percentile(errs, 95)), "max": float(errs.max())},
        "clip_seconds": {"min": float(durations.min()), "median": float(np.median(durations)),
                         "max": float(durations.max())},
        "shards": dict(per_shard),
    }, indent=2))


if __name__ == "__main__":
    main()
