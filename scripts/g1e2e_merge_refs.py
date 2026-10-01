"""Merge the per-GPU reference shards per split, then verify the world frame before anything consumes them.

`check_upright` is the gate: it reads the library back through the env's OWN forward kinematics and config,
which is the only way a wrong root convention shows up -- the fit error cannot see it, because that error is
computed against a SMPL target rotated by the same amount (hml_phys/g1_retarget.py).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    d = Path(args.out_dir).resolve()

    from hml_phys.g1_retarget import check_upright

    summary = {}
    for split in args.splits.split(","):
        shards = sorted(d.glob(f"refs_{split}_shard*.pkl"))
        assert shards, f"no shards for {split} in {d}"
        lib, text = {}, {}
        for s in shards:
            part = joblib.load(s)
            dup = set(part) & set(lib)
            assert not dup, f"{s.name} overlaps on {len(dup)} clips"
            lib.update(part)
            text.update(json.loads(s.with_suffix(".text.json").read_text()))
            print(f"  {s.name}: {len(part)}")
        assert set(lib) == set(text), "library and caption index disagree on clip ids"

        stats = check_upright(lib)
        errs = np.array([v["fit_err_m"] for v in lib.values()])
        frames = np.array([v["dof"].shape[0] for v in lib.values()])
        caps = sum(len(v["captions"]) for v in text.values())
        print(f"=== {split}: {len(lib)} clips, {frames.sum()} frames "
              f"({frames.sum() / 30 / 60:.1f} min), {caps} captions")
        print(f"    fit err (m): mean {errs.mean():.4f} median {np.median(errs):.4f} p95 {np.percentile(errs, 95):.4f}")
        print(f"    world frame: head-above-pelvis {stats['head_above_pelvis_median']:.3f} m, "
              f"upright {stats['frac_upright']:.3f}, lowest foot {stats['lowest_foot_median']:.3f} m  OK")

        joblib.dump(lib, d / f"refs_{split}.pkl")
        (d / f"refs_{split}.text.json").write_text(json.dumps(text, ensure_ascii=False, indent=1))
        summary[split] = dict(clips=len(lib), frames=int(frames.sum()),
                              minutes=float(frames.sum() / 30 / 60), captions=caps,
                              fit_err_m=float(errs.mean()), **stats)

    (d / "refs_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote refs_{{{args.splits}}}.pkl + .text.json + refs_summary.json in {d}")


if __name__ == "__main__":
    main()
