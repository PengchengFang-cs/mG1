"""Merge the per-GPU reference shards per split, then verify the world frame before anything consumes them.

`check_upright` is the gate: it reads the library back through the env's OWN forward kinematics and config,
which is the only way a wrong root convention shows up -- the fit error cannot see it, because that error is
computed against a SMPL target rotated by the same amount (hml_phys/g1_retarget.py).
"""
import argparse
import json
import re
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
    ap.add_argument("--shards", default="",
                    help="merge only these shard indices, e.g. '0-9' or '0,1,2'. Lets the finished half "
                         "of a two-card build be merged and recorded while the other half is still "
                         "running -- recording is per-clip independent, so there is no reason to idle a "
                         "free card. Default: every shard present.")
    ap.add_argument("--name", default="",
                    help="output stem instead of refs_<split>, e.g. 'refs_train_part1'. Required with "
                         "--shards so a partial merge cannot overwrite the full library.")
    args = ap.parse_args()
    want = None
    if args.shards:
        want = set()
        for piece in args.shards.split(","):
            if "-" in piece:
                lo, hi = piece.split("-")
                want.update(range(int(lo), int(hi) + 1))
            else:
                want.add(int(piece))
        assert args.name, "--shards needs --name, so a partial merge cannot overwrite refs_<split>.pkl"
    d = Path(args.out_dir).resolve()

    from hml_phys.g1_retarget import check_upright

    summary = {}
    for split in args.splits.split(","):
        # The builder writes siblings next to each shard: `<stem>.skipped.pkl` and, while a shard is
        # still running or after it was killed, `<stem>.part.pkl` (the resume checkpoint). A bare *.pkl
        # glob picks both up; the .part one would then be merged with its keys "lib"/"text"/"skipped"
        # treated as clip ids, which is the one path that can silently corrupt the merged library.
        shards = sorted(q for q in d.glob(f"refs_{split}_shard*.pkl")
                        if not q.name.endswith((".skipped.pkl", ".part.pkl", ".part.pkl.tmp")))
        if want is not None:
            shards = [q for q in shards
                      if int(re.search(r"shard(\d+)\.pkl$", q.name).group(1)) in want]
            missing = want - {int(re.search(r"shard(\d+)\.pkl$", q.name).group(1)) for q in shards}
            assert not missing, f"--shards asked for {sorted(missing)} but those {split} shards are absent"
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

        stem = args.name or f"refs_{split}"
        joblib.dump(lib, d / f"{stem}.pkl")
        (d / f"{stem}.text.json").write_text(json.dumps(text, ensure_ascii=False, indent=1))
        summary[split] = dict(clips=len(lib), frames=int(frames.sum()),
                              minutes=float(frames.sum() / 30 / 60), captions=caps,
                              fit_err_m=float(errs.mean()), **stats)

    # A partial merge must not overwrite the full build's summary either.
    sfile = f"{args.name}_summary.json" if args.name else "refs_summary.json"
    (d / sfile).write_text(json.dumps(summary, indent=2))
    stem = args.name or f"refs_{{{args.splits}}}"
    print(f"\nwrote {stem}.pkl + .text.json + {sfile} in {d}")


if __name__ == "__main__":
    main()
