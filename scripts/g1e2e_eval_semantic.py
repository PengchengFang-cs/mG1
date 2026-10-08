"""The semantic ladder for the end-to-end G1 line: does the robot do what the caption says?

WHY A LADDER AND NOT A NUMBER. The Guo evaluator was trained on SMPL humans in HumanML3D's 263-d
representation. A G1 rollout has to be mapped into that joint set to be scored, and that mapping is
lossy and ours (`hml_phys/g1_to_smpl.py`). So an absolute R@1 from this pipeline is NOT comparable to
any published HumanML3D number -- and the literature does not give us one to compare against either:
the convention is to score the GENERATOR on human motion with Guo and the TRACKER on robot-native
success rate and tracking error (RoboGhost, arXiv 2510.14952, keeps exactly that separation; FRoM-W1
reports robot-side results only as bar charts). Nobody publishes R@1 for executed robot motion.

What makes our number interpretable is therefore not an external baseline but four rungs measured
through ONE identical path, on the same clips:

  1  kinematic GT     HumanML3D's own motion, scored directly       the representation's ceiling
  2  reference        the retargeted reference through the mapper   what the G1 SHAPE + the mapper cost
  3  teacher          the tracker following that reference          what physics and tracking cost
  4  ours             text in, no reference at all                  what text conditioning costs

Rung 2 is the one that makes this honest. It is the correct motion, executed by nothing -- so if rung 2
already scores near chance, the mapper has destroyed the signal and no policy row below it means
anything. Rung 2 validates the instrument before the instrument is used. It costs no simulation: the env
computes the reference's own body positions every step and the rollout now saves them as `body_pos_gt`.

PROTOCOL. Clips are the 512 of the BC training prompt pool (CLAUDE.md §11), so the GT distribution is
the TRAIN split, not test -- the question is "can it execute what it was taught", and R@1 against test
captions would be a different question. Published HumanML3D numbers are test-split; ours are not, which
is a second independent reason they do not compare. Fallen clips are TRUNCATED, not excluded (§2).
One rollout per checkpoint, each metric once (§4, asserted inside HMLEvaluator.evaluate).

Run on a compute node:
    python scripts/g1e2e_eval_semantic.py \
      --npz ours=outputs/g1e2e/eval_ppo_ppoA.bodypos.npz \
            teacher=outputs/g1e2e/teacher_part1_pos.bodypos.npz \
      --reference-from outputs/g1e2e/teacher_part1_pos.bodypos.npz \
      --split train --out outputs/g1e2e/semantic_ladder.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hml_phys.evaluator import HMLEvaluator, build_gt_items, MIN_MOTION_LEN   # noqa: E402
from hml_phys.g1_to_smpl import G1ToSMPL, g1_body_pos_to_hml263               # noqa: E402

CONTROL_HZ = 50.0
# The 22 rigid bodies legged_gym produces for g1_21dof.urdf with fixed joints collapsed, then the three
# EXTEND bodies in the order `extend_parent_ids: [17, 21, 0]` gives them.
LINKS22 = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link",
]
EXTENDS = ["left_hand_site", "right_hand_site", "head_link"]


def body_names_for(n):
    if n == 22:
        return LINKS22
    if n == 25:
        return LINKS22 + EXTENDS
    raise SystemExit(f"{n} bodies is neither the 22-link nor the 25-link-with-extends layout")


def to_items(bp, fs, hz, keys, cap2tok, mapper, tag):
    """[B,T,n,3] z-up 50 Hz robot (or reference) positions -> evaluator items, truncated at each fall."""
    items, short, nofinite, notok = [], 0, 0, 0
    valid = np.minimum(fs, hz)
    for i, k in enumerate(keys):
        n = int(valid[i])
        if n < 8:
            short += 1
            continue
        feat, _ = g1_body_pos_to_hml263(bp[i, :n].astype(np.float64), mapper, src_fps=CONTROL_HZ)
        if len(feat) < MIN_MOTION_LEN:
            short += 1
            continue
        if not np.isfinite(feat).all():
            nofinite += 1
            continue
        tt = cap2tok.get(k)
        if not tt:
            notok += 1
            continue
        items.append(dict(key=k, base=k, motion=feat, length=len(feat),
                          texts=[dict(caption=c, tokens=t) for c, t in tt]))
    print(f"  {tag}: {len(items)} items, dropped {short} too short, {nofinite} non-finite, "
          f"{notok} without a caption", flush=True)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="label=path/to.bodypos.npz")
    ap.add_argument("--reference-from", default=None,
                    help="a npz holding body_pos_gt; adds rung 2 (the reference through the same mapper)")
    ap.add_argument("--split", default="train", help="the GT split the captions come from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-keys", action="store_true",
                    help="restrict the GT set to exactly the clips that were rolled out. WITHOUT this, "
                         "rung 1 is all ~24.5k train motions while rungs 2-4 are the ~490 clips of the "
                         "prompt pool, so rung 1 is NOT a matched control and the rung1->rung2 drop "
                         "cannot be attributed. Rungs 2,3,4 are matched to each other either way, "
                         "because they are the same clips through the same mapper and differ only in "
                         "what produced the motion.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    gt_items, _ = build_gt_items(args.split)
    cap2tok = {}
    for it in gt_items:
        cap2tok[it["key"]] = [(t["caption"], t["tokens"]) for t in it["texts"]]
    print(f"GT split {args.split}: {len(gt_items)} items, {time.time() - t0:.0f}s", flush=True)

    rungs = []
    if args.reference_from:
        d = np.load(args.reference_from)
        assert "body_pos_gt" in d.files, (
            f"{args.reference_from} has no body_pos_gt (keys: {d.files}). Rollouts taken before "
            f"2026-10-08 did not save it; re-run one to add rung 2.")
        bp = d["body_pos_gt"]
        m = G1ToSMPL(body_names_for(bp.shape[2]))
        print(f"reference mapper: {m.describe()}", flush=True)
        # The reference is the motion the clip ASKS for, so it runs to the clip's own end: no fall.
        hz = d["horizon"].astype(int)
        rungs.append(("2_reference", to_items(bp, hz, hz, [str(k) for k in d["keys"]], cap2tok, m,
                                              "2_reference")))

    for spec in args.npz:
        assert "=" in spec, f"--npz wants label=path, got {spec!r}"
        label, path = spec.split("=", 1)
        d = np.load(path)
        key = "body_pos_ext" if "body_pos_ext" in d.files else "body_pos"
        bp = d[key]
        m = G1ToSMPL(body_names_for(bp.shape[2]))
        print(f"{label}: {key} {bp.shape}; {m.describe()}", flush=True)
        rungs.append((label, to_items(bp, d["fall_step"].astype(int), d["horizon"].astype(int),
                                      [str(k) for k in d["keys"]], cap2tok, m, label)))

    if args.match_keys:
        used = {it["key"] for _, items in rungs for it in items}
        before = len(gt_items)
        gt_items = [it for it in gt_items if it["key"] in used]
        print(f"\n--match-keys: GT restricted {before} -> {len(gt_items)} items, the same clips the "
              f"rollouts cover. Rung 1 is now a matched control.", flush=True)
        assert len(gt_items) >= 32, f"only {len(gt_items)} GT items left; R-precision needs 32"

    ev = HMLEvaluator(args.device)
    out = {}

    def val(summary, *names):
        """HMLEvaluator aggregates over replications, so each entry is dict(mean=, ci95=). With
        replications pinned to 1 (CLAUDE.md §4) ci95 is identically 0, so only the mean is ever read
        and nothing is ever reported as +-."""
        for n in names:
            if n in summary:
                v = summary[n]
                return float(v["mean"]) if isinstance(v, dict) else float(v)
        return float("nan")

    # Rung 1: the GT against itself. For HumanML3D test this pipeline is known to give R@1 about 0.511
    # and FID about 0.002, so a wildly different value here means the GT path itself is broken.
    print("\nrung 1: kinematic GT against itself", flush=True)
    s1, _ = ev.evaluate(gt_items, None, replications=1)
    out["1_kinematic_gt"] = s1
    print(f"  R@1 {val(s1, 'real_top1'):.4f}  R@2 {val(s1, 'real_top2'):.4f}  "
          f"R@3 {val(s1, 'real_top3'):.4f}  MM-Dist {val(s1, 'real_mm_dist'):.4f}  "
          f"Diversity {val(s1, 'real_diversity'):.4f}", flush=True)

    for label, items in rungs:
        if len(items) < 32:
            print(f"\n{label}: only {len(items)} items, R-precision needs 32 -- skipped", flush=True)
            out[label] = dict(skipped=f"{len(items)} items < 32")
            continue
        print(f"\n{label}: {len(items)} items", flush=True)
        s, _ = ev.evaluate(gt_items, items, replications=1, gen_official_crop=False,
                           physics=False)
        out[label] = s
        g = lambda k: val(s, k, "gen_" + k)
        print(f"  R@1 {g('top1'):.4f}  R@2 {g('top2'):.4f}  R@3 {g('top3'):.4f}  "
              f"FID {g('fid'):.4f}  MM-Dist {g('mm_dist'):.4f}  Diversity {g('diversity'):.4f}",
              flush=True)

    Path(args.out).write_text(json.dumps(dict(ladder=out, args=vars(args)), indent=1))
    print(f"\nwrote {args.out}")
    print("NOT comparable to published HumanML3D numbers: different split (train, per CLAUDE.md §11), "
          "and the motion passes through a G1 retarget plus our own joint mapping. Comparable BETWEEN "
          "the rungs above, which is what they are for.")


if __name__ == "__main__":
    main()
