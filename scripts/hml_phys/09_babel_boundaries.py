"""Phase 0 of the advancement-signal pilot: join BABEL frame annotations onto our physics clips.

Why: the pilot asks whether the intent residual can tell "this sub-instruction is finished" from "still
executing".  That needs ground-truth segment boundaries.  HumanML3D's own tagged sub-captions are far too
sparse for this (test split: 3.8% of captions carry a time tag, only 9 clips have >= 2), but BABEL annotates
the same AMASS sequences frame by frame, and the G1 line already ships it
(TextOp/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/{train,val}.pkl).

Frame mapping (exact, from scripts/hml_phys/03_build_dataset.py:196-203):
    a clip's AMASS span is  t_start = (start_frame + head_trim) / 20 s ... t_end = (end_frame + head_trim) / 20 s
    physics indices         phys_i0 = round(t_start * fps_eff),  phys_i1 = round(t_end * fps_eff)
    so clip-local frame j  <->  AMASS time (phys_i0 + j) / fps_eff
    hence                  BABEL time t  ->  j = round(t * fps_eff) - phys_i0

What comes out (one row per boundary that falls strictly inside a clip):
    clip index, boundary frame j, the label ending and the label starting, whether a BABEL `transition`
    segment touches the boundary, and the two segment durations.
`transition` is BABEL's own explicit label, which is why it is recorded rather than guessed.

Open-ended instructions ("walk", "wave") have no well-defined completion; they are NOT dropped here, they
are flagged (`spans_whole_clip`, plus the label text) so the Phase-1 analysis can exclude them explicitly.
"""
import argparse, json, os, re, sys
from collections import defaultdict

import joblib
import numpy as np

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import ROOT

BABEL_DIR = "/iridisfs/scratch/pf2m24/projects/motion_rebot/TextOp/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps"

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test", help="our split; val is banned (CLAUDE.md §1)")
ap.add_argument("--out", default="")
ap.add_argument("--min_seg_frames", type=int, default=8, help="drop segments shorter than this after mapping")
ap.add_argument("--edge_margin", type=int, default=8, help="a boundary must sit this far inside the clip")
args = ap.parse_args()
assert args.split != "val", "the val split is banned in this project (CLAUDE.md §1)"
out_path = args.out or os.path.join(ROOT, f"babel_boundaries_{args.split}.npz")


def norm_amass(p):
    """AMASS path -> comparable key.  Ours are './pose_data/KIT/3/x_poses.npy', BABEL's 'KIT/3/x_poses.pkl'."""
    p = p.replace("\\", "/").lower()
    p = re.sub(r"^\./", "", p)
    p = re.sub(r"^pose_data/", "", p)
    p = re.sub(r"_(poses|stageii|stagei)\.(npy|npz|pkl)$", "", p)
    return re.sub(r"\.(npy|npz|pkl)$", "", p)


print("indexing BABEL ...", flush=True)
babel = {}
for part in ("train", "val"):                       # BABEL's own split, unrelated to ours
    for e in joblib.load(os.path.join(BABEL_DIR, f"{part}.pkl")):
        babel.setdefault(norm_amass(e["feat_p"]), []).append(e)
print(f"  {len(babel)} unique AMASS sequences", flush=True)

d = joblib.load(os.path.join(ROOT, f"hml_phys_{args.split}.pkl"))
n_clips = len(d["name"])
rows, per_clip, stat = [], [], defaultdict(int)
segs_by_clip = {}      # clip -> [(start, end, label), ...]  Phase 1 needs the label of the segment a window sits in

for i in range(n_clips):
    key = norm_amass(d["source_file"][i])
    ent = babel.get(key)
    if ent is None:
        stat["no_babel_entry"] += 1
        continue
    if len(ent) > 1:                                 # same AMASS file annotated twice: ambiguous, skip
        stat["ambiguous_multiple_babel_entries"] += 1
        continue
    ent = ent[0]
    fps, i0, n = float(d["fps_eff"][i]), int(d["phys_i0"][i]), int(d["n_frames"][i])
    # sanity: the clip must lie inside the annotated sequence
    if (i0 + n) / fps > float(ent["duration"]) + 1.0:
        stat["clip_past_babel_duration"] += 1
        continue

    segs = []
    for (a, b, raw, proc) in ent["frame_ann"]:
        ja, jb = int(round(float(a) * fps)) - i0, int(round(float(b) * fps)) - i0
        ja, jb = max(0, ja), min(n, jb)
        if jb - ja >= args.min_seg_frames:
            segs.append((ja, jb, str(raw), list(proc)))
    if len(segs) < 2:
        stat["fewer_than_two_segments_in_clip"] += 1
        continue
    segs.sort(key=lambda s: (s[0], s[1]))

    # overlapping non-transition labels = BABEL's simultaneous actions; the boundary notion breaks down
    real = [s for s in segs if s[2] != "transition"]
    overlap = any(real[k + 1][0] < real[k][1] - 2 for k in range(len(real) - 1))
    if overlap:
        stat["overlapping_labels"] += 1
        continue

    segs_by_clip[i] = [(int(a), int(b), str(r)) for (a, b, r, _) in segs]
    n_here = 0
    for k in range(len(segs) - 1):
        cur, nxt = segs[k], segs[k + 1]
        j = cur[1]                                   # boundary = end of the current segment
        if not (args.edge_margin <= j < n - args.edge_margin):
            continue
        if cur[2] == "transition" and nxt[2] == "transition":
            continue
        rows.append(dict(clip=i, frame=j,
                         prev_label=cur[2], next_label=nxt[2],
                         prev_frames=cur[1] - cur[0], next_frames=nxt[1] - nxt[0],
                         touches_transition=int(cur[2] == "transition" or nxt[2] == "transition"),
                         prev_spans_whole_clip=int((cur[1] - cur[0]) > 0.8 * n),
                         gap=max(0, nxt[0] - cur[1])))
        n_here += 1
    if n_here:
        stat["clips_with_boundaries"] += 1
        per_clip.append(dict(clip=i, name=d["name"][i], n_frames=n, fps_eff=fps,
                             n_boundaries=n_here, n_segments=len(segs),
                             labels=[s[2] for s in segs]))
    else:
        stat["clip_no_usable_boundary"] += 1

print(f"\n=== {args.split}: {n_clips} clips ===")
for k, v in sorted(stat.items(), key=lambda x: -x[1]):
    print(f"  {k:<36} {v}")
print(f"  usable boundaries                    {len(rows)}")
if rows:
    tr = sum(r["touches_transition"] for r in rows)
    ws = sum(r["prev_spans_whole_clip"] for r in rows)
    print(f"    of which touch a BABEL `transition` {tr} ({100*tr/len(rows):.0f}%)")
    print(f"    of which the ending label spans >80% of the clip (open-ended) {ws}")
    labs = defaultdict(int)
    for r in rows:
        labs[r["prev_label"]] += 1
    print("  most common ending labels:", ", ".join(f"{k}({v})" for k, v in sorted(labs.items(), key=lambda x: -x[1])[:10]))

np.savez_compressed(out_path,
                    clip=np.array([r["clip"] for r in rows], np.int64),
                    frame=np.array([r["frame"] for r in rows], np.int64),
                    prev_frames=np.array([r["prev_frames"] for r in rows], np.int64),
                    next_frames=np.array([r["next_frames"] for r in rows], np.int64),
                    touches_transition=np.array([r["touches_transition"] for r in rows], np.int8),
                    prev_spans_whole_clip=np.array([r["prev_spans_whole_clip"] for r in rows], np.int8),
                    gap=np.array([r["gap"] for r in rows], np.int64),
                    prev_label=np.array([r["prev_label"] for r in rows]),
                    next_label=np.array([r["next_label"] for r in rows]),
                    segments=json.dumps({str(k): v for k, v in segs_by_clip.items()}),
                    meta=json.dumps(dict(split=args.split, n_clips=n_clips, stats=dict(stat),
                                         min_seg_frames=args.min_seg_frames, edge_margin=args.edge_margin,
                                         per_clip=per_clip[:50])))
print(f"\nwrote {out_path}  ({len(rows)} boundaries over {stat['clips_with_boundaries']} clips)")
