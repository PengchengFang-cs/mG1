"""Semantic scoring of G1 closed-loop rollouts (docs/04 line A).

Input is a pkl written by `scripts/g1_eval_rollout.py`: per episode the recorded link positions and the
segments (one contiguous span per active prompt).  Each segment becomes one item: link positions ->
22 SMPL joints (`hml_phys/g1_to_smpl.py`) -> 20 fps -> official 263-d -> our Guo evaluator, scored against the
caption that was active while it was produced.

**How far these numbers travel.**  The joint mapping is a geometric correspondence, not a retarget, and the
evaluator was trained on human motion, so absolute R-precision and FID from this path are NOT comparable to
published HumanML3D numbers (docs/06 §2.2b) and are not comparable to our own SMPL rows either.  They are
comparable between G1 rows produced through this identical path -- our policies against each other and against
the tracker ceiling (`--source tracker`) and the hold floor.  Always report the ceiling alongside.

Conventions kept from the SMPL side: fallen episodes are TRUNCATED at the fall, never dropped (CLAUDE.md §2),
and the whole thing is computed exactly once (§4).
"""
import argparse, json, os, sys, time
from collections import Counter

import joblib
import numpy as np

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.evaluator import HML_ROOT, HMLEvaluator, MIN_MOTION_LEN, build_gt_items, format_summary
from hml_phys.g1_to_smpl import G1ToSMPL, g1_body_pos_to_hml263

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", nargs="+", required=True)
ap.add_argument("--gt_split", default="test", help="HumanML3D split providing the FID reference distribution")
ap.add_argument("--replications", type=int, default=1, help="project CLAUDE.md §4: permanently 1")
ap.add_argument("--min_seg_s", type=float, default=2.0, help="segments shorter than this are dropped (40 frames @20fps)")
ap.add_argument("--caption_template", default="a person {}",
                help="how a BABEL label is turned into a caption. The Guo text encoder was trained on "
                     "HumanML3D sentences, not bare labels; the template was fixed once on the kinematic "
                     "reference row before any policy was scored (docs/04).")
ap.add_argument("--shuffle_captions", type=int, default=0,
                help="control: randomly permute the captions across segments. Physically identical motion "
                     "with the wrong label, i.e. what this instrument scores by chance. Not a reportable row "
                     "on its own -- it calibrates how far above chance a policy row is.")
ap.add_argument("--first_segment_only", type=int, default=0,
                help="score only each episode's FIRST prompt segment. Truncating at the fall still leaves a "
                     "confound between rows with different fall rates: an episode that survives contributes "
                     "later, post-switch segments, which are harder than a fresh start. One segment per "
                     "episode removes it (CLAUDE.md §2).")
ap.add_argument("--pos_cache", default="data/g1_rollouts/hml_word_pos.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.replications != 1:
    raise SystemExit("repeated evaluation is permanently banned (project CLAUDE.md §4): --replications must be 1")


def word_pos_map(cache):
    """word -> its most frequent POS tag in the HumanML3D corpus.

    The evaluator's text encoder consumes 'word/POS' tokens, and our prompts are BABEL labels that carry no
    tags.  Rather than invent a tagger, take the tag HumanML3D itself uses for that word; anything unseen
    falls back to 'unk/OTHER', which is what the evaluator already uses for out-of-vocabulary tokens.
    """
    if os.path.exists(cache):
        return json.load(open(cache))
    import glob
    cnt = {}
    for f in glob.glob(os.path.join(HML_ROOT, "texts", "*.txt")):
        for line in open(f, encoding="utf-8", errors="ignore"):
            parts = line.strip().split("#")
            if len(parts) < 2:
                continue
            for tok in parts[1].split(" "):
                if "/" in tok:
                    w, p = tok.rsplit("/", 1)
                    cnt.setdefault(w, Counter())[p] += 1
    out = {w: c.most_common(1)[0][0] for w, c in cnt.items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


W2P = word_pos_map(args.pos_cache)


def tag(caption):
    toks = []
    for w in caption.lower().replace("-", " ").split():
        w = w.strip(".,!?")
        if not w:
            continue
        toks.append(f"{w}/{W2P[w]}" if w in W2P else "unk/OTHER")
    return toks or ["unk/OTHER"]


gen_items, phys, n_ep, n_fell, dropped, oov_caps = [], [], 0, 0, {"short": 0, "nonfinite": 0}, set()
mapper = None
t0 = time.time()
for p in args.rollouts:
    obj = joblib.load(p)
    if mapper is None:
        mapper = G1ToSMPL(obj["body_names"]); print(mapper.describe())
    fps = float(obj.get("fps", 50))
    phys.append(obj["physical"])
    for e in obj["episodes"]:
        n_ep += 1; n_fell += int(e["fell"])
        bp = np.asarray(e["body_pos"], np.float32)
        for (s, t, cap) in (e["segments"][:1] if args.first_segment_only else e["segments"]):
            if (t - s) / fps < args.min_seg_s:
                dropped["short"] += 1; continue
            feat, _ = g1_body_pos_to_hml263(bp[s:t], mapper, src_fps=fps)
            if len(feat) < MIN_MOTION_LEN:
                dropped["short"] += 1; continue
            if not np.isfinite(feat).all():
                dropped["nonfinite"] += 1; continue
            cap = args.caption_template.format(cap)
            toks = tag(cap)
            if all(x == "unk/OTHER" for x in toks):
                oov_caps.add(cap)
            key = f"g1_{len(gen_items)}"
            gen_items.append(dict(key=key, base=key, motion=feat, length=len(feat), rep=0,
                                  texts=[dict(caption=cap, tokens=toks)]))

if args.shuffle_captions:
    caps = [g["texts"][0] for g in gen_items]
    perm = np.random.RandomState(args.shuffle_captions).permutation(len(caps))
    same = 0
    for g, j in zip(gen_items, perm):
        same += int(caps[j]["caption"] == g["texts"][0]["caption"])
        g["texts"] = [caps[j]]
    print(f"caption shuffle (seed {args.shuffle_captions}): {same}/{len(gen_items)} segments kept their own "
          f"caption by chance")

print(f"episodes {n_ep}, fell {n_fell} ({100*n_fell/max(1,n_ep):.1f}%), segments scored {len(gen_items)}, "
      f"dropped {dropped}, captions with no in-vocabulary word {len(oov_caps)} {sorted(oov_caps)[:5]}, "
      f"convert {time.time()-t0:.0f}s", flush=True)
assert gen_items, "no scorable segments"

gt_items, _ = build_gt_items(args.gt_split)
ev = HMLEvaluator("cuda")
summary, _ = ev.evaluate(gt_items, gen_items, replications=args.replications, physics=True, gen_official_crop=False)
summary["duration_completion"] = 1.0 - n_fell / max(1, n_ep)       # ADAPT's "success": the episode never fell
summary["n_episodes"] = n_ep; summary["n_fell"] = n_fell; summary["n_segments"] = len(gen_items)
summary["segment_len_s"] = float(np.mean([g["length"] for g in gen_items]) / 20.0)
summary["physical"] = phys
summary["gt_split"] = args.gt_split
summary["caption_template"] = args.caption_template
summary["shuffle_captions"] = args.shuffle_captions
summary["first_segment_only"] = args.first_segment_only
summary["note"] = ("G1 links -> 22 SMPL joints -> Guo evaluator. Absolute values are NOT comparable to "
                   "published HumanML3D numbers, only between G1 rows scored through this identical path "
                   "(policies, the tracker ceiling, the hold floor). Single rollout, single computation.")
print(format_summary(summary))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(summary, open(args.out, "w"), indent=1, default=float)
print(f"wrote {args.out}")
