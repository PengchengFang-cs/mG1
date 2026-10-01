"""Build ADAPT's evaluation prompt pool and the matching reference-motion whitelist.

ADAPT evaluates on "the same prompt pool of 130 commands, covering diverse locomotion, exercises, and
upper-body gestures" (§4.1) and terminates on illegal torso contact (Appendix C).  Those two facts go
together: BABEL also labels sitting, lying, kneeling and climbing, where torso contact is the CORRECT
behaviour, so a contact-based fall criterion is only meaningful once such motions are out of the pool.
Measured on our data: the tracker scores 0.761 under a height proxy but 0.629 under the contact criterion,
and `sit` alone accounts for 634 s of the val annotations.

This script therefore produces two things from the BABEL frame annotations we already have:
  * `g1_prompt_pool_130.txt` -- the command pool, 130 labels drawn from the three categories the paper
    names, ranked by how much annotated time our data actually has for them (a command with no data is not
    a fair test of anything)
  * `adapt_motion_whitelist_<split>.txt` -- the reference motions whose every non-transition label is
    admissible, i.e. the motions on which "torso contact = fall" holds

The category rules are keyword-based and deliberately written out in full below rather than hidden in a
model, so they can be read and argued with.  The paper does not publish its 130 commands, so this list is
our reconstruction; that is recorded in docs/05 §7.
"""
import argparse, os, re, sys
from collections import Counter

import joblib

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")

# Torso contact is the correct behaviour for these, so they cannot be scored with a contact-based fall
# criterion. Matched as substrings against the normalised label.
# Torso contact is the correct behaviour for these, so they cannot be scored with a contact-based fall
# criterion. Matched on WHOLE WORDS: a substring test made "sit" match "t position", "back to position" and
# "walk transition", wrongly excluding 40 train / 23 val labels and ~30 usable reference motions.
GROUND = ("sit", "sits", "sitting", "lie", "lies", "lying", "lay", "kneel", "kneels", "kneeling", "crawl",
          "crawls", "crawling", "climb", "climbs", "climbing", "floor", "ground", "plank", "yoga", "sleep")
GROUND_PHRASES = ("lay down", "roll over", "get up", "stand up", "push up", "rest on", "bend over backwards")
# Labels that say nothing about WHAT to do: a command the evaluator cannot distinguish from any other.
VAGUE = ("transition", "unknown", "other", "none", "a pose", "apose", "t pose", "tpose", "t position",
         "neutral", "idle", "pose", "action", "movement", "motion", "still stand", "hold", "look", "step",
         "circle", "swing", "place", "lift", "exercise", "series", "etc", "something")
ARTICLES = {"a", "an", "the", "to", "with", "and", "in", "on", "of", "his", "her", "their", "its"}

CATEGORIES = {
    "locomotion": ("walk", "run", "jog", "sprint", "step", "turn", "jump", "hop", "march", "skip", "stride",
                   "shuffle", "back up", "backward", "forward", "sideways", "side step", "circle", "pace",
                   "crouch", "duck walk", "tiptoe", "stop", "stand"),
    "exercise": ("squat", "stretch", "lunge", "kick", "punch", "jumping jack", "arm circle", "rotate", "twist",
                 "bend", "exercise", "warm up", "boxing", "swing", "balance", "leg raise", "toe touch"),
    "gesture": ("wave", "clap", "point", "raise arm", "raise hand", "throw", "catch", "salute", "nod",
                "shake head", "gesture", "hand", "arm", "reach", "grab", "pick up", "place", "put down",
                "look", "knock", "hit", "push", "pull", "lift", "hold", "drink", "eat", "phone"),
}


def normalise(lab):
    """the same text the policy is trained on -- `adapt.data.clean_label` is applied at training time, so a
    pool entry that skips it would be conditioned on an embedding whose training data carried another string."""
    from hml_phys.babel_labels import clean_label
    lab = str(lab).strip().lower()
    lab = re.sub(r"^transition to\s+", "", lab)
    lab = re.sub(r"\s+", " ", lab)
    return clean_label(lab)


def words(lab):
    return [w for w in re.findall(r"[a-z]+", lab)]


def admissible(lab):
    """a label that can appear in a contact-scored episode at all"""
    w = set(words(lab))
    return not (w & set(GROUND)) and not any(ph in lab for ph in GROUND_PHRASES)


def is_vague(lab):
    w = words(lab)
    return lab in VAGUE or (len(w) == 1 and lab in VAGUE) or any(v in w for v in ("series", "etc"))


def dedup_key(lab):
    """Collapse literal synonyms. Two prompts that mean the same thing are fatal under Eq. S12-S13, which
    ranks the ground-truth motion against ALL candidates: the diagonal entry becomes unrankable and R@1 is
    capped below the paper's range by construction. Measured 12 such collisions in the first pool
    (walk in circle / circles / in a circle, squat / squats, step back / step backwards, ...).
    Key = the multiset of crudely stemmed content words, so word order and plurals collapse together."""
    out = []
    for w in words(lab):
        if w in ARTICLES:
            continue
        # BABEL spelling variants that stemming cannot merge; each one left a duplicate in the first pool
        w = {"backwards": "backward", "forwards": "forward", "sidestep": "sidestepping",
             "swinge": "swing", "swinges": "swing", "wipes": "wipe", "cleans": "clean",
             "bends": "bend", "raises": "raise", "intewine": "intertwine"}.get(w, w)
        if len(w) > 3 and w.endswith("es"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return tuple(sorted(out))


def category(lab):
    for cat, keys in CATEGORIES.items():
        if any(k in lab for k in keys):
            return cat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", nargs="+", default=["data/g1_motions/train_all_meta.pkl",
                                                  "data/g1_motions/val_all_meta.pkl"])
    ap.add_argument("--n_commands", type=int, default=130, help="ADAPT §4.1")
    ap.add_argument("--min_seconds", type=float, default=20.0, help="drop commands our data barely covers")
    ap.add_argument("--out_pool", default="data/g1_prompt_pool_130.txt")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--screen", default="start", choices=["start", "all"],
                    help="'start' screens only the label covering frame 0 (the only thing the reference "
                         "motion still determines once the command term is frozen); 'all' screens every label")
    args = ap.parse_args()

    dur, segs, per_split = Counter(), Counter(), {}
    for mp in args.meta:
        split = os.path.basename(mp).replace("_meta.pkl", "")
        per_split[split] = joblib.load(mp)
        for name, m in per_split[split].items():
            for a, b, lab, *_ in m.get("frame_ann", []):
                lab = normalise(lab)
                if lab in ("transition",) or not lab:
                    continue
                dur[lab] += float(b) - float(a); segs[lab] += 1

    # ---- the pool: admissible, categorisable, enough data, ranked by annotated time within each category
    pool, by_cat, seen_key = [], {c: [] for c in CATEGORIES}, {}
    n_dup = 0
    for lab, d in dur.most_common():          # most annotated time first, so the survivor of a clash is the
        if d < args.min_seconds or not admissible(lab) or is_vague(lab):   # better-supported spelling
            continue
        c = category(lab)
        if not c:
            continue
        k = dedup_key(lab)
        if k in seen_key:
            n_dup += 1; continue
        seen_key[k] = lab
        by_cat[c].append((lab, d, segs[lab]))
    print(f"{n_dup} literal-synonym collisions dropped")

    def _near(a, b):                      # cheap edit distance, for the residual-collision report
        if abs(len(a) - len(b)) > 2:
            return 99
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    # round-robin across the three categories so no single one dominates the pool
    i = 0
    while len(pool) < args.n_commands and any(by_cat[c] for c in by_cat):
        c = list(CATEGORIES)[i % len(CATEGORIES)]
        if by_cat[c]:
            pool.append((c,) + by_cat[c].pop(0))
        i += 1
    print(f"{len(dur)} distinct labels -> pool of {len(pool)}")
    for c in CATEGORIES:
        n = sum(1 for p in pool if p[0] == c)
        print(f"  {c:11s} {n:3d} commands, {sum(p[2] for p in pool if p[0] == c):.0f} s")
    os.makedirs(os.path.dirname(args.out_pool) or ".", exist_ok=True)
    labs = [lab for _, lab, _, _ in pool]
    residual = [(a, b) for i, a in enumerate(labs) for b in labs[i + 1:] if _near(a, b) <= 2]
    if residual:
        print(f"WARNING {len(residual)} near-duplicate pairs remain (edit distance <= 2): {residual[:6]}")
    with open(args.out_pool, "w") as f:
        for lab in labs:
            f.write(lab + "\n")
    print(f"wrote {args.out_pool}")

    # ---- the whitelist: motions with at least one admissible label and no ground-contact label at all
    for split, meta in per_split.items():
        keep, dropped = [], Counter()
        for name, m in meta.items():
            ann = sorted((float(a), float(b), normalise(l)) for a, b, l, *_ in m.get("frame_ann", []))
            labs = [l for _, _, l in ann if l and l != "transition"]
            if not labs:
                dropped["no label"] += 1; continue
            if args.screen == "start":
                # the reference motion only sets the INITIAL POSE: `start_from_zero_step=True` and the
                # evaluation now freezes the command term, so nothing after frame 0 is ever tracked. Screening
                # the whole clip drops ~214 train motions that begin standing and merely sit down later.
                at0 = [l for a, b, l in ann if a <= 0.02 and l and l != "transition"] or labs[:1]
                bad = [l for l in at0 if not admissible(l)]
            else:
                bad = [l for l in labs if not admissible(l)]
            if bad:
                dropped[bad[0]] += 1; continue
            keep.append(name)
        out = os.path.join(args.out_dir, f"adapt_motion_whitelist_{split}.txt")
        with open(out, "w") as f:
            f.write("\n".join(sorted(keep)) + "\n")
        print(f"{split}: {len(keep)}/{len(meta)} motions kept -> {out}")
        print(f"  top reasons dropped: {dropped.most_common(6)}")


if __name__ == "__main__":
    main()
