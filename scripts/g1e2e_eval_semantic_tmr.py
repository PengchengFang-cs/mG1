"""Score G1 rollouts for semantic alignment with our own text <-> robot-motion retrieval model.

This replaces the SMPL-mapping ladder (`scripts/g1e2e_eval_semantic.py`), which was shown to have no
discriminative power where it matters: with the ground truth matched to the same clips, every EXECUTED
rollout -- the teacher's included, and it is handed the correct motion -- landed at R@1 0.048 to 0.081
against a chance level of 0.031, and the ordering flipped between ground-truth sets (STATUS.md §5.11d).
The human-motion evaluator saturates on robot motion.

The retrieval model trained on our own 136,647 robot trajectories reaches R@1 0.5924 on held-out robot
motion against the same chance level (STATUS.md §5.12), i.e. 19x chance and comparable to Guo's 0.511
on human motion. So robot motion carries the semantics perfectly well; the old instrument was the
problem. Five of eleven surveyed text-to-humanoid papers do exactly this (STATUS.md §5.11c).

The retrieval model never saw the 512 evaluation clips: `--holdout-refs` removed them from its training
set, so a high score here is alignment and not memorisation.

METRICS, all in the retrieval model's 512-d co-embedding space, Euclidean, batches of 32:
    R@1/R@2/R@3   does the caption retrieve its own rollout out of 32 candidates
    MM-Dist       mean distance between a caption and its own rollout
    Diversity     mean pairwise distance between rollout embeddings
    FID           Frechet distance to RECORDED teacher motion (--real), the "real robot motion"
                  distribution for these captions

NOT comparable to any other paper's R@1 -- but neither are theirs to each other, since each trains its
own retrieval model, so this subfield compares only within a paper against its own baselines. These
numbers go beside OUR anchors and never into a table beside someone else's (CLAUDE.md §2).

Run on a compute node:
    python scripts/g1e2e_eval_semantic_tmr.py --tmr outputs/g1e2e/tmr/best.pt \
      --npz teacher=outputs/g1e2e/teacher_x.bodypos.npz ppoC=outputs/g1e2e/eval_ppo_ppoC.bodypos.npz \
      --real data/g1_e2e/rollouts_test.ref.pkl --out outputs/g1e2e/semantic_tmr.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hml_phys.evaluator import GLOVE_DIR, read_split, read_texts            # noqa: E402
from hml_phys.g1_tmr import MAX_LEN, MIN_LEN, UNIT_LEN, load_tmr, motion_feature   # noqa: E402
from hml_phys.t2m.metrics import (calculate_activation_statistics,          # noqa: E402
                                  calculate_diversity, calculate_frechet_distance)
from hml_phys.t2m.word_vectorizer import WordVectorizer                     # noqa: E402

BATCH = 32
MAX_TEXT_LEN = 20


def encode_tokens(wv, tokens):
    if len(tokens) < MAX_TEXT_LEN:
        tokens = ["sos/OTHER"] + list(tokens) + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        tokens = ["sos/OTHER"] + list(tokens[:MAX_TEXT_LEN]) + ["eos/OTHER"]
        sent_len = len(tokens)
    pos, emb = [], []
    for tok in tokens:
        we, po = wv[tok]
        pos.append(po[None]); emb.append(we[None])
    return (np.concatenate(emb, 0).astype(np.float32),
            np.concatenate(pos, 0).astype(np.float32), sent_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmr", required=True)
    ap.add_argument("--npz", nargs="+", required=True, help="label=path/to.bodypos.npz (needs proprio)")
    ap.add_argument("--real", default=None, help="recorded robot rollouts, the FID reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, mean, std, meta = load_tmr(args.tmr, dev)
    print(f"TMR from {args.tmr}: trained to step {meta.get('step')}, held-out robot R@1 "
          f"{meta.get('r1', float('nan')):.4f} (chance {1 / BATCH:.4f})", flush=True)

    t0 = time.time()
    cap2tok = {}
    for split in ("train", "test", "val"):
        try:
            names = read_split(split)
        except Exception:
            continue
        for n in names:
            try:
                for t in read_texts(n):
                    cap2tok.setdefault(n, []).append(t["tokens"])
            except Exception:
                pass
    print(f"captions: {len(cap2tok)} clips, {time.time() - t0:.0f}s", flush=True)
    wv = WordVectorizer(GLOVE_DIR, "our_vab")

    def prep(feat_list, key_list, tag):
        items = []
        short = 0
        for f, k in zip(feat_list, key_list):
            if len(f) < MIN_LEN or k not in cap2tok:
                short += 1
                continue
            L = min((len(f) // UNIT_LEN) * UNIT_LEN, MAX_LEN)
            items.append(dict(key=k, feat=f[:L], toks=cap2tok[k]))
        print(f"  {tag}: {len(items)} items, dropped {short}", flush=True)
        return items

    def from_npz(path):
        d = np.load(path)
        assert "proprio" in d.files, (
            f"{path} has no proprio (keys: {d.files}). Rollouts taken before the retrieval model "
            f"existed did not save it; re-run the evaluation to add it.")
        pr, fs, hz = d["proprio"], d["fall_step"].astype(int), d["horizon"].astype(int)
        keys = [str(k) for k in d["keys"]]
        n = np.minimum(fs, hz)          # truncate at the fall (CLAUDE.md §2)
        feats = [motion_feature(pr[i, :int(n[i])]) for i in range(len(keys)) if int(n[i]) >= 8]
        kk = [keys[i] for i in range(len(keys)) if int(n[i]) >= 8]
        return feats, kk

    rows = []
    if args.real:
        d = joblib.load(args.real)
        feats, keys = [], []
        for k, v in d.items():
            feats.append(motion_feature(v["proprio"], float(v.get("fps", 50))))
            keys.append(str(v.get("base_key", k)))
        rows.append(("0_recorded_teacher(real)", prep(feats, keys, "0_recorded_teacher(real)")))
        del d
    for spec in args.npz:
        assert "=" in spec, f"--npz wants label=path, got {spec!r}"
        label, path = spec.split("=", 1)
        f, k = from_npz(path)
        rows.append((label, prep(f, k, label)))

    @torch.no_grad()
    def score(items, rng):
        perm = rng.permutation(len(items))
        r = np.zeros(3); match = 0.0; n = 0; ems = []
        for b in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[b:b + BATCH]
            mo = np.zeros((BATCH, MAX_LEN, model.input_dim), np.float32)
            ml, we, po, cl = [], [], [], []
            for j, i in enumerate(idx):
                it = items[i]
                x = (it["feat"] - mean) / std
                mo[j, :len(x)] = x
                ml.append(len(x))
                a, bb, c = encode_tokens(wv, it["toks"][rng.randint(len(it["toks"]))])
                we.append(a); po.append(bb); cl.append(c)
            order = np.argsort(-np.asarray(cl), kind="stable")
            g = lambda a: torch.from_numpy(np.asarray(a)[order]).to(dev)
            te, me = model.co_embed(g(we).float(), g(po).float(), g(cl).long(),
                                    torch.from_numpy(mo[order]).to(dev), g(ml).long())
            d = torch.cdist(te, me)
            rank = d.argsort(dim=1)
            hit = (rank == torch.arange(BATCH, device=dev)[:, None]).float()
            for k in range(3):
                r[k] += hit[:, :k + 1].sum().item()
            match += d.diagonal().sum().item()
            n += BATCH
            ems.append(me.cpu().numpy())
        ems = np.concatenate(ems, 0)
        return dict(top1=r[0] / n, top2=r[1] / n, top3=r[2] / n, mm_dist=match / n,
                    diversity=float(calculate_diversity(ems, min(300, len(ems) - 1))),
                    n_items=int(n)), ems

    out, real_stats = {}, None
    rng0 = np.random.RandomState(args.seed)
    for label, items in rows:
        if len(items) < BATCH:
            print(f"{label}: only {len(items)} items, need {BATCH} -- skipped", flush=True)
            out[label] = dict(skipped=len(items))
            continue
        s, ems = score(items, np.random.RandomState(args.seed))
        if real_stats is None:
            real_stats = calculate_activation_statistics(ems)      # the first row is the real set
            s["fid"] = 0.0
        else:
            mu, cov = calculate_activation_statistics(ems)
            s["fid"] = float(calculate_frechet_distance(real_stats[0], real_stats[1], mu, cov))
        out[label] = s
        print(f"{label:28s} R@1 {s['top1']:.4f}  R@2 {s['top2']:.4f}  R@3 {s['top3']:.4f}  "
              f"FID {s['fid']:8.3f}  MM-Dist {s['mm_dist']:.3f}  Div {s['diversity']:.3f}  "
              f"({s['n_items']})", flush=True)
    del rng0

    Path(args.out).write_text(json.dumps(dict(rows=out, tmr=meta, args=vars(args)), indent=1))
    print(f"\nwrote {args.out}")
    print(f"chance R@1 = {1 / BATCH:.4f}. Our own retrieval model, trained on robot motion with the "
          f"512 evaluation clips held out. Comparable between these rows, NOT across papers.")


if __name__ == "__main__":
    main()
