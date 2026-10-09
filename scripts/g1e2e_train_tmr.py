"""Train the text <-> G1-motion retrieval model on our own robot trajectories.

See `hml_phys/g1_tmr.py` for why this exists and why it is the field's approach rather than mapping
robot motion back onto SMPL joints. In short: a human-motion evaluator saturates near chance on
executed robot motion, so it cannot rank policies, and five of eleven surveyed text-to-humanoid papers
train exactly this instead.

DATA. `data/g1_e2e/rollouts_train_part{1,2}.pkl` -- 146,297 teacher-driven trajectories over 8,731
clips with 26,086 captions, recorded under the teacher's own domain randomisation (STATUS.md §5.6).
Each carries `proprio [T,51]` at 50 Hz and the clip's captions.

TWO EXCLUSIONS, both load-bearing:

  --holdout-refs  Drops every clip in the closed-loop evaluation set. Policies are scored on the BC
                  training prompt pool (CLAUDE.md §11), so without this the retrieval model would have
                  trained on the very (text, motion) pairs it is later asked to retrieve, and R@1 would
                  be inflated by memorisation rather than measuring alignment.

  --drop-failed   Drops rollouts the recorder marked `failed`. A fallen trajectory does not depict its
                  caption; training on it teaches the model that falling matches the text. Keeping them
                  out also means a policy's fallen rollouts are out of distribution at scoring time and
                  score badly, which is the behaviour we want.

SELECTION. Checkpoints are selected on `rollouts_test.ref.pkl` -- the G1 rollout dataset has a train
and a test split, so test is its evaluation split and selection belongs there (CLAUDE.md §1).

Run on a compute node:
    python scripts/g1e2e_train_tmr.py \
      --rollouts data/g1_e2e/rollouts_train_part1.pkl,data/g1_e2e/rollouts_train_part2.pkl \
      --rollouts-test data/g1_e2e/rollouts_test.ref.pkl \
      --holdout-refs data/g1_e2e/refs_train_part1.pkl --holdout-n 512 \
      --out outputs/g1e2e/tmr --device cuda:0
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

from hml_phys.evaluator import GLOVE_DIR, HML_ROOT, read_split, read_texts  # noqa: E402
from hml_phys.g1_tmr import (DIM_POS, G1TMR, MAX_LEN, MIN_LEN, UNIT_LEN,  # noqa: E402
                             motion_feature, save_tmr)
from hml_phys.t2m.word_vectorizer import WordVectorizer                   # noqa: E402

BATCH = 32          # R-precision is defined over batches of 32 -- do not change (hml_phys/evaluator.py)
MAX_TEXT_LEN = 20


def encode_tokens(wv, tokens):
    """Guo's text encoding, as HMLEvaluator.encode_tokens does it -- kept identical on purpose."""
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
        pos.append(po[None])
        emb.append(we[None])
    return (np.concatenate(emb, 0).astype(np.float32),
            np.concatenate(pos, 0).astype(np.float32), sent_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="comma-separated training rollout pickles")
    ap.add_argument("--rollouts-test", required=True)
    ap.add_argument("--holdout-refs", default=None,
                    help="the reference library the closed loop is evaluated on; its first "
                         "--holdout-n clips are removed from TRAINING so scoring is not memorisation")
    ap.add_argument("--holdout-n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slice", default="",
                    help="i/n -- keep only clip-slice i of n, partitioned by a stable hash of the "
                         "clip key. Used to train several retrieval models on DISJOINT data so an "
                         "ensemble's members fail differently: Coste et al. (arXiv 2310.02743) show "
                         "that averaging an ensemble is NOT conservative (one member overestimating "
                         "is enough to be exploited) while taking the MINIMUM is, with no "
                         "hyperparameter. Different seeds alone would leave the members sharing every "
                         "data idiosyncrasy. Note WARM-style weight averaging is NOT applicable here: "
                         "it needs the members to share a pretrained init so they stay linearly mode "
                         "connected, and these are trained from scratch.")
    ap.add_argument("--max-reps", type=int, default=20,
                    help="domain-randomised passes kept per clip; 20 is all of them")
    ap.add_argument("--drop-failed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # ---- captions -> Guo's POS tokens, by (clip key, caption) -------------------------------------
    t0 = time.time()
    cap2tok = {}
    for split in ("train", "test", "val"):
        try:
            names = read_split(split)
        except Exception:
            continue
        for n in names:
            try:
                for t in read_texts(n, HML_ROOT):
                    cap2tok[(n, t["caption"].strip())] = t["tokens"]
            except Exception:
                pass
    print(f"caption->tokens: {len(cap2tok)} pairs, {time.time() - t0:.0f}s", flush=True)

    holdout = set()
    if args.holdout_refs:
        lib = joblib.load(args.holdout_refs)
        holdout = set(list(lib)[:args.holdout_n])
        print(f"holding {len(holdout)} evaluation clips out of TRAINING", flush=True)

    wv = WordVectorizer(GLOVE_DIR, "our_vab")

    keep_slice = None
    if args.slice:
        import hashlib
        si, sn = (int(v) for v in args.slice.split("/"))
        assert 0 <= si < sn, args.slice
        # A stable hash, not Python's randomised hash(), so the partition is identical across runs.
        keep_slice = lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % sn == si
        print(f"slice {si}/{sn}: keeping the clips whose md5 falls in bucket {si}", flush=True)

    def build(paths, drop_keys, tag, max_reps):
        feats, texts, n_skip_tok, n_skip_short, n_fail, n_held = [], [], 0, 0, 0, 0
        n_slice = 0
        per_clip = {}
        for p in paths:
            d = joblib.load(p)
            for k, v in d.items():
                bk = str(v.get("base_key", k))
                if bk in drop_keys:
                    n_held += 1
                    continue
                if args.drop_failed and v.get("failed", False):
                    n_fail += 1
                    continue
                if per_clip.get(bk, 0) >= max_reps:
                    continue
                if keep_slice is not None and not keep_slice(bk):
                    n_slice += 1
                    continue
                f = motion_feature(v["proprio"], float(v.get("fps", 50)))
                if len(f) < MIN_LEN:
                    n_skip_short += 1
                    continue
                toks = [cap2tok.get((bk, c.strip())) for c in v.get("captions", [])]
                toks = [t for t in toks if t]
                if not toks:
                    n_skip_tok += 1
                    continue
                per_clip[bk] = per_clip.get(bk, 0) + 1
                feats.append(f[:MAX_LEN])
                texts.append(toks)
            del d
        print(f"{tag}: {len(feats)} trajectories over {len(per_clip)} clips "
              f"(held out {n_held}, other slices {n_slice}, failed {n_fail}, "
              f"too short {n_skip_short}, no tokens {n_skip_tok})",
              flush=True)
        return feats, texts

    tr_f, tr_t = build([s for s in args.rollouts.split(",")], holdout, "train", args.max_reps)
    _ks, keep_slice = keep_slice, None       # the test split is NEVER sliced
    te_f, te_t = build([args.rollouts_test], set(), "test", 1)
    keep_slice = _ks
    assert len(tr_f) > args.batch and len(te_f) >= BATCH

    # ---- normalisation, from the training split only ---------------------------------------------
    cat = np.concatenate(tr_f[:20000], 0)
    mean, std = cat.mean(0), cat.std(0)
    std[std < 1e-6] = 1.0
    print(f"normaliser over {len(cat)} frames: |mean| max {np.abs(mean).max():.3f}, "
          f"std {std.min():.4f}-{std.max():.4f}", flush=True)

    model = G1TMR(device=dev).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"G1TMR: {n_par / 1e6:.1f} M params, motion input {model.input_dim}", flush=True)

    def pack(feats, texts, idx, crop_rng):
        """-> padded motion [B,MAX,51], m_lens, word_embs, pos, cap_lens (all descending by cap_len)."""
        mo = np.zeros((len(idx), MAX_LEN, model.input_dim), np.float32)
        ml, we, po, cl = [], [], [], []
        for j, i in enumerate(idx):
            f = feats[i]
            L = (len(f) // UNIT_LEN) * UNIT_LEN          # Guo quantises length to the unit
            L = max(min(L, MAX_LEN), UNIT_LEN * (MIN_LEN // UNIT_LEN))
            s = crop_rng.randint(0, max(len(f) - L, 0) + 1)
            x = (f[s:s + L] - mean) / std
            mo[j, :len(x)] = x
            ml.append(len(x))
            tk = texts[i][crop_rng.randint(len(texts[i]))]
            a, b, c = encode_tokens(wv, tk)
            we.append(a); po.append(b); cl.append(c)
        order = np.argsort(-np.asarray(cl), kind="stable")   # packed GRUs need descending lengths
        g = lambda a: np.asarray(a)[order]
        return (torch.from_numpy(mo[order]).to(dev), torch.from_numpy(g(ml)).long().to(dev),
                torch.from_numpy(np.stack(we)[order]).to(dev),
                torch.from_numpy(np.stack(po)[order]).to(dev),
                torch.from_numpy(g(cl)).long().to(dev))

    @torch.no_grad()
    def evaluate(feats, texts, seed=0):
        """R@1/2/3 over batches of 32, Euclidean, exactly the metric's own definition."""
        model.eval()
        r = np.zeros(3); n = 0
        er = np.random.RandomState(seed)
        perm = er.permutation(len(feats))
        for b in range(0, len(perm) - BATCH + 1, BATCH):
            idx = perm[b:b + BATCH]
            mo, ml, we, po, cl = pack(feats, texts, idx, er)
            te, me = model.co_embed(we, po, cl, mo, ml)
            d = torch.cdist(te, me)
            rank = d.argsort(dim=1)
            hit = (rank == torch.arange(BATCH, device=dev)[:, None]).float()
            for k in range(3):
                r[k] += hit[:, :k + 1].sum().item()
            n += BATCH
        model.train()
        return r / max(n, 1), n

    best, hist = -1.0, []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = rng.randint(0, len(tr_f), size=args.batch)
        mo, ml, we, po, cl = pack(tr_f, tr_t, idx, rng)
        te, me = model.co_embed(we, po, cl, mo, ml)
        loss = model.loss(te, me)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"step {step} loss {float(loss):.4f} temp {float(model.log_temp.exp()):.3f} "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r, n = evaluate(te_f, te_t)
            rec = dict(step=step, loss=float(loss), r1=float(r[0]), r2=float(r[1]), r3=float(r[2]),
                       n_test=int(n), minutes=(time.time() - t0) / 60)
            hist.append(rec)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            flag = ""
            if r[0] > best:
                best = r[0]
                save_tmr(out / "best.pt", model, mean, std,
                         dict(step=step, r1=float(r[0]), r2=float(r[1]), r3=float(r[2]),
                              n_train=len(tr_f), n_test=len(te_f), args=vars(args)))
                flag = "  <- new best"
            print(f"  [test] step {step} R@1 {r[0]:.4f} R@2 {r[1]:.4f} R@3 {r[2]:.4f} "
                  f"(chance {1 / BATCH:.4f}, {n} items){flag}", flush=True)

    print(f"done, best test R@1 {best:.4f}, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
