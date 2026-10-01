"""Score our H-GPT generations with FRoM-W1's own released evaluator, to check our setup against their
reported R Top-1 = 0.332 on the HumanML3D-X benchmark (their Table 1).

Why this can be done without the gated data: R-precision and MM Dist are retrieval between the GENERATED
motion embeddings and the text embeddings (`t2m.py:158-170`) -- ground-truth motions are only needed for FID
and Diversity. So the captions from our local HumanML3D test split plus our own generations are enough, and
the scoring model is the one they released (`eval/finest.tar`, retrained by them on HumanML3D-X).

Reproduced exactly from `hGPT/metrics/t2m.py`: shuffle once, take groups of R_size = 32, euclidean distance
matrix between the group's text and motion embeddings, top-k over the argsort, MM Dist from the diagonal.
"""
import argparse, glob, os, sys

import numpy as np
import torch

R = "/iridisfs/scratch/pf2m24/projects/motion_rebot"
sys.path.insert(0, f"{R}/external/FRoM-W1/H-GPT")

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True, help="H-GPT samples_* directories")
ap.add_argument("--eval_ckpt", default=f"{R}/external/fromw1_weights/eval/finest.tar")
ap.add_argument("--eval_meta", default=f"{R}/external/fromw1_weights/eval/meta")
ap.add_argument("--data_meta", default=f"{R}/external/fromw1_data/data/retarget_assets/meta",
                help="the DATASET normalisation the generations are expressed in; `renorm4t2m` "
                     "(MotionX.py:181) de-normalises with these before re-normalising with the evaluator's")
ap.add_argument("--glove", default="/iridisfs/scratch/pf2m24/projects/Umdd/KV-Control/glove/")
ap.add_argument("--hml_texts", default="/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D/texts")
ap.add_argument("--hml_vecs", default="/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D/new_joint_vecs",
                help="source of the ground-truth clip lengths")
ap.add_argument("--ids", default=f"{R}/data/fromw1_eval/hml_test_ids.txt")
ap.add_argument("--crop_to_gt", type=int, default=1,
                help="hgpt.py:115-116 writes `feats_rst[i,:min_len]=motion[:,:lengths[i]]` with the GT length "
                     "and then encodes with `lengths_ref`, so a generation longer than its reference is cut "
                     "and a shorter one is zero-padded. Scoring at the generation's own length instead "
                     "measures a different thing.")
ap.add_argument("--gt_fps_scale", type=float, default=1.5,
                help="HumanML3D vectors are 20 fps, HumanML3D-X motions are 30 fps")
ap.add_argument("--r_size", type=int, default=32)
ap.add_argument("--top_k", type=int, default=3)
ap.add_argument("--max_text_len", type=int, default=20)
ap.add_argument("--unit_length", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=f"{R}/outputs/fromw1/rprec.json")
args = ap.parse_args()

from hGPT.data.default.word_vectorizer import WordVectorizer   # noqa: E402
from hGPT.models.evaluator import (MotionEncoderBiGRUCo, MovementConvEncoder,  # noqa: E402
                                   TextEncoderBiGRUCo)

dev = "cuda"
ck = torch.load(args.eval_ckpt, map_location="cpu")
txt_enc = TextEncoderBiGRUCo(word_size=300, pos_size=15, hidden_size=512, output_size=512)
mov_enc = MovementConvEncoder(input_size=623 - 4, hidden_size=512, output_size=512)
mot_enc = MotionEncoderBiGRUCo(input_size=512, hidden_size=1024, output_size=512)
txt_enc.load_state_dict(ck["text_encoder"]); mov_enc.load_state_dict(ck["movement_encoder"])
mot_enc.load_state_dict(ck["motion_encoder"])
for m in (txt_enc, mov_enc, mot_enc):
    m.eval().to(dev)
e_mean = torch.from_numpy(np.load(f"{args.eval_meta}/mean.npy")).float().to(dev)
e_std = torch.from_numpy(np.load(f"{args.eval_meta}/std.npy")).float().to(dev)
d_mean = torch.from_numpy(np.load(f"{args.data_meta}/mean.npy")).float().to(dev)
d_std = torch.from_numpy(np.load(f"{args.data_meta}/std.npy")).float().to(dev)
print(f"[score] evaluator loaded; feature dim {e_mean.shape[0]}; "
      f"dataset-vs-eval stats differ by |dmean| {float((d_mean - e_mean).abs().max()):.3f}")

# ---- the POS-tagged tokens the text encoder wants come from HumanML3D's own text files
wv = WordVectorizer(args.glove, "our_vab")
cap2tok = {}
for p in glob.glob(os.path.join(args.hml_texts, "*.txt")):
    for line in open(p, encoding="utf-8", errors="ignore"):
        parts = line.strip().split("#")
        if len(parts) >= 2 and parts[0]:
            cap2tok.setdefault(parts[0].strip(), parts[1].split(" "))
print(f"[score] {len(cap2tok)} captions with POS tags available")


def encode_text(caption):
    toks = cap2tok.get(caption.strip())
    if toks is None:
        return None
    if len(toks) < args.max_text_len:
        toks = ["sos/OTHER"] + toks + ["eos/OTHER"]
        n = len(toks)
        toks = toks + ["unk/OTHER"] * (args.max_text_len + 2 - n)
    else:
        toks = ["sos/OTHER"] + toks[: args.max_text_len] + ["eos/OTHER"]
        n = len(toks)
    we, po = zip(*[wv[t] for t in toks])
    return np.stack(we), np.stack(po), n


# caption -> ground-truth length, so a generation can be cropped the way their evaluation crops it
cap2len = {}
if args.crop_to_gt:
    import os.path as _op
    ids = [l.strip() for l in open(args.ids) if l.strip()]
    caps = [l.strip() for l in open(args.ids.replace("_ids", "_captions"))]
    for cid, cap in zip(ids, caps):
        f = _op.join(args.hml_vecs, cid + ".npy")
        if _op.exists(f):
            n20 = len(np.load(f, mmap_mode="r"))
            cap2len[cap] = int(round(n20 * args.gt_fps_scale))
    print(f"[score] {len(cap2len)} captions have a ground-truth length "
          f"(median {int(np.median(list(cap2len.values())))} frames @30fps)")

feats, embs, pos, lens, mlens, missing, n_crop, n_pad = [], [], [], [], [], 0, 0, 0
for d in args.samples:
    for fp in sorted(glob.glob(os.path.join(d, "*_feats_out.npy")),
                     key=lambda p: int(os.path.basename(p).split("_")[0])):
        i = int(os.path.basename(fp).split("_")[0])
        cp = os.path.join(d, f"{i}_text_in.txt")
        if not os.path.exists(cp):
            continue
        enc = encode_text(open(cp).read().strip())
        if enc is None:
            missing += 1; continue
        x = np.load(fp)[0]                                        # [T, 623]
        cap = open(cp).read().strip()
        if args.crop_to_gt:
            g = cap2len.get(cap)
            if g is None:
                missing += 1; continue
            if len(x) > g:
                x = x[:g]; n_crop += 1
            elif len(x) < g:
                x = np.concatenate([x, np.zeros((g - len(x), x.shape[1]), x.dtype)]); n_pad += 1
        t = (len(x) // args.unit_length) * args.unit_length       # the movement encoder downsamples by 4
        if t < args.unit_length * 2:
            missing += 1; continue
        feats.append(x[:t]); embs.append(enc[0]); pos.append(enc[1]); lens.append(enc[2]); mlens.append(t)
print(f"[score] {len(feats)} pairs scored, {missing} dropped; cropped to GT {n_crop}, zero-padded {n_pad}")
assert len(feats) > args.r_size, "not enough pairs for one retrieval group"

# ---- embed, padding each batch to its longest motion as the dataloader would
tmax = max(mlens)
T = torch.zeros(len(feats), tmax, 623)
for k, x in enumerate(feats):
    T[k, : len(x)] = torch.from_numpy(x).float()
# Both encoders wrap pack_padded_sequence with enforce_sorted=True, so each batch has to be handed over in
# descending length order and put back afterwards -- their dataloader does the same with
# `align_idx = np.argsort(lengths)[::-1]` (t2m.py:250).
with torch.no_grad():
    mot_e = torch.zeros(len(feats), 512)
    txt_e = torch.zeros(len(feats), 512)
    for s in range(0, len(feats), 64):
        sl = slice(s, min(s + 64, len(feats)))
        ml = np.array(mlens[sl]); tl = np.array(lens[sl])
        m_ord = np.argsort(-ml); t_ord = np.argsort(-tl)
        b = T[sl][m_ord].to(dev)
        b = b * d_std + d_mean               # renorm4t2m step 1: back to raw feature units
        b = (b - e_mean) / e_std              # renorm4t2m step 2: into the evaluator's normalisation
        mv = mov_enc(b[..., :-4])
        L = torch.tensor([max(int(m) // args.unit_length, 1) for m in ml[m_ord]], device=dev)
        e = mot_enc(mv, L).flatten(1).cpu()
        mot_e[sl] = e[np.argsort(m_ord)]
        te = txt_enc(torch.from_numpy(np.stack(embs[sl])[t_ord]).float().to(dev),
                     torch.from_numpy(np.stack(pos[sl])[t_ord]).float().to(dev),
                     torch.tensor(tl[t_ord], device=dev)).flatten(1).cpu()
        txt_e[sl] = te[np.argsort(t_ord)]


def euclidean_distance_matrix(a, b):
    d = (a[:, None] - b[None]).pow(2).sum(-1).sqrt()
    return d


g = torch.Generator().manual_seed(args.seed)
idx = torch.randperm(len(mot_e), generator=g)
txt_e, mot_e = txt_e[idx], mot_e[idx]
n_group = len(mot_e) // args.r_size
topk = torch.zeros(args.top_k); match = 0.0
for i in range(n_group):
    gt = txt_e[i * args.r_size:(i + 1) * args.r_size]
    gm = mot_e[i * args.r_size:(i + 1) * args.r_size]
    dm = euclidean_distance_matrix(gt, gm).nan_to_num()
    match += float(dm.trace())
    order = torch.argsort(dm, dim=1)
    hit = (order == torch.arange(args.r_size)[:, None])
    for k in range(args.top_k):
        topk[k] += float(hit[:, : k + 1].any(1).sum())
n = n_group * args.r_size
res = {f"R_precision_top_{k+1}": float(topk[k] / n) for k in range(args.top_k)}
res["MM_Dist"] = match / n
res.update(n_pairs=len(mot_e), n_scored=n, groups=n_group, seed=args.seed)
print("[score] " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
import json; os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"), indent=1)
print(f"[score] wrote {args.out}")
