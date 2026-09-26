"""Phase 1 of the advancement-signal pilot: can the intent residual tell "this sub-instruction is finished"
from "still executing"?

The idea under test is that a closed-loop policy could decide WHEN to move to the next sub-instruction by
watching the gap between the immediate intent it predicted and the intent the body actually produced.  The
cross-model review of the proposal put the weight exactly here:

    the residual measures PREDICTION CONSISTENCY, not COMPLETION.  It can stay small through correct ongoing
    walking, and it can be large for motion that is correct but simply unlike the sampled prediction.

So this probe does not ask "is the residual large at boundaries".  It asks whether the residual carries
completion-specific information BEYOND "the motion just changed".  Hence three controls, the third of which
is the one that can kill the idea:

  residual      ||IIP-sampled immediate intent  -  VAE(actual next 16 frames)||
  ctl_change    ||VAE(previous 16 frames)       -  VAE(actual next 16 frames)||   <- "the motion changed"
  ctl_speed     mean squared joint velocity over the next 16 frames               <- "the motion is fast"

If `residual` separates boundary windows from interior windows no better than `ctl_change` does, the residual
is only re-reading the motion change and the direction is dead.  The decisive number is therefore the AUROC
on **boundary vs HIGH-CHANGE interior**, where `ctl_change` is by construction uninformative.

Ground truth comes from scripts/hml_phys/09_babel_boundaries.py (BABEL frame annotations mapped onto our
clips).  Open-ended labels (the segment fills >80% of the clip) are excluded: "walk" has no completion.

No training, no rollout -- one forward pass per window.
"""
import argparse, json, os, sys
from collections import defaultdict

import joblib
import numpy as np
import torch

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys import tokens as tk
from hml_phys import intent_flow as ifl
from hml_phys.dataset import ROOT, TokenStats
from hml_phys.intent_data import state_from_tokens
from hml_phys.mc_rollout import load_policy
from hml_phys.text_clip import ClipText

L = 16                                   # MIND's intent horizon, and what the frozen VAE was trained on

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/mc_A_nosparse/step_200000.pt")
ap.add_argument("--split", default="test")
ap.add_argument("--boundaries", default="")
ap.add_argument("--text", default="babel", choices=["babel", "caption"],
                help="babel = condition on the CURRENT segment's BABEL label (the deployment setting); "
                     "caption = condition on the clip's HumanML3D caption (what the policy saw in training)")
ap.add_argument("--neg_per_pos", type=int, default=3, help="interior windows sampled per boundary, same clip")
ap.add_argument("--interior_margin", type=int, default=10, help="an interior window must be this far from any boundary")
ap.add_argument("--num_steps", type=int, default=32); ap.add_argument("--cfg", type=float, default=3.5)
ap.add_argument("--batch", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_pos", type=int, default=0)
ap.add_argument("--out", default="")
args = ap.parse_args()
assert args.split != "val", "the val split is banned in this project (CLAUDE.md §1)"
bnd_path = args.boundaries or os.path.join(ROOT, f"babel_boundaries_{args.split}.npz")
out_path = args.out or os.path.join(ROOT, f"intent_residual_probe_{args.split}_{args.text}.json")
dev = "cuda"
rng = np.random.RandomState(args.seed)


def auroc(pos, neg):
    """rank-based AUROC; returns 0.5 for a useless score."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------- data
print("loading ...", flush=True)
z = np.load(bnd_path, allow_pickle=True)
SEGS = {int(k): v for k, v in json.loads(str(z["segments"])).items()}   # clip -> [(start, end, label)]


def label_at(c, j):
    """the BABEL label of the segment the window's HISTORY sits in (frame j-1); '' if none covers it."""
    for a, b, lab in SEGS.get(c, []):
        if a <= j - 1 < b:
            return lab
    return ""
keep = (z["prev_spans_whole_clip"] == 0)          # open-ended instructions have no completion to detect
# a window needs L frames of history and L of future, so boundaries closer than L to either edge are unusable
# (09_babel_boundaries.py only guaranteed an 8-frame margin)
d_pre = joblib.load(os.path.join(ROOT, f"hml_phys_{args.split}.pkl"))
nfr = np.asarray(d_pre["n_frames"])
fits = (z["frame"] >= L) & (z["frame"] <= nfr[z["clip"]] - L)
n_edge = int((keep & ~fits).sum())
keep = keep & fits
b_clip, b_frame = z["clip"][keep], z["frame"][keep]
b_prev = z["prev_label"][keep]
b_trans = z["touches_transition"][keep]
print(f"{len(b_clip)} boundaries usable (dropped {int((z['prev_spans_whole_clip']==1).sum())} open-ended, {n_edge} too close to a clip edge)")

d = d_pre
stats = TokenStats(os.path.join(ROOT, "token_stats_v3.npz"))
by_clip = defaultdict(list)
for c, f in zip(b_clip, b_frame):
    by_clip[int(c)].append(int(f))

# build the sample list:每个边界一个正样本，同片段内取 neg_per_pos 个远离边界的负样本
samples = []          # (clip, frame, is_boundary, prev_label)
for k in range(len(b_clip)):
    c, f = int(b_clip[k]), int(b_frame[k])
    samples.append((c, f, 1, label_at(c, f) or str(b_prev[k])))
if args.max_pos:
    rng.shuffle(samples); samples = samples[:args.max_pos]
pos_clips = sorted({s[0] for s in samples})
for c in pos_clips:
    n = int(d["n_frames"][c])
    bad = np.array(by_clip[c])
    cand = [j for j in range(L, n - L) if np.all(np.abs(bad - j) > args.interior_margin)]
    if not cand:
        continue
    pick = rng.choice(cand, size=min(args.neg_per_pos * max(1, len(by_clip[c])), len(cand)), replace=False)
    for j in pick:
        samples.append((c, int(j), 0, label_at(c, int(j))))
samples = [s for s in samples if s[3]]
print(f"{sum(s[2] for s in samples)} boundary windows + {sum(1-s[2] for s in samples)} interior windows")

# ----------------------------------------------------------------- model
model, mstats, envc, margs, step = load_policy(args.ckpt, device=dev)
assert margs.get("arch") == "intent", "this probe needs the intent architecture (HIP/IIP + frozen VAE)"
s_read = float(margs.get("cond_aug_test", 0.75)) if float(margs.get("cond_aug", 0.0)) > 0 else 1.0
clip_enc = ClipText()
print(f"policy step {step}, s_read {s_read}, text source = {args.text}", flush=True)


def window_states(c, j):
    """-> (prev 16 frames, next 16 frames) as normalised 366-d VAE states, canonicalised on frame j-1."""
    sl = slice(j - L, j + L)
    with np.errstate(invalid="ignore", divide="ignore"):
        root, body = tk.window_tokens(d["body_pos"][c][sl], d["dof_state"][c][sl],
                                      d["root_state"][c][sl], d["action"][c][sl], origin=L - 1)
    st = state_from_tokens(*stats.norm(root, body))
    return st[:L], st[L:]


res = defaultdict(list)
order = list(range(len(samples)))
for b0 in range(0, len(order), args.batch):
    idx = order[b0:b0 + args.batch]
    hist = np.stack([window_states(samples[i][0], samples[i][1])[0] for i in idx])
    fut = np.stack([window_states(samples[i][0], samples[i][1])[1] for i in idx])
    caps = []
    for i in idx:
        c, j, _, lab = samples[i]
        if args.text == "babel":
            caps.append(lab)
        else:
            t = d["texts"][c]
            caps.append(t[0]["caption"] if t else "")
    tok, pool, tl = clip_enc.encode(caps)
    B = len(idx)
    text = (torch.from_numpy(tok).to(dev), torch.from_numpy(pool).to(dev), torch.from_numpy(tl).to(dev))
    etok, epool, elen = clip_enc.encode([""] * B)
    text_u = (torch.from_numpy(etok).to(dev), torch.from_numpy(epool).to(dev), torch.from_numpy(elen).to(dev))
    H = torch.from_numpy(hist).to(dev).float()
    F = torch.from_numpy(fut).to(dev).float()
    prog = torch.tensor([samples[i][1] / max(1.0, float(d["n_frames"][samples[i][0]])) for i in idx], device=dev)
    tot = torch.tensor([float(d["n_frames"][samples[i][0]]) / 30.0 for i in idx], device=dev)
    scal = torch.stack([prog, tot / 10.0], -1).float()
    g = torch.Generator(device=dev); g.manual_seed(args.seed + b0)
    with torch.no_grad():
        enc = lambda x: (model.vae.encode(x)[1] - model.lat_mean) / model.lat_std
        I_h, I_fut = enc(H), enc(F)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem_c, mv_c = model.adapter(text[0].float(), text[2])
            mem_u, mv_u = model.adapter(text_u[0].float(), text_u[2])
            I_H = ifl.sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                                    cfg_scale=args.cfg, generator=g, device=dev)
            hH_c = ifl.intent_hidden(model.hip, I_H, s_read, g, mem=mem_c, mem_valid=mv_c)
            hH_u = ifl.intent_hidden(model.hip, I_H, s_read, g, mem=mem_u, mem_valid=mv_u)
            I_pred = ifl.sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=args.num_steps,
                                       cfg_scale=args.cfg, generator=g, prefix=I_h, scalars=scal,
                                       extra=hH_c, extra_u=hH_u, device=dev)
    dist = lambda a, b: (a.float() - b.float()).flatten(1).norm(dim=-1).cpu().numpy()
    res["residual"].append(dist(I_pred, I_fut))
    res["ctl_change"].append(dist(I_h, I_fut))
    vel = (F[:, 1:] - F[:, :-1]).pow(2).mean(dim=(1, 2)).cpu().numpy()
    res["ctl_speed"].append(vel)
    res["is_b"].append(np.array([samples[i][2] for i in idx]))
    if b0 % (args.batch * 10) == 0:
        print(f"  {b0}/{len(order)}", flush=True)

M = {k: np.concatenate(v) for k, v in res.items()}
isb = M["is_b"].astype(bool)
M["random"] = np.random.RandomState(1).rand(len(isb))

# high-change interior = the top 20% of interior windows by ctl_change (where "motion changed" cannot help)
inter_change = M["ctl_change"][~isb]
thr = np.quantile(inter_change, 0.8)
hard = (~isb) & (M["ctl_change"] >= thr)

report = dict(ckpt=args.ckpt, step=int(step), text=args.text, n_boundary=int(isb.sum()),
              n_interior=int((~isb).sum()), n_hard_interior=int(hard.sum()),
              num_steps=args.num_steps, cfg=args.cfg, s_read=s_read,
              note="single sample per window; the IIP is a flow model so the residual carries sampling noise")
for name in ("residual", "ctl_change", "ctl_speed", "random"):
    report[f"auroc_vs_interior/{name}"] = auroc(M[name][isb], M[name][~isb])
    report[f"auroc_vs_HARD_interior/{name}"] = auroc(M[name][isb], M[name][hard])
for name in ("residual", "ctl_change"):
    report[f"mean_boundary/{name}"] = float(M[name][isb].mean())
    report[f"mean_interior/{name}"] = float(M[name][~isb].mean())

print("\n=== Phase 1: does the intent residual detect sub-instruction completion? ===")
print(f"ckpt step {step}  text={args.text}  {report['n_boundary']} boundary / {report['n_interior']} interior "
      f"({report['n_hard_interior']} high-change interior)")
print(f"{'score':<14}{'AUROC vs interior':>20}{'AUROC vs HIGH-CHANGE interior':>32}")
for name in ("residual", "ctl_change", "ctl_speed", "random"):
    print(f"{name:<14}{report[f'auroc_vs_interior/{name}']:>20.4f}{report[f'auroc_vs_HARD_interior/{name}']:>32.4f}")
r, c = report["auroc_vs_HARD_interior/residual"], report["auroc_vs_HARD_interior/ctl_change"]
ri = report["auroc_vs_interior/residual"]; ci = report["auroc_vs_interior/ctl_change"]
if ri < 0.55:
    v = f"residual {ri:.4f} is at chance -> it does NOT detect boundaries at all"
elif ri <= ci + 0.02:
    v = f"residual {ri:.4f} does not beat the motion-change control {ci:.4f} -> nothing beyond 'the motion changed'"
elif r <= 0.5:
    v = (f"residual {ri:.4f} beats the control on the easy split, but on high-change interiors it is "
         f"{r:.4f} (<0.5, i.e. anti-predictive) -> the signal is motion magnitude, not completion")
else:
    v = f"residual {r:.4f} vs control {c:.4f} on the decisive split -> completion-specific information"
print("\nverdict: " + v)
json.dump(report, open(out_path, "w"), indent=1)
print(f"wrote {out_path}")
