"""Train the intent VAE for G1 (docs/04 line A; MIND §4.2, docs/07 §20).

The SMPL intent VAE cannot be reused: it encodes 366-d SMPL states, G1's proprioception is 67-d
(root_lin_vel_b 3 | root_ang_vel_b 3 | projected_gravity_b 3 | joint_pos 29 | joint_vel 29).  `IntentVAE`
already takes `input_dim`, so only the data side changes.

Objective is MotionStreamer's / MIND's unchanged: optimal-sigma Gaussian NLL + lambda_KL * KL, lambda_KL = 1e-5,
AdamW 5e-5, linear warm-up, no weight decay, no gradient clipping.  The VAE is trained ONCE and then frozen —
it only exists to define the intent space that HIP/IIP regress into.

Sequence length is 28 frames (0.56 s at 50 fps), matching the SMPL side's 16/30 = 0.53 s and divisible by the
4x temporal downsampling.  Each window contributes three sequences (history / immediate future / holistic),
exactly as on the SMPL side.

Split rule (project CLAUDE.md §1): the G1 dataset has only train/val, so train trains and **val evaluates**.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.g1_data import G1_DIR, L_INTENT, PROPRIO_DIM, G1WindowDataset, collate_g1
from hml_phys.intent_vae import IntentVAE, kl_loss, sigma_vae_nll

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--latent_dim", type=int, default=32, help="MIND ablation: 32 beats 16/64/128/256")
ap.add_argument("--width", type=int, default=1024); ap.add_argument("--down_t", type=int, default=2)
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--dilation", type=int, default=3)
ap.add_argument("--kl", type=float, default=1e-5, help="lambda_KL (MIND appendix A)")
ap.add_argument("--batch", type=int, default=128, help="windows per batch; each yields 3 sequences")
ap.add_argument("--iters", type=int, default=100000); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--stride", type=int, default=13, help="window stride; 13 keeps the epoch a sane size")
ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--eval_items", type=int, default=4096)
ap.add_argument("--ckpt_every", type=int, default=25000); ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max_rollouts", type=int, default=0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", default="")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device("cuda")

mk = lambda sp: G1WindowDataset(sp, intent=True, stride=args.stride, max_rollouts=args.max_rollouts, seed=args.seed)
train_ds, eval_ds = mk("train"), mk("val")
sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_items, len(eval_ds)), replace=False)
coll = lambda b: collate_g1(b)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, persistent_workers=args.workers > 0, pin_memory=True, collate_fn=coll)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, sel), batch_size=args.batch, shuffle=False,
                         num_workers=2, collate_fn=coll)

model = IntentVAE(input_dim=PROPRIO_DIM, hidden_size=args.width, width=args.width, down_t=args.down_t,
                  stride_t=2, depth=args.depth, dilation_growth_rate=args.dilation,
                  latent_dim=args.latent_dim).to(dev)
args.input_dim, args.L = PROPRIO_DIM, L_INTENT
log("args", json.dumps(vars(args)))
log(f"G1 intent VAE: {PROPRIO_DIM}-d states, L={L_INTENT} frames (0.56 s @50fps) -> "
    f"{L_INTENT // (2 ** args.down_t)} x {args.latent_dim} latent; "
    f"train {train_ds.n_rollouts} rollouts / {len(train_ds)} windows, val {eval_ds.n_rollouts} / {len(eval_ds)} "
    f"-> {len(sel)} fixed; {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
step, best = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    best = ck.get("best", float("inf")); log(f"resumed from {args.resume} at step {step}")


def three_streams(b):
    """the batch's history / immediate-future / holistic sequences, stacked into one pass."""
    return torch.cat([b["hist"], b["fut"], b["holi"]], 0).to(dev, non_blocking=True)


@torch.no_grad()
def evaluate():
    model.eval(); tot, n = {}, 0
    for b in eval_loader:
        x = three_streams(b)
        pred, mu, logvar = model(x)
        vals = dict(rec=float(sigma_vae_nll(pred, x)) / x.numel(), kl=float(kl_loss(mu, logvar)),
                    mse=float(((pred - x) ** 2).mean()), var=float(logvar.exp().mean()))
        for k, v in vals.items():
            tot[k] = tot.get(k, 0.0) + v * x.shape[0]
        n += x.shape[0]
    model.train()
    return {k: v / n for k, v in tot.items()}


def save(path, tag):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best,
                    args=vars(args), tag=tag), path)
    log(f"saved {path} ({tag}) at step {step}")


model.train(); t0 = time.time(); run, n_run = {}, 0
it = iter(train_loader)
while step < args.iters:
    try:
        b = next(it)
    except StopIteration:
        it = iter(train_loader); b = next(it)
    step += 1
    for gparam in opt.param_groups:
        gparam["lr"] = args.lr * min(1.0, step / max(args.warmup, 1))
    x = three_streams(b)
    pred, mu, logvar = model(x)
    l_rec, l_kl = sigma_vae_nll(pred, x), kl_loss(mu, logvar)
    loss = l_rec + args.kl * l_kl
    if not torch.isfinite(loss):
        log("non-finite loss at step", step); save(os.path.join(args.out, f"nan_{step}.pt"), "non-finite"); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    vals = dict(loss=float(loss) / x.numel(), rec=float(l_rec) / x.numel(), kl=float(l_kl),
                mse=float(((pred - x) ** 2).mean()))
    for k, v in vals.items():
        run[k] = run.get(k, 0.0) + v
    n_run += 1
    if step % args.log_every == 0:
        log(f"step {step} " + " ".join(f"{k}={v/n_run:.5f}" for k, v in run.items())
            + f" lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/n_run*1000:.0f}ms/it")
        run, n_run, t0 = {}, 0, time.time()
    if step % args.eval_every == 0 or step == args.iters:
        ev = evaluate()
        log(f"[val] step {step} " + " ".join(f"{k}={v:.5f}" for k, v in ev.items()))
        if ev["mse"] < best:
            best = ev["mse"]; save(os.path.join(args.out, "best_val.pt"), f"best_val={best:.6f}")
    if step % args.ckpt_every == 0 or step == args.iters:
        save(os.path.join(args.out, f"step_{step}.pt"), "periodic")
log("done")
