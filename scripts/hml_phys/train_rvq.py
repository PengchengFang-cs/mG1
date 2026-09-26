"""Train MoMask's 6-layer residual VQ-VAE (docs/08) on our physics tokens.

Model: hml_phys/rvq.py. Default --structure whole is MoMask's original RVQ-VAE (one encoder, one 6-layer
residual quantiser, one decoder, over every channel of the variant); --structure part is the part-structured
variant kept for reference. MoGeFlow's own part-VQ is NOT used (user, 2026-09-21).
Objective / optimiser: MoMask's (models/vq/vq_trainer.py, options/vq_option.py) -- smooth-L1 reconstruction
+ 0.5 * smooth-L1 on the geometric channels + 0.02 * commitment; AdamW lr 2e-4 with 2000 warm-up iterations and a
MultiStepLR decay; batch 256, window 64 frames.
Three channel variants (--variant): action (69 PD-action channels), token (all 435), state (the 366 non-action ones).
The periodic loss curve and the best checkpoint use the TEST split only (project CLAUDE.md §1).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import TokenStats, load_env_constants, ROOT
from hml_phys.rvq import PartRVQVAE, rvq_losses
from hml_phys.rvq_data import RVQWindowDataset, variant_parts        # owned by the data agent

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--variant", required=True, choices=["action", "token", "state"])
ap.add_argument("--window", type=int, default=64, help="MoMask: 64 frames")
ap.add_argument("--nb_code", type=int, default=2048); ap.add_argument("--code_dim", type=int, default=512)
ap.add_argument("--n_quant", type=int, default=6, help="6 residual layers (user's choice, MoMask style)")
ap.add_argument("--structure", default="whole", choices=["whole", "part"],
                help="whole = MoMask original RVQ-VAE (one encoder + one 6-layer RVQ + one decoder over every channel); "
                     "part = the part-structured variant (six parts, each with its own 6-layer RVQ)")
ap.add_argument("--width", type=int, default=512); ap.add_argument("--down_t", type=int, default=2)
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--dilation", type=int, default=3)
ap.add_argument("--shared_codebook", type=int, default=0); ap.add_argument("--dropout_prob", type=float, default=0.2)
ap.add_argument("--mu", type=float, default=0.99)
ap.add_argument("--commit", type=float, default=0.02); ap.add_argument("--w_explicit", type=float, default=0.5)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--iters", type=int, default=200000)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--warmup", type=int, default=2000)
ap.add_argument("--milestones", default="150000"); ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--eval_windows", type=int, default=2048)
ap.add_argument("--ckpt_every", type=int, default=50000); ap.add_argument("--latest_every", type=int, default=2000); ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--eval_split", default="test"); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--stats", default=os.path.join(ROOT, "token_stats_v3.npz"))
ap.add_argument("--env_constants", default=os.path.join(ROOT, "env_constants.npz"))
ap.add_argument("--max_clips", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="")
args = ap.parse_args()
assert args.eval_split != "val", "val split is banned in this project"

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device("cuda")
env_c = load_env_constants(args.env_constants)
train_ds = RVQWindowDataset("train", variant=args.variant, window=args.window, stats_path=args.stats,
                            env_constants=env_c, train=True, max_clips=args.max_clips, seed=args.seed)
eval_ds = RVQWindowDataset(args.eval_split, variant=args.variant, window=args.window, stats_path=args.stats,
                           env_constants=env_c, train=False, max_clips=args.max_clips)
eval_sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_windows, len(eval_ds)), replace=False)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0, pin_memory=True)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, eval_sel), batch_size=args.batch, shuffle=False, num_workers=4)

parts, explicit_idx = variant_parts(args.variant)          # local channel indices per part; geometric channels
if args.structure == "whole":                                # MoMask original: no part axis at all
    parts = [np.arange(train_ds.n_channels, dtype=np.int64)]
model = PartRVQVAE(train_ds.n_channels, parts, width=args.width, down_t=args.down_t, depth=args.depth,
                   dilation=args.dilation, code_dim=args.code_dim, nb_code=args.nb_code, n_quant=args.n_quant,
                   shared_codebook=bool(args.shared_codebook), dropout_prob=args.dropout_prob, mu=args.mu).to(dev)
args.n_channels = int(train_ds.n_channels)
log("args", json.dumps(vars(args)))
log(f"variant {args.variant} [{args.structure}]: {train_ds.n_channels} channels in {len(parts)} group(s) {[len(p) for p in parts]}; "
    f"train windows {len(train_ds)}, {args.eval_split} windows {len(eval_ds)} -> {len(eval_sel)} fixed; "
    f"{args.window} frames -> {args.window // model.down} codes x {args.n_quant} layers x {args.nb_code} entries; "
    f"model {model.num_params()/1e6:.1f}M params")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
ms = [int(x) for x in args.milestones.split(",") if x]
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=args.gamma) if ms else None
expl = torch.from_numpy(np.asarray(explicit_idx)).to(dev) if explicit_idx is not None and len(explicit_idx) else None
it, best = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it = ck["step"]; best = ck.get("best", float("inf"))
    if sched is not None and ck.get("sched"): sched.load_state_dict(ck["sched"])
    log(f"resumed from {args.resume} at iter {it}")


@torch.no_grad()
def evaluate():
    model.eval()
    tot = {}
    n = 0
    for x in eval_loader:
        x = x.to(dev, non_blocking=True)
        rec, st = model(x)
        b = x.shape[0]; n += b
        vals = dict(mse=float(((rec - x) ** 2).mean()), l1=float((rec - x).abs().mean()),
                    perplexity=float(st["perplexity"]), commit=float(st["commit"]))
        for q in range(st["perplexity_per_layer"].shape[-1]):   # is the 2048-entry codebook actually used, layer by
            vals[f"ppl_L{q+1}"] = float(st["perplexity_per_layer"][..., q].mean())   # layer, and is one layer's
            vals[f"commit_L{q+1}"] = float(st["commit_per_layer"][..., q].mean())    # residual scale dominating?
        for k, v in vals.items():
            tot[k] = tot.get(k, 0.0) + v * b
    model.train()
    return {k: v / n for k, v in tot.items()}


def save(path, tag):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict() if sched else None,
                    step=it, best=best, args=vars(args), parts=[np.asarray(p) for p in parts],
                    explicit_idx=np.asarray(explicit_idx) if explicit_idx is not None else None, tag=tag), path)
    log(f"saved {path} ({tag}) at iter {it}")


model.train(); t0 = time.time(); run, n_run = {}, 0
data_iter = iter(train_loader)
while it < args.iters:
    try:
        x = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader); x = next(data_iter)
    it += 1
    if it <= args.warmup:                                   # MoMask's linear warm-up
        for g in opt.param_groups:
            g["lr"] = args.lr * it / (args.warmup + 1)
    x = x.to(dev, non_blocking=True)
    rec, st = model(x)
    loss, parts_loss = rvq_losses(rec, x, st, explicit_idx=expl, w_explicit=args.w_explicit, w_commit=args.commit)
    if not torch.isfinite(loss):
        log("non-finite loss at iter", it, {k: float(v) for k, v in parts_loss.items()})
        save(os.path.join(args.out, f"nan_iter_{it}.pt"), "non-finite loss")   # keep the state that blew up
        raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if sched is not None and it > args.warmup:
        sched.step()
    with torch.no_grad():
        vals = dict(loss=float(loss), rec=float(parts_loss["rec"]), expl=float(parts_loss["explicit"]),
                    commit=float(parts_loss["commit"]), perp=float(st["perplexity"]),
                    mse=float(((rec - x) ** 2).mean()))
    for k, v in vals.items():
        run[k] = run.get(k, 0.0) + v
    n_run += 1                                              # counted, not assumed: after --resume the first window
    if it % args.log_every == 0:                            # is shorter than log_every
        log(f"iter {it} " + " ".join(f"{k}={v/n_run:.4f}" for k, v in run.items())
            + f" lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/n_run*1000:.0f}ms/it")
        run, n_run, t0 = {}, 0, time.time()
    if it % args.eval_every == 0 or it == args.iters:
        ev = evaluate()
        log(f"[{args.eval_split}] iter {it} " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
        if ev["mse"] < best:
            best = ev["mse"]; save(os.path.join(args.out, f"best_{args.eval_split}.pt"), f"best_{args.eval_split}={best:.5f}")
    if args.latest_every and it % args.latest_every == 0:   # MoMask keeps a rolling `latest` (vq_option.py:54);
        save(os.path.join(args.out, "latest.pt"), "latest")  # without one, a crash costs up to ckpt_every steps
    if it % args.ckpt_every == 0 or it == args.iters:
        save(os.path.join(args.out, f"iter_{it}.pt"), "periodic")
log("done")
