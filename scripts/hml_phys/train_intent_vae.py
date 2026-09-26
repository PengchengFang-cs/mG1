"""Train MIND's intent VAE on our physics states (docs/07 §20).

Model / loss: hml_phys/intent_vae.py (MotionStreamer causal TAE + sigma-VAE loss; MIND: down 4, latent 32,
lambda_KL 1e-5). Optimiser: MotionStreamer's (AdamW lr 5e-5, betas 0.9/0.99, no weight decay, linear warm-up
1000 iterations, no gradient clipping). Each batch = B windows x {history, immediate future, holistic}.
The periodic loss curve and the best checkpoint use the TEST split (project CLAUDE.md: val is banned);
test reconstruction is deterministic (decode the posterior mean).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import load_env_constants, ROOT
from hml_phys.intent_vae import IntentVAE, sigma_vae_nll, kl_loss, VAE_STATE_DIM
from hml_phys.intent_data import IntentSeqDataset
from hml_phys.tokens import ROOT_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--L", type=int, default=16, help="sequence length (MIND: history 16, immediate 16, holistic 16)")
ap.add_argument("--latent_dim", type=int, default=32); ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--down_t", type=int, default=2, help="2 stride-2 stages = 4x temporal downsampling (MIND)")
ap.add_argument("--depth", type=int, default=3); ap.add_argument("--dilation", type=int, default=3)
ap.add_argument("--kl", type=float, default=1e-5, help="lambda_KL (MIND appendix A)")
ap.add_argument("--root_loss", type=float, default=0.0, help="MotionStreamer's extra root NLL weight (its default 7; MIND: not stated)")
ap.add_argument("--holi_aug", type=int, default=0, help="v2: random sub-span + phase for the holistic training sequence (docs/07 §20.1)")
ap.add_argument("--p_holi_exact", type=float, default=0.25, help="v2: prob. of the exact whole-clip holistic sampling")
ap.add_argument("--batch", type=int, default=128, help="windows per batch; each gives 3 sequences")
ap.add_argument("--iters", type=int, default=100000); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--milestones", default="", help="comma-separated MultiStepLR milestones (MotionStreamer gamma 0.05)")
ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--eval_items", type=int, default=4096)
ap.add_argument("--ckpt_every", type=int, default=25000); ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--eval_split", default="test", help="project CLAUDE.md: test only, val is banned")
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max_clips", type=int, default=0)
ap.add_argument("--stats", default=os.path.join(ROOT, "token_stats_v3.npz"))
ap.add_argument("--env_constants", default=os.path.join(ROOT, "env_constants.npz"))
ap.add_argument("--seed", type=int, default=123)
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
train_ds = IntentSeqDataset("train", args.stats, env_c, train=True, L=args.L, max_clips=args.max_clips, seed=args.seed,
                            holi_aug=bool(args.holi_aug), p_holi_exact=args.p_holi_exact)
eval_ds = IntentSeqDataset(args.eval_split, args.stats, env_c, train=False, L=args.L, max_clips=args.max_clips)
eval_sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_items, len(eval_ds)), replace=False)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0, pin_memory=True)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, eval_sel), batch_size=args.batch, shuffle=False, num_workers=4)
stats = train_ds.stats
vae = IntentVAE(input_dim=VAE_STATE_DIM, hidden_size=args.width, width=args.width, down_t=args.down_t,
                depth=args.depth, dilation_growth_rate=args.dilation, latent_dim=args.latent_dim).to(dev)
args.input_dim = VAE_STATE_DIM
log("args", json.dumps(vars(args)))
log(f"train windows {len(train_ds)} ({len(train_ds.holi)} clips), {args.eval_split} windows {len(eval_ds)} -> {len(eval_sel)} fixed; "
    f"VAE {vae.num_params()/1e6:.1f}M params, {args.L} frames -> {args.L // vae.down} x {args.latent_dim} latents")
opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
ms = [int(x) for x in args.milestones.split(",") if x]
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=args.gamma) if ms else None

# channel groups of the 366-d state and the physical de-normalisation used by the test metrics
GROUPS = dict(root=(0, 15), local_pos=(15, 87), local_vel=(87, 159), dof_pose6d=(159, 297), dof_vel=(297, 366))
mean = torch.from_numpy(np.concatenate([stats.root_mean, stats.body_mean[:351]])).to(dev)
std = torch.from_numpy(np.concatenate([stats.root_std, stats.body_std[:351]])).to(dev)
KINDS = ("hist", "fut", "holi")


@torch.no_grad()
def evaluate():
    vae.eval()
    acc = {}
    mus = []
    for b in eval_loader:
        for k in KINDS:
            x = b[k].to(dev)
            _, mu, logvar = vae.encode(x)
            rec = vae.decode(mu)                       # deterministic reconstruction
            d = (rec - x) ** 2
            xs, rs = x * std + mean, rec * std + mean  # physical units
            jp = (xs[..., 15:87] - rs[..., 15:87]).view(*x.shape[:2], 24, 3).norm(dim=-1).mean() * 1000   # mm
            rp = (xs[..., 0:3] - rs[..., 0:3]).norm(dim=-1).mean() * 1000                              # mm
            rh = (xs[..., 2] - rs[..., 2]).abs().mean() * 1000                                         # mm
            lv = (xs[..., 87:159] - rs[..., 87:159]).view(*x.shape[:2], 24, 3).norm(dim=-1).mean() * 30  # local_vel is m/frame -> m/s
            vals = dict(mse=d.mean(), jpe_mm=jp, root_pos_mm=rp, root_h_mm=rh, vel_ms=lv,
                        kl=kl_loss(mu, logvar), var=logvar.exp().mean(),
                        **{f"g_{g}": d[..., a:e].mean() for g, (a, e) in GROUPS.items()})
            n = x.shape[0]
            for name, v in vals.items():
                acc[(k, name)] = acc.get((k, name), 0.0) + float(v) * n
            acc[(k, "n")] = acc.get((k, "n"), 0) + n
            mus.append(mu.reshape(-1, mu.shape[-1]).float().cpu())
    vae.train()
    out = {k: {name: acc[(k, name)] / acc[(k, "n")] for (kk, name) in acc if kk == k and name != "n"} for k in KINDS}
    m = torch.cat(mus)
    out["latent"] = dict(mu_abs=float(m.abs().mean()), mu_std=float(m.std(0).mean()),
                         active_units=int((m.var(0) > 0.01).sum()), dims=int(m.shape[1]))
    out["select"] = float(np.mean([out[k]["mse"] for k in KINDS]))
    return out


def save(path, tag, it, ev=None):
    torch.save(dict(model=vae.state_dict(), opt=opt.state_dict(), step=it, args=vars(args), tag=tag, eval=ev,
                    stats_path=args.stats), path)
    log(f"saved {path} ({tag}) at iter {it}")


vae.train(); it, best, t0, run = 0, float("inf"), time.time(), {}
data_iter = iter(train_loader)
while it < args.iters:
    try:
        b = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader); b = next(data_iter)
    it += 1
    if it <= args.warmup:                              # MotionStreamer's linear warm-up
        for g in opt.param_groups:
            g["lr"] = args.lr * it / (args.warmup + 1)
    x = torch.cat([b[k] for k in KINDS]).to(dev, non_blocking=True)
    pred, mu, logvar = vae(x)
    l_rec = sigma_vae_nll(pred, x)
    l_kl = kl_loss(mu, logvar)
    loss = l_rec + args.kl * l_kl
    if args.root_loss > 0:
        loss = loss + args.root_loss * sigma_vae_nll(pred, x, dims=slice(0, ROOT_DIM))
    if not torch.isfinite(loss):
        log("non-finite loss at iter", it); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if sched is not None and it > args.warmup:
        sched.step()
    with torch.no_grad():
        B = b["hist"].shape[0]
        se = ((pred - x) ** 2).mean(dim=(1, 2))
        se_root = ((pred[..., :ROOT_DIM] - x[..., :ROOT_DIM]) ** 2).mean()
        vals = dict(nll_per_elem=l_rec.item() / x.numel(), kl=l_kl.item(), mse=se.mean().item(), mse_root=se_root.item(),
                    mse_hist=se[:B].mean().item(), mse_fut=se[B:2 * B].mean().item(), mse_holi=se[2 * B:].mean().item(),
                    mu_abs=mu.abs().mean().item(), var=logvar.exp().mean().item())
    for k, v in vals.items():
        run[k] = run.get(k, 0.0) + v
    if it % args.log_every == 0:
        log(f"iter {it} " + " ".join(f"{k}={v/args.log_every:.4g}" for k, v in run.items())
            + f" lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/args.log_every*1000:.0f}ms/it")
        run, t0 = {}, time.time()
    if it % args.eval_every == 0 or it == args.iters:
        ev = evaluate()
        log(f"[{args.eval_split}] iter {it} select(mean mse)={ev['select']:.5f} latent={json.dumps(ev['latent'])}")
        for k in KINDS:
            log(f"   {k:4s} " + " ".join(f"{n}={v:.4g}" for n, v in ev[k].items()))
        if ev["select"] < best:
            best = ev["select"]; save(os.path.join(args.out, f"best_{args.eval_split}.pt"), f"best_{args.eval_split}={best:.5f}", it, ev)
    if it % args.ckpt_every == 0 or it == args.iters:
        save(os.path.join(args.out, f"iter_{it}.pt"), "periodic", it)
log("done")
