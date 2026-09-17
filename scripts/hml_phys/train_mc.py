"""Train the physics policy DiT (docs/07 §10 / §12).

  loss = flow (velocity space, future frames, per-stream channel weights)
       + w_cons * root position/velocity consistency on the de-normalised x0 prediction (future frames):
         (root_trans[t+1] - root_trans[t]) * 30 vs root_trans_vel[t]   (smooth-L1)
  AdamW lr 1e-4 wd 0.01, grad clip 1.0, bf16 autocast, EMA 0.995 every 10 steps (MotionCraft defaults);
  text dropout 0.1 -> empty caption; checkpoints every --ckpt_every steps + best val loss (EMA weights evaluated).
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import PhysWindowDataset, TextCache, TokenStats, collate, load_env_constants, ROOT
from hml_phys.mc_model import PhysPolicyDiT
from hml_phys import flow as fl
from hml_phys.tokens import ROOT_SLICES

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--F", type=int, default=32)
ap.add_argument("--whole_sequence", action="store_true", help="v2: future = rest of the clip (variable length, capped at F)")
ap.add_argument("--hidden", type=int, default=768); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--root_depth", default="2,4"); ap.add_argument("--body_depth", default="3,6")
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--steps", type=int, default=300000)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--ema_decay", type=float, default=0.995); ap.add_argument("--ema_every", type=int, default=10)
ap.add_argument("--text_dropout", type=float, default=0.1)
ap.add_argument("--p_mean", type=float, default=-0.8); ap.add_argument("--p_std", type=float, default=0.8)
ap.add_argument("--v_eps", type=float, default=0.05)
ap.add_argument("--w_action", type=float, default=1.0); ap.add_argument("--w_root", type=float, default=1.0); ap.add_argument("--w_body", type=float, default=1.0)
ap.add_argument("--w_cons", type=float, default=0.01)
ap.add_argument("--p_rest", type=float, default=0.1); ap.add_argument("--p_neutral", type=float, default=0.05); ap.add_argument("--sigma_hist", type=float, default=0.0)
ap.add_argument("--ckpt_every", type=int, default=50000); ap.add_argument("--val_every", type=int, default=5000); ap.add_argument("--val_windows", type=int, default=2048)
ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--stats", default=os.path.join(ROOT, "token_stats.npz")); ap.add_argument("--env_constants", default=os.path.join(ROOT, "env_constants.npz"))
ap.add_argument("--resume", default=""); ap.add_argument("--max_clips", type=int, default=0, help="smoke: limit clips"); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
log("args", json.dumps(vars(args)))
dev = torch.device("cuda")
text_cache = TextCache()
stats = TokenStats(args.stats)
env_c = load_env_constants(args.env_constants)
empty_idx = text_cache.index.get("", -1)
train_ds = PhysWindowDataset("train", H=args.H, F=args.F, stats_path=args.stats, text_cache=text_cache, env_constants=env_c,
                             p_rest=args.p_rest, p_neutral=args.p_neutral, sigma_hist=args.sigma_hist, seed=args.seed, train=True, max_clips=args.max_clips,
                             whole_sequence=args.whole_sequence)
val_ds = PhysWindowDataset("val", H=args.H, F=args.F, stats_path=args.stats, text_cache=text_cache, env_constants=env_c, train=False, max_clips=args.max_clips,
                           whole_sequence=args.whole_sequence)
val_sel = np.random.RandomState(0).choice(len(val_ds), min(args.val_windows, len(val_ds)), replace=False)
log(f"train windows {len(train_ds)} (short clips skipped {train_ds.n_clips_short}), val windows {len(val_ds)} -> {len(val_sel)} fixed for validation")
coll = lambda b: collate(b, text_cache)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=coll, drop_last=True, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_sel), batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=coll)

rd = [int(x) for x in args.root_depth.split(",")]; bd = [int(x) for x in args.body_depth.split(",")]
model = PhysPolicyDiT(hidden_dim=args.hidden, num_heads=args.heads, root_depth_double=rd[0], root_depth_single=rd[1],
                      body_depth_double=bd[0], body_depth_single=bd[1], text_token_dim=text_cache.tokens.shape[2],
                      text_pooled_dim=text_cache.dim, max_text_tokens=text_cache.max_tokens).to(dev)
log(f"model params {model.num_params()/1e6:.1f}M")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
# channel weights: root stream all w_root; body stream w_body for state, w_action for the 69 action dims
w_root = torch.full((model.root_dim,), args.w_root, device=dev)
w_body = torch.full((model.body_dim,), args.w_body, device=dev); w_body[351:420] = args.w_action
root_mean = torch.from_numpy(stats.root_mean).to(dev); root_std = torch.from_numpy(stats.root_std).to(dev)
sl_tr, sl_tv = ROOT_SLICES["root_trans"], ROOT_SLICES["root_trans_vel"]
step, best_val = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); ema = {k: v.float() for k, v in ck["ema"].items()}
    step, best_val = ck["step"], ck.get("best_val", float("inf")); log(f"resumed from {args.resume} at step {step}")


def to_dev(b):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def apply_text_dropout(b, p):
    if p <= 0 or empty_idx < 0:
        return b
    drop = torch.rand(b["root"].shape[0], device=dev) < p
    tok, po, ln = text_cache.get(empty_idx)
    b["text_tokens"][drop] = torch.from_numpy(tok).to(dev); b["text_pooled"][drop] = torch.from_numpy(po).to(dev); b["text_len"][drop] = ln
    return b


def compute_loss(b, train=True, generator=None):
    root, body, mask, valid = b["root"], b["body"], b["observed_mask"], b["valid"]
    B = root.shape[0]
    t = fl.sample_t(B, dev, args.p_mean, args.p_std, generator=generator)
    zr, _, _ = fl.build_state(root, mask, t, generator=generator); zb, _, _ = fl.build_state(body, mask, t, generator=generator)
    scalars = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xr, xb = model(zr, zb, mask, t, b["text_tokens"], b["text_pooled"], b["text_len"], scalars, valid=valid)
    xr, xb = xr.float(), xb.float()
    vr_hat, vr = fl.velocity_pair(xr, root, zr, t, args.v_eps); vb_hat, vb = fl.velocity_pair(xb, body, zb, t, args.v_eps)
    l_root = fl.masked_mse(vr_hat, vr, mask, w_root, valid); l_body = fl.masked_mse(vb_hat, vb, mask, w_body, valid)
    # root position/velocity consistency on de-normalised x0 (future frames only)
    xr_un = xr * root_std + root_mean
    d_pos = (xr_un[:, 1:, sl_tr[0]:sl_tr[1]] - xr_un[:, :-1, sl_tr[0]:sl_tr[1]]) * 30.0
    vel = xr_un[:, :-1, sl_tv[0]:sl_tv[1]]
    fut_pair = ((1 - mask[:, 1:]) * (1 - mask[:, :-1]) * valid[:, 1:] * valid[:, :-1])
    l_cons = fl.masked_smooth_l1(d_pos, vel, fut_pair)
    loss = l_root + l_body + args.w_cons * l_cons
    return loss, dict(flow_root=l_root.item(), flow_body=l_body.item(), cons=l_cons.item())


@torch.no_grad()
def validate():
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(model_state[k].dtype) for k, v in ema.items()}); model.eval()
    tot, n, parts = 0.0, 0, {}
    g = torch.Generator(device=dev); g.manual_seed(0)  # deterministic t / noise -> comparable val losses across steps
    for b in val_loader:
        b = to_dev(b); loss, p = compute_loss(b, train=False, generator=g); tot += loss.item() * b["root"].shape[0]; n += b["root"].shape[0]
        for k, v in p.items(): parts[k] = parts.get(k, 0) + v * b["root"].shape[0]
    model.load_state_dict(model_state); model.train()
    return tot / n, {k: v / n for k, v in parts.items()}


def save(path, tag):
    torch.save(dict(model=model.state_dict(), ema=ema, opt=opt.state_dict(), step=step, best_val=best_val, args=vars(args),
                    stats=dict(root_mean=stats.root_mean, root_std=stats.root_std, body_mean=stats.body_mean, body_std=stats.body_std),
                    env_constants={k: np.asarray(v) for k, v in env_c.items()}, tag=tag), path)
    log(f"saved {path} ({tag}) at step {step}")


model.train(); t0 = time.time(); it = iter(train_loader); run = {}
while step < args.steps:
    try:
        b = next(it)
    except StopIteration:
        it = iter(train_loader); b = next(it)
    b = apply_text_dropout(to_dev(b), args.text_dropout)
    loss, parts = compute_loss(b)
    if not torch.isfinite(loss):
        log("non-finite loss at step", step); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step(); step += 1
    if args.ema_every > 0 and step % args.ema_every == 0:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(args.ema_decay).add_(v.float(), alpha=1 - args.ema_decay)
                else: ema[k].copy_(v)
    for k, v in parts.items(): run[k] = run.get(k, 0) + v
    run["loss"] = run.get("loss", 0) + loss.item(); run["gn"] = run.get("gn", 0) + float(gn)
    if step % args.log_every == 0:
        log(f"step {step} " + " ".join(f"{k}={v/args.log_every:.4f}" for k, v in run.items()) + f" {(time.time()-t0)/args.log_every*1000:.0f}ms/it"); run = {}; t0 = time.time()
    if step % args.val_every == 0 or step == args.steps:
        vl, vp = validate(); log(f"[val] step {step} loss={vl:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in vp.items()))
        if vl < best_val:
            best_val = vl; save(os.path.join(args.out, "best_val.pt"), f"best_val={vl:.4f}")
    if step % args.ckpt_every == 0 or step == args.steps:
        save(os.path.join(args.out, f"step_{step}.pt"), "periodic")
log("done")
