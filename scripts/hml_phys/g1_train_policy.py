"""Train the text-driven action policy for G1 (docs/04 line A).

Carries over exactly what was measured to work on the SMPL side and nothing else:
  * intent mechanism (HIP + IIP + frozen intent VAE), jointly trained with the policy, equal loss weights
  * conditioning augmentation: the intent hidden states are read at s ~ U(cond_aug, 1) in training, fixed at
    cond_aug_test in the closed loop -- the cure for the exposure bias between ground-truth and sampled intents
  * no sparse long history (four independent pieces of evidence that it hurts)
  * `sentence_xattn` text routing
Dropped: the 6-part input projection (MoGeFlow's, for its per-part VQ codebooks; our v3->v4 ablation was
0.52 -> 0.54, inside the noise) and everything VQ.

Rectified flow, project conventions: t = 1 clean, x0 prediction, loss in VELOCITY space with the
1/clamp(1-t, v_eps) factor, logit-normal t.  Only the ACTION channels of the future rows are generated; the
proprioceptive channels come from the simulator.

**What the policy is allowed to see of the future** (`--obs_future_state`).  The token carries proprioception
as well as the action, so a future row's proprio channels have to be masked or they leak the answer: the
tracker's action is very nearly a function of the state it produced, and a policy shown s_{t+1} can simply
invert the dynamics.  Measured on the first (unmasked) run at 100k steps: val action loss 0.0443 as trained,
3.52 with the future proprio zeroed -- an 80x gap, i.e. the policy had learned almost nothing that survives
into the closed loop.  The SMPL route A avoided this with `intent_flow.policy_input`, which zeroes every
non-action channel of the future rows; this is the G1 counterpart.

The default is `first`, not `none`, because of how the window lines up with the closed loop.  A row is
(state s_t, action a_t applied in it), so with H history rows the planner knows s_0..s_H and a_0..a_{H-1} and
must produce a_H..a_{H+F-1}.  The first generated row's state is therefore genuinely available -- it is what
the simulator has just returned -- while every later one is not.  `none` zeroes that row too (matches the SMPL
convention exactly, at the cost of hiding the current state) and `all` restores the leaking behaviour, kept
only so the gap can be re-measured.

Split rule (project CLAUDE.md §1): G1 has only train/val, so train trains and **val evaluates**.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys import intent_flow as ifl
from hml_phys.codeflow_flow import logit_normal_t
from hml_phys.g1_data import ACTION_DIM, G1_DIR, PROPRIO_DIM, TOKEN_DIM, G1WindowDataset, collate_g1
from hml_phys.g1_model import G1FlowPolicy, G1IntentPolicy
from hml_phys.intent_flow import build_state_elem, velocity_pair
from hml_phys.text_clip import ClipText

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--depth", default="3,6", help="double,single blocks")
ap.add_argument("--mlp_ratio", type=float, default=4.0)
ap.add_argument("--H", type=int, default=27, help="history frames (0.54 s @50fps, matching SMPL's 16 @30fps)")
ap.add_argument("--F", type=int, default=7, help="generated action frames (0.14 s, matching SMPL's 4 @30fps)")
ap.add_argument("--obs_future_state", default="first", choices=["none", "first", "all"],
                help="future rows whose PROPRIO channels the policy sees; see the module docstring")
ap.add_argument("--intent", type=int, default=1)
ap.add_argument("--vae", default="outputs/g1_intent_vae/best_val.pt")
ap.add_argument("--latent_stats", default=os.path.join(G1_DIR, "g1_intent_latent_stats.npz"))
ap.add_argument("--intent_dim", type=int, default=384); ap.add_argument("--intent_heads", type=int, default=6)
ap.add_argument("--intent_depth", type=int, default=4); ap.add_argument("--intent_mlp", type=float, default=1.5)
ap.add_argument("--cond_aug", type=float, default=0.5); ap.add_argument("--cond_aug_test", type=float, default=0.75)
ap.add_argument("--p_uncond", type=float, default=0.1); ap.add_argument("--v_eps", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--steps", type=int, default=200000)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr_schedule", default="cosine",
                                                                  choices=["const", "cosine"])
ap.add_argument("--lr_warmup", type=int, default=2000); ap.add_argument("--lr_final_ratio", type=float, default=0.01)
ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--stride", type=int, default=7)
ap.add_argument("--eval_every", type=int, default=10000); ap.add_argument("--eval_items", type=int, default=2048)
ap.add_argument("--ckpt_every", type=int, default=50000); ap.add_argument("--latest_every", type=int, default=2000)
ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_rollouts", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device("cuda")
dd = [int(x) for x in args.depth.split(",")]

policy_kw = dict(hidden_dim=args.hidden, num_heads=args.heads, depth_double=dd[0], depth_single=dd[1],
                 mlp_ratio=args.mlp_ratio, text_mode="sentence_xattn", text_cross_attention=True)
if args.intent:
    model = G1IntentPolicy(policy_kw, args.vae, args.latent_stats, intent_dim=args.intent_dim,
                           intent_heads=args.intent_heads, intent_depth=args.intent_depth,
                           intent_mlp=args.intent_mlp, device=dev)
    params = model.trainable(); net = model.policy
else:
    model = G1FlowPolicy(**policy_kw).to(dev); params = list(model.parameters()); net = model

mk = lambda sp, tr: G1WindowDataset(sp, H=args.H, F=args.F, stride=args.stride, intent=bool(args.intent),
                                    max_rollouts=args.max_rollouts, seed=args.seed)
train_ds, eval_ds = mk("train", True), mk("val", False)
sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_items, len(eval_ds)), replace=False)
coll = lambda b: collate_g1(b)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, persistent_workers=args.workers > 0, pin_memory=True, collate_fn=coll)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, sel), batch_size=args.batch, shuffle=False,
                         num_workers=2, collate_fn=coll)
clip_enc = ClipText()
E_TOK, E_POOL, E_LEN = clip_enc.encode([""])

log("args", json.dumps(vars(args)))
log(f"G1 policy: token {TOKEN_DIM} (proprio {PROPRIO_DIM} + action {ACTION_DIM}), window [{args.H} history | "
    f"{args.F} generated] @50fps; train {train_ds.n_rollouts} rollouts / {len(train_ds)} windows, "
    f"val {eval_ds.n_rollouts} / {len(eval_ds)} -> {len(sel)} fixed; "
    f"{(model.num_params() if args.intent else net.num_params())/1e6:.1f}M trainable")

opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
trainable_mod = model
EMA_SKIP = ("vae.",)
ema = {k: v.detach().clone().float() for k, v in trainable_mod.state_dict().items() if not k.startswith(EMA_SKIP)}
step, best = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    trainable_mod.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"]); step = ck["step"]
    best = ck.get("best", float("inf")); ema = {k: v.to(dev).float() for k, v in ck["ema"].items()}
    log(f"resumed from {args.resume} at step {step}")


def set_lr(s):
    if args.lr_schedule == "const":
        return args.lr
    if s <= args.lr_warmup:
        lr = args.lr * s / max(args.lr_warmup, 1)
    else:
        import math
        p = min(1.0, (s - args.lr_warmup) / max(args.steps - args.lr_warmup, 1))
        lr = args.lr * (args.lr_final_ratio + (1 - args.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
    for gparam in opt.param_groups:
        gparam["lr"] = lr
    return lr


ACT = slice(TOKEN_DIM - ACTION_DIM, TOKEN_DIM)


def mask_future_state(z, H):
    """zero the proprio channels of the future rows the closed loop cannot observe (see the docstring)."""
    if args.obs_future_state == "all":
        return z
    keep = H + (1 if args.obs_future_state == "first" else 0)
    z[:, keep:, :PROPRIO_DIM] = 0.0
    return z


def prepare(b, drop=None):
    x = b["x"].to(dev, non_blocking=True)
    mask = b["mask"].to(dev, non_blocking=True)
    B = x.shape[0]
    tok, pool, tl = clip_enc.encode([t if t else "" for t in b["text"]])
    tok = torch.from_numpy(tok).to(dev); pool = torch.from_numpy(pool).to(dev); tl = torch.from_numpy(tl).to(dev)
    if drop is not None and drop.any():                      # CFG dropout -> the CLIP("") features
        et = torch.from_numpy(E_TOK).to(dev); ep = torch.from_numpy(E_POOL).to(dev)
        tok = torch.where(drop[:, None, None], et.expand_as(tok), tok)
        pool = torch.where(drop[:, None], ep.expand_as(pool), pool)
        tl = torch.where(drop, torch.full_like(tl, int(E_LEN[0])), tl)
    scal = torch.stack([b["progress"].to(dev), b["total_len"].to(dev) / 10.0], -1).float()
    gen = (1.0 - mask)[..., None]                            # 1 on the rows whose actions are generated
    return x, mask, gen, (tok, pool, tl), scal


def intent_parts(b, text, scal, drop, generator=None, s_read=None):
    """HIP / IIP, unchanged from the SMPL route A: both flow-matched onto the frozen VAE's latents, and the
    policy reads their hidden states under conditioning augmentation."""
    B = b["x"].shape[0]
    I_H = model.encode_latent(b["holi"].to(dev))
    I_I = model.encode_latent(b["fut"].to(dev))
    I_h = model.encode_latent(b["hist"].to(dev))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem, mv = model.adapter(text[0].float(), text[2])
        t_h = logit_normal_t(B, dev, generator)
        z_h = build_state_elem(I_H, torch.ones_like(I_H), t_h, generator=generator)
        vh, vh_t = velocity_pair(model.hip(z_h, t_h, mem, mv)[0].float(), I_H, z_h, t_h, args.v_eps)
        s1 = s_read if s_read is not None else float(np.random.uniform(args.cond_aug, 1.0))
        hH = ifl.intent_hidden(model.hip, I_H, s1, generator, mem=mem, mem_valid=mv)
        t_i = logit_normal_t(B, dev, generator)
        z_i = build_state_elem(I_I, torch.ones_like(I_I), t_i, generator=generator)
        vi, vi_t = velocity_pair(model.iip(z_i, t_i, mem, mv, prefix_latent=I_h, scalars=scal,
                                           mem_extra=hH)[0].float(), I_I, z_i, t_i, args.v_eps)
        s2 = s_read if s_read is not None else float(np.random.uniform(args.cond_aug, 1.0))
        hI = ifl.intent_hidden(model.iip, I_I, s2, generator, mem=mem, mem_valid=mv, prefix_latent=I_h,
                               scalars=scal, mem_extra=hH)
    keep = (~drop) if drop is not None else torch.ones(B, dtype=torch.bool, device=dev)
    toks, tv = model.intent_tokens(hH, hI, keep)
    return ifl.latent_loss(vh, vh_t), ifl.latent_loss(vi, vi_t), toks, tv


def loss_on(b, t=None, generator=None, drop=None, s_read=None):
    x, mask, gen, text, scal = prepare(b, drop)
    B = x.shape[0]
    t = logit_normal_t(B, dev, generator) if t is None else t
    a0 = x[..., ACT]
    z_act = build_state_elem(a0, gen, t, generator=generator)
    z = x.clone(); z[..., ACT] = z_act
    z = mask_future_state(z, args.H)
    kw, extra = {}, {}
    if args.intent:
        l_hip, l_iip, toks, tv = intent_parts(b, text, scal, drop, generator, s_read)
        kw = dict(extra_tokens=toks, extra_valid=tv); extra = dict(hip=l_hip, iip=l_iip)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a_hat = net(z, mask, t, text[0], text[1], text[2], scal, **kw)
    v_hat, v = velocity_pair(a_hat.float(), a0, z_act, t, args.v_eps)
    denom = gen.sum().clamp_min(1.0) * ACTION_DIM
    per = {"act": ((v_hat - v) ** 2 * gen).sum() / denom}
    loss = per["act"]
    if args.intent:
        per.update(extra); loss = loss + extra["hip"] + extra["iip"]     # equal weights, as MIND trains them
    return loss, per


@torch.no_grad()
def evaluate():
    bak = {k: v.detach().clone() for k, v in trainable_mod.state_dict().items() if k in ema}
    trainable_mod.load_state_dict({k: v.to(bak[k].dtype) for k, v in ema.items()}, strict=False)
    trainable_mod.eval()
    tot, n = {}, 0
    for b in eval_loader:
        g = torch.Generator(device=dev); g.manual_seed(1234)
        loss, per = loss_on(b, t=torch.full((b["x"].shape[0],), 0.5, device=dev), generator=g,
                            s_read=args.cond_aug_test if args.intent else None)
        B = b["x"].shape[0]
        for k, v in dict(loss=float(loss), **{k: float(v) for k, v in per.items()}).items():
            tot[k] = tot.get(k, 0.0) + v * B
        n += B
    trainable_mod.load_state_dict(bak, strict=False); trainable_mod.train()
    return {k: v / n for k, v in tot.items()}


def save(path, tag):
    torch.save(dict(model={k: v for k, v in trainable_mod.state_dict().items() if k in ema}, ema=ema,
                    opt=opt.state_dict(), step=step, best=best, args=vars(args), policy_kw=policy_kw,
                    tag=tag), path)
    log(f"saved {path} ({tag}) at step {step}")


trainable_mod.train(); t0 = time.time(); run, n_run = {}, 0
it = iter(train_loader)
while step < args.steps:
    try:
        b = next(it)
    except StopIteration:
        it = iter(train_loader); b = next(it)
    cur_lr = set_lr(step + 1)
    drop = torch.rand(b["x"].shape[0], device=dev) < args.p_uncond
    loss, per = loss_on(b, drop=drop)
    if not torch.isfinite(loss):
        log("non-finite loss at step", step); save(os.path.join(args.out, f"nan_{step}.pt"), "non-finite"); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
    opt.step(); step += 1
    with torch.no_grad():
        d = args.ema
        sd = trainable_mod.state_dict()
        for k in ema:
            ema[k].mul_(d).add_(sd[k].detach().float(), alpha=1 - d)
    vals = dict(loss=float(loss), gn=float(gn), **{k: float(v) for k, v in per.items()})
    for k, v in vals.items():
        run[k] = run.get(k, 0.0) + v
    n_run += 1
    if step % args.log_every == 0:
        log(f"step {step} " + " ".join(f"{k}={v/n_run:.4f}" for k, v in run.items())
            + f" lr={cur_lr:.2e} {(time.time()-t0)/n_run*1000:.0f}ms/it")
        run, n_run, t0 = {}, 0, time.time()
    if step % args.eval_every == 0 or step == args.steps:
        ev = evaluate()
        log(f"[val] step {step} " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
        # Select on the ACTION loss, not the total.  HIP regresses text -> holistic latent, and BABEL's label
        # vocabulary is small enough that it overfits: measured on the intent run, val hip rose 2.55 (30k) ->
        # 2.87 (40k) -> 3.12 (50k) while val act kept falling 0.1540 -> 0.1511 -> 0.1503.  Selecting on the
        # total would therefore freeze the checkpoint early on a term that is not what the closed loop runs on.
        sel = ev["act"]
        if sel < best:
            best = sel; save(os.path.join(args.out, "best_val.pt"), f"best_val_act={best:.5f}")
    if args.latest_every and step % args.latest_every == 0:
        save(os.path.join(args.out, "latest.pt"), "latest")
    if step % args.ckpt_every == 0 or step == args.steps:
        save(os.path.join(args.out, f"step_{step}.pt"), "periodic")
log("done")
