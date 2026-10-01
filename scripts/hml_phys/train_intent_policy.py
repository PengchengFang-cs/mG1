"""MIND intent arch (docs/07 §21): joint end-to-end training of HIP + IIP + the action-only policy on frozen intent-VAE latents.

  L = L_HIP + L_IIP + L_ADiT   (MIND eq. 5, equal weights)
  HIP : text -> holistic intent latents (4 x 32)
  IIP : text + history intent + HIP hidden -> immediate intent latents (4 x 32)
  ADiT: part-structured v5 policy generating only the F_act future ACTION rows; intent hidden states (HIP, IIP; read
        from a clean-latent forward at t = 1) enter as 4 + 4 extra tokens of its joint-attention text stream.
All three use x0 prediction with the velocity-space loss and logit-normal t. Text dropout 0.1 with one mask for the
three models; a dropped sample also carries no intent tokens (= the CFG unconditional state of the policy).
Periodic test-split denoising loss (L_HIP, L_IIP, L_ADiT reported separately) and the lowest-loss checkpoint use the
TEST split only (project CLAUDE.md §1).
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import TextCache, TokenStats, load_env_constants, ROOT
from hml_phys.intent_policy_data import IntentPolicyDataset, collate_intent
from hml_phys.intent_model import IntentPolicy
from hml_phys.intent_vae import load_intent_vae
from hml_phys import flow as fl
from hml_phys import intent_flow as ifl

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--F_act", type=int, default=4, help="MIND: action horizon 4")
ap.add_argument("--H_sparse", type=int, default=16); ap.add_argument("--L_max", type=int, default=154); ap.add_argument("--alpha", type=float, default=3.0)
ap.add_argument("--p_no_sparse", type=float, default=0.15); ap.add_argument("--alpha_min", type=float, default=0.0); ap.add_argument("--alpha_max", type=float, default=5.0)
ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--heads", type=int, default=12); ap.add_argument("--depth", default="3,6")
ap.add_argument("--mlp_ratio", type=float, default=4.0)
ap.add_argument("--intent_dim", type=int, default=384); ap.add_argument("--intent_heads", type=int, default=6)
ap.add_argument("--intent_depth", type=int, default=4); ap.add_argument("--intent_mlp", type=float, default=1.5, help="SwiGLU ratio; 1.5 keeps the total <= 100M (docs/07 §21.7 实施记录)")
ap.add_argument("--hip_aug", type=int, default=0, help="§21.8 a: VAE-v2-style random sub-span / phase for the holistic targets")
ap.add_argument("--span_scalars", type=int, default=0, help="§21.8 d: progress / total length of tagged captions refer to the tagged span")
ap.add_argument("--cond_aug", type=float, default=0.0, help="§21.8 b: read intent hidden states at s ~ U(cond_aug, 1) in training (0 = clean, §21.4-4)")
ap.add_argument("--cond_aug_test", type=float, default=0.75, help="§21.8 b: fixed read-out level for sampled intents at test time")
ap.add_argument("--select", default="sum", choices=["sum", "chain"], help="§21.8 c: best_test on the sum of the three losses, or on the test-chain action loss")
ap.add_argument("--vae", default="outputs/intent_vae_v2/best_test.pt")
ap.add_argument("--latent_stats", default=os.path.join(ROOT, "intent_latent_stats_vae_v2.npz"))
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--steps", type=int, default=200000)
ap.add_argument("--lr_schedule", default="const", choices=["const", "cosine"],
                help="cosine: linear warm-up then half-cosine decay to lr*lr_final_ratio, identical in form to "
                     "MoGeFlow's half_cosine (vendor_mogeflow/models/codeflow/trainer.py:390-396, README:183-184) "
                     "and MotionStreamer's WarmupCosineDecayScheduler. The constant 1e-4 "
                     "used so far peaks R@1 at ~100k and then degrades it (docs/06), which decay may prevent.")
ap.add_argument("--lr_final_ratio", type=float, default=0.01)
ap.add_argument("--lr_warmup", type=int, default=2000, help="MoGeFlow / MoMask / MotionStreamer all use 2000")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--ema_decay", type=float, default=0.995); ap.add_argument("--ema_every", type=int, default=10)
ap.add_argument("--text_dropout", type=float, default=0.1)
ap.add_argument("--p_mean", type=float, default=-0.8); ap.add_argument("--p_std", type=float, default=0.8); ap.add_argument("--v_eps", type=float, default=0.05)
ap.add_argument("--p_rest", type=float, default=0.1); ap.add_argument("--p_neutral", type=float, default=0.05)
ap.add_argument("--ckpt_every", type=int, default=50000); ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--eval_windows", type=int, default=2048)
ap.add_argument("--eval_split", default="test", help="project CLAUDE.md §1: test only, val is banned")
ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--stats", default=os.path.join(ROOT, "token_stats_v3.npz")); ap.add_argument("--env_constants", default=os.path.join(ROOT, "env_constants.npz"))
ap.add_argument("--resume", default=""); ap.add_argument("--max_clips", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
assert args.eval_split != "val", "val split is banned in this project"
args.arch = "intent"

torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device("cuda")
tc = TextCache(); stats = TokenStats(args.stats); env_c = load_env_constants(args.env_constants)
empty_idx = tc.index.get("", -1)
kw = dict(H=args.H, stats_path=args.stats, text_cache=tc, env_constants=env_c, H_sparse=args.H_sparse, L_max=args.L_max,
          alpha=args.alpha, p_no_sparse=args.p_no_sparse, alpha_range=(args.alpha_min, args.alpha_max), max_clips=args.max_clips)
train_ds = IntentPolicyDataset("train", F_act=args.F_act, p_rest=args.p_rest, p_neutral=args.p_neutral, seed=args.seed,
                               train=True, randomize_history=True, hip_aug=bool(args.hip_aug), span_scalars=bool(args.span_scalars), **kw)
eval_ds = IntentPolicyDataset(args.eval_split, F_act=args.F_act, train=False, randomize_history=False,
                              span_scalars=bool(args.span_scalars), **kw)
eval_sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_windows, len(eval_ds)), replace=False)
coll = lambda b: collate_intent(b, tc)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=coll,
                          drop_last=True, persistent_workers=True, pin_memory=True)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, eval_sel), batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=coll)

# ---- models
if args.heads % 6 != 0:
    raise SystemExit("the part-structured policy needs --heads a multiple of 6")
args.hidden = (args.hidden // (2 * args.heads)) * (2 * args.heads)
dd = [int(x) for x in args.depth.split(",")]
policy_kw = dict(hidden_dim=args.hidden, num_heads=args.heads, depth_double=dd[0], depth_single=dd[1], mlp_ratio=args.mlp_ratio,
                 text_token_dim=tc.tokens.shape[2], text_pooled_dim=tc.dim, max_text_tokens=tc.max_tokens,
                 text_cross_attention=True, text_mode="sentence_xattn")
model = IntentPolicy(policy_kw, args.intent_dim, args.intent_heads, args.intent_depth, args.intent_mlp, tc.tokens.shape[2]).to(dev)
vae, _ = load_intent_vae(args.vae, dev)
lz = np.load(args.latent_stats)
lat_mean = torch.from_numpy(lz["mean"]).to(dev); lat_std = torch.from_numpy(lz["std"]).to(dev)
log("args", json.dumps(vars(args)))
log(f"train windows {len(train_ds)}, {args.eval_split} windows {len(eval_ds)} -> {len(eval_sel)} fixed; "
    f"window [{args.H_sparse} sparse | {args.H} dense | {args.F_act} future actions], intent targets 16 frames -> 4 x 32")
log(f"params {model.num_params()/1e6:.2f}M: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in model.num_params_by_part().items())
    + f"  (+ frozen VAE {sum(p.numel() for p in vae.parameters())/1e6:.1f}M, not trained)")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


def set_lr(step):
    """constant (the schedule used so far) or warm-up + cosine decay to lr * lr_final_ratio by --steps."""
    if args.lr_schedule == "const":
        return args.lr
    if step <= args.lr_warmup:
        lr = args.lr * step / max(args.lr_warmup, 1)
    else:
        import math as _m
        prog = min(1.0, (step - args.lr_warmup) / max(args.steps - args.lr_warmup, 1))
        lr = args.lr * (args.lr_final_ratio + (1 - args.lr_final_ratio) * 0.5 * (1 + _m.cos(_m.pi * prog)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr
ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
act_mask = ifl.action_channel_mask(dev)
part_idx = ifl.action_part_indices(dev)
step, best_eval = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); ema = {k: v.float().to(dev) for k, v in ck["ema"].items()}
    step = ck["step"]; best_eval = ck.get("best_eval", float("inf")) if ck["args"].get("eval_split") == args.eval_split else float("inf")
    log(f"resumed from {args.resume} at step {step}")


def to_dev(b):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def encode(x):
    with torch.no_grad():
        _, mu, _ = vae.encode(x.float())
    return (mu - lat_mean) / lat_std


def apply_text_dropout(b, p, generator=None):
    """one mask for all three models; returns keep [B] (False = no text AND no intent tokens for the policy)."""
    B = b["root"].shape[0]
    drop = b["text_dropped"].to(dev).clone()
    if p > 0 and empty_idx >= 0:
        drop |= torch.rand(B, device=dev, generator=generator) < p
        tok, po, ln = tc.get(empty_idx)
        b["text_tokens"][drop] = torch.from_numpy(tok).to(dev).to(b["text_tokens"].dtype)
        b["text_pooled"][drop] = torch.from_numpy(po).to(dev).to(b["text_pooled"].dtype)
        b["text_len"][drop] = ln
    return ~drop


def sample_t(B, generator):
    return fl.sample_t(B, dev, args.p_mean, args.p_std, generator=generator, dist="logit_normal")


def hidden_level(B, generator):
    """noise level s at which the intent hidden states are read in training: 1.0 (clean, §21.4-4) or, with
    --cond_aug s_min, s ~ U(s_min, 1) per sample (conditioning augmentation, §21.8)."""
    if args.cond_aug <= 0:
        return 1.0
    return args.cond_aug + (1.0 - args.cond_aug) * torch.rand(B, device=dev, generator=generator)


@torch.no_grad()
def chain_action_loss(b, keep, generator):
    """The policy's action loss when its intents come from the test-time chain -- HIP sampled from text, IIP sampled
    from text + history -- exactly as in the closed loop (Euler 32, CFG 3.5, hidden states read at the test level)."""
    root, body, mask, valid, fidx = b["root"], b["body"], b["observed_mask"], b["valid"], b["frame_index"]
    B = root.shape[0]
    scalars = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
    I_h = encode(b["hist"])
    s_t = float(args.cond_aug_test) if args.cond_aug > 0 else 1.0
    etok, epool, elen = tc.get(empty_idx)
    e_tok = torch.from_numpy(etok).to(dev).float()[None].expand(B, -1, -1).contiguous()
    e_len = torch.full((B,), int(elen), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem_c, mv_c = model.adapter(b["text_tokens"].float(), b["text_len"])
        mem_u, mv_u = model.adapter(e_tok, e_len)
        I_H = ifl.sample_latent(model.hip, B, mem_c, mv_c, mem_u, mv_u, num_steps=32, cfg_scale=3.5, generator=generator, device=dev)
        hH_c = ifl.intent_hidden(model.hip, I_H, s_t, generator, mem=mem_c, mem_valid=mv_c)
        hH_u = ifl.intent_hidden(model.hip, I_H, s_t, generator, mem=mem_u, mem_valid=mv_u)
        I_I = ifl.sample_latent(model.iip, B, mem_c, mv_c, mem_u, mv_u, num_steps=32, cfg_scale=3.5, generator=generator,
                                prefix=I_h, scalars=scalars, extra=hH_c, extra_u=hH_u, device=dev)
        hI_c = ifl.intent_hidden(model.iip, I_I, s_t, generator, mem=mem_c, mem_valid=mv_c, prefix_latent=I_h,
                                 scalars=scalars, mem_extra=hH_c)
        x0 = ifl.policy_input(torch.cat([root, body], -1), mask, act_mask)
        gen = ifl.generated_elements(mask, valid, act_mask)
        tA = sample_t(B, generator)
        z = ifl.build_state_elem(x0, gen, tA, generator=generator)
        toks, tv = model.intent_tokens(hH_c, hI_c, keep)
        xA = model.policy(z, mask, tA, b["text_tokens"], b["text_pooled"], b["text_len"], scalars, valid=valid,
                          frame_index=fidx, extra_tokens=toks, extra_valid=tv)
    vA_hat, vA = ifl.velocity_pair(xA.float(), x0, z, tA, args.v_eps)
    return ifl.policy_loss(vA_hat, vA, gen, part_idx)[0]


def compute_loss(b, keep, generator=None):
    root, body, mask, valid, fidx = b["root"], b["body"], b["observed_mask"], b["valid"], b["frame_index"]
    B = root.shape[0]
    scalars = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
    I_h, I_I, I_H = encode(b["hist"]), encode(b["fut"]), encode(b["holi"])
    ones = torch.ones(B, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem, mem_valid = model.adapter(b["text_tokens"].float(), b["text_len"])
        # HIP
        tH = sample_t(B, generator); zH = fl.build_state(I_H, torch.zeros(B, I_H.shape[1], device=dev), tH, generator=generator)[0]
        xH, _ = model.hip(zH, tH, mem, mem_valid)
        hH = ifl.intent_hidden(model.hip, I_H, hidden_level(B, generator), generator, mem=mem, mem_valid=mem_valid)
        # IIP
        tI = sample_t(B, generator); zI = fl.build_state(I_I, torch.zeros(B, I_I.shape[1], device=dev), tI, generator=generator)[0]
        xI, _ = model.iip(zI, tI, mem, mem_valid, prefix_latent=I_h, scalars=scalars, mem_extra=hH)
        hI = ifl.intent_hidden(model.iip, I_I, hidden_level(B, generator), generator, mem=mem, mem_valid=mem_valid,
                               prefix_latent=I_h, scalars=scalars, mem_extra=hH)
        # action-only policy
        x0 = ifl.policy_input(torch.cat([root, body], -1), mask, act_mask)
        gen = ifl.generated_elements(mask, valid, act_mask)
        tA = sample_t(B, generator)
        z = ifl.build_state_elem(x0, gen, tA, generator=generator)
        toks, tv = model.intent_tokens(hH, hI, keep)
        xA = model.policy(z, mask, tA, b["text_tokens"], b["text_pooled"], b["text_len"], scalars, valid=valid,
                          frame_index=fidx, extra_tokens=toks, extra_valid=tv)
    vH_hat, vH = ifl.velocity_pair(xH.float(), I_H, zH, tH, args.v_eps)
    vI_hat, vI = ifl.velocity_pair(xI.float(), I_I, zI, tI, args.v_eps)
    vA_hat, vA = ifl.velocity_pair(xA.float(), x0, z, tA, args.v_eps)
    l_hip, l_iip = ifl.latent_loss(vH_hat, vH), ifl.latent_loss(vI_hat, vI)
    l_act, per_part = ifl.policy_loss(vA_hat, vA, gen, part_idx)
    loss = l_hip + l_iip + l_act
    return loss, torch.stack([l_hip, l_iip, l_act] + list(per_part.values())).detach(), ["hip", "iip", "act"] + [f"a_{n}" for n in per_part]


@torch.no_grad()
def evaluate_loss():
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(state[k].dtype) for k, v in ema.items()}); model.eval()
    g = torch.Generator(device=dev); g.manual_seed(0)
    g2 = torch.Generator(device=dev); g2.manual_seed(1)
    tot, n, names, chain = None, 0, None, 0.0
    for b in eval_loader:
        b = to_dev(b); keep = ~b["text_dropped"].to(dev)
        _, parts, names = compute_loss(b, keep, generator=g)
        B = b["root"].shape[0]
        tot = parts * B if tot is None else tot + parts * B; n += B
        if args.select == "chain":
            chain += float(chain_action_loss(b, keep, g2)) * B
    model.load_state_dict(state); model.train()
    vals = dict(zip(names, (tot / n).tolist()))
    if args.select == "chain":
        vals["act_chain"] = chain / n
        return vals["act_chain"], vals          # select on what the closed loop actually feeds the policy (§21.8)
    return float(vals["hip"] + vals["iip"] + vals["act"]), vals


def save(path, tag):
    torch.save(dict(model=model.state_dict(), ema=ema, opt=opt.state_dict(), step=step, best_eval=best_eval, args=vars(args),
                    policy_kw=policy_kw, latent_stats=dict(mean=lz["mean"], std=lz["std"]),
                    stats=dict(root_mean=stats.root_mean, root_std=stats.root_std, body_mean=stats.body_mean, body_std=stats.body_std),
                    env_constants={k: np.asarray(v) for k, v in env_c.items()}, tag=tag), path)
    log(f"saved {path} ({tag}) at step {step}")


model.train(); t0 = time.time(); it = iter(train_loader); run = None
while step < args.steps:
    try:
        b = next(it)
    except StopIteration:
        it = iter(train_loader); b = next(it)
    b = to_dev(b)
    keep = apply_text_dropout(b, args.text_dropout)
    loss, parts, names = compute_loss(b, keep)
    if not torch.isfinite(loss):
        log("non-finite loss at step", step); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward()
    cur_lr = set_lr(step + 1)                       # schedule applies to the step about to be taken
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step(); step += 1
    if args.ema_every > 0 and step % args.ema_every == 0:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(args.ema_decay).add_(v.float(), alpha=1 - args.ema_decay)
                else: ema[k].copy_(v)
    cur = torch.cat([parts, gn.detach().float().view(1)])
    run = cur if run is None else run + cur
    if step % args.log_every == 0:
        r = (run / args.log_every).tolist()
        log(f"step {step} loss={sum(r[:3]):.4f} " + " ".join(f"{k}={v:.4f}" for k, v in zip(names + ["gn"], r))
            + f" {(time.time()-t0)/args.log_every*1000:.0f}ms/it"); run = None; t0 = time.time()
    if step % args.eval_every == 0 or step == args.steps:
        el, ep = evaluate_loss()
        log(f"[{args.eval_split}] step {step} select({args.select})={el:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in ep.items()))
        if el < best_eval:
            best_eval = el; save(os.path.join(args.out, f"best_{args.eval_split}.pt"), f"best_{args.eval_split}={el:.4f}")
    if step % args.ckpt_every == 0 or step == args.steps:
        save(os.path.join(args.out, f"step_{step}.pt"), "periodic")
log("done")
