"""Train the CodeFlow policy: a MoGeFlow-style frame flow over a FROZEN RVQ tokenizer (docs/08 §10).

Version 1 (this script, `--intent 0`): text + observed history codes -> the codes of the next chunk. No intent
mechanism, no sparse distant history -- the tokenizer is a temporal convolution and needs contiguous frames.
Version 2 adds the unchanged HIP/IIP intent on top (`--intent 1`, see docs/07 §21).

Flow loss only (MoGeFlow's released recipe has `terminal loss 0.0`); x0 prediction, velocity-space loss averaged
per residual level; logit-normal t; Euler 32 / CFG 3.5 at sampling time.
The periodic loss curve and the best checkpoint use the TEST split only (project CLAUDE.md §1).

`--rvq_obs <ckpt>` is the 码本对照 control (user, 2026-09-22): the OBSERVATION is encoded by a second frozen
tokenizer over the full 435-channel token while `--rvq` stays the 69-channel ACTION tokenizer the policy
generates in.  That matches route A's task exactly -- observe the whole state, produce only actions -- so the
one remaining difference against `train_intent_policy.py` is discrete codes vs continuous actions.  Without it
this script is unchanged.  Windows are then loaded once as the "token" variant and the action channels are
sliced out of the future half on the GPU (hml_phys/codeflow_data.py, module docstring).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import TextCache, load_env_constants
from hml_phys.codeflow import CodeFlowIntentPolicy, CodeFlowPolicy
from hml_phys import intent_flow as ifl
from hml_phys.codeflow_data import CodeFlowDataset, collate_codeflow
from hml_phys.codeflow_flow import latent_mask, level_loss, logit_normal_t
from hml_phys.intent_flow import build_state_elem, velocity_pair
from hml_phys.rvq_data import DEFAULT_STATS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--rvq", required=True, help="frozen GENERATION tokenizer checkpoint from train_rvq.py")
ap.add_argument("--rvq_obs", default="", help="码本对照 control: a second frozen tokenizer (normally the 435-channel "
                                             "'token' one) that encodes the OBSERVED history; --rvq then only has "
                                             "to carry the generated action channels")
ap.add_argument("--window", type=int, default=64); ap.add_argument("--n_hist", type=int, default=32)
ap.add_argument("--hidden", type=int, default=504); ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--depth_double", type=int, default=3); ap.add_argument("--depth_single", type=int, default=6)
ap.add_argument("--text_mode", default="joint_tokens", choices=["joint_tokens", "sentence_xattn", "xattn_only"])
ap.add_argument("--text_xattn", type=int, default=0)
ap.add_argument("--intent", type=int, default=0, help="version 2: MIND's HIP/IIP intent, unchanged (docs/07 §21)")
ap.add_argument("--vae", default="outputs/intent_vae_v2/best_test.pt")
ap.add_argument("--latent_stats", default=os.path.join(os.path.dirname(DEFAULT_STATS), "intent_latent_stats_vae_v2.npz"))
ap.add_argument("--intent_dim", type=int, default=384); ap.add_argument("--intent_heads", type=int, default=6)
ap.add_argument("--intent_depth", type=int, default=4); ap.add_argument("--intent_mlp", type=float, default=1.5)
ap.add_argument("--cond_aug", type=float, default=0.5, help="read the intent hidden states at s ~ U(cond_aug,1)")
ap.add_argument("--cond_aug_test", type=float, default=0.75)
ap.add_argument("--p_uncond", type=float, default=0.1, help="caption dropout for classifier-free guidance")
ap.add_argument("--v_eps", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--steps", type=int, default=200000)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--eval_every", type=int, default=5000); ap.add_argument("--eval_windows", type=int, default=2048)
ap.add_argument("--ckpt_every", type=int, default=50000); ap.add_argument("--latest_every", type=int, default=2000)
ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--eval_split", default="test"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--stats", default=DEFAULT_STATS); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--max_clips", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="")
args = ap.parse_args()
assert args.eval_split != "val", "val split is banned in this project (CLAUDE.md §1)"

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
log_f = open(os.path.join(args.out, "train_log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device("cuda")

kw = dict(hidden_dim=args.hidden, num_heads=args.heads, depth_double=args.depth_double,
          depth_single=args.depth_single, rvq_obs_ckpt=(args.rvq_obs or None))
if args.intent:
    args.text_mode, args.text_xattn = "sentence_xattn", 1
    model = CodeFlowIntentPolicy(args.rvq, args.vae, args.latent_stats, device=dev, intent_dim=args.intent_dim,
                                 intent_heads=args.intent_heads, intent_depth=args.intent_depth,
                                 intent_mlp=args.intent_mlp, **kw)
    params = model.trainable()
else:
    model = CodeFlowPolicy(args.rvq, device=dev, text_mode=args.text_mode,
                           text_cross_attention=bool(args.text_xattn), **kw)
    params = list(model.policy.parameters())
DUAL = model.dual                          # 码本对照 control: observation tokenizer != generation tokenizer
# the dataset feeds the OBSERVATION tokenizer; in the single-tokenizer case that is the same variant as before
variant = model.obs_variant
tc = TextCache()
mk = lambda split, train: CodeFlowDataset(split, variant=variant, window=args.window, n_hist=args.n_hist,
                                          text_cache=tc, stride=args.stride, stats_path=args.stats,
                                          max_clips=args.max_clips, seed=args.seed, train=train,
                                          intent=bool(args.intent))
train_ds, eval_ds = mk("train", True), mk(args.eval_split, False)
eval_sel = np.random.RandomState(0).choice(len(eval_ds), min(args.eval_windows, len(eval_ds)), replace=False)
coll = lambda b: collate_codeflow(b, tc)
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0, pin_memory=True, collate_fn=coll)
eval_loader = DataLoader(torch.utils.data.Subset(eval_ds, eval_sel), batch_size=args.batch, shuffle=False,
                         num_workers=2, collate_fn=coll)

args.variant, args.n_channels = model.variant, model.n_channels                 # generation tokenizer
args.obs_variant, args.obs_n_channels = model.obs_variant, model.obs_n_channels  # observation tokenizer
log("args", json.dumps(vars(args)))
log(f"generation tokenizer {args.rvq}: variant {model.variant}, {model.n_channels} channels, {model.n_quant} "
    f"levels x {model.code_dim} dims, {model.rvq.nb_code} codes, /{model.down} downsampling")
if DUAL:
    log(f"observation tokenizer {args.rvq_obs}: variant {model.obs_variant}, {model.obs_n_channels} channels "
        f"(码本对照 control: observe the full state, generate only the action codes)")
log(f"window {args.window} -> {train_ds.n_lat} latent frames = [{train_ds.n_lat_hist} observed | "
    f"{train_ds.n_lat - train_ds.n_lat_hist} generated]; latent dim {model.latent_dim}; "
    f"train windows {len(train_ds)}, {args.eval_split} {len(eval_ds)} -> {len(eval_sel)} fixed; "
    f"policy {model.num_params()/1e6:.1f}M trainable params")

trainable_mod = model if args.intent else model.policy
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
EMA_SKIP = ("rvq.", "rvq_obs.", "vae.")   # frozen submodules stay out of the EMA and out of the checkpoint
ema = {k: v.detach().clone().float() for k, v in trainable_mod.state_dict().items()
       if not k.startswith(EMA_SKIP)}
step, best = 0, float("inf")
if args.resume:
    ck = torch.load(args.resume, map_location="cpu")
    trainable_mod.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"]); step = ck["step"]
    best = ck.get("best", float("inf"))
    ema = {k: v.to(dev).float() for k, v in ck["ema"].items()}
    log(f"resumed from {args.resume} at step {step}")

empty_idx = tc.index.get("", -1)
e_tok, e_pool, e_len = tc.get(empty_idx)
E_TOK = torch.from_numpy(e_tok).to(dev); E_POOL = torch.from_numpy(e_pool).to(dev); E_LEN = int(e_len)


def batch_to_dev(b):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def prepare(b, drop=None):
    """-> (x0 [B,T',Q*D], obs, gen, text triple, scalars).

    The history and the future are encoded as SEPARATE windows. Encoding the whole 64-frame window and calling
    its first half "observed" leaks the future into the history: the encoder's receptive field is about +-86
    frames, wider than the window, and perturbing only the future 32 frames was measured to change 46.4% of the
    history codes (87.5% in the latent frame next to the boundary). A policy trained that way reads the answer
    out of its own history and collapses in the closed loop. Encoding the halves separately is exactly what the
    rollout can do, and costs the tokenizer only 5% reconstruction MSE on 32-frame windows (0.0895 vs 0.0849).

    In the 码本对照 mode the two halves additionally go through DIFFERENT tokenizers: the history keeps all
    435 channels and is encoded by the observation tokenizer, the future is cut down to the 69 action channels
    and encoded by the generation tokenizer. Both give 8 latent frames of Q*D, so they concatenate as before.
    """
    x = b["x"]
    B = x.shape[0]
    nh = train_ds.n_hist
    if DUAL:
        x0_h, _ = model.encode_obs(x[:, :nh].contiguous())                       # 435 channels, obs codebook
        x0_f, _ = model.encode_window(model.gen_channels_of(x[:, nh:]).contiguous())   # 69 channels, gen codebook
    else:
        x0_h, _ = model.encode_window(x[:, :nh].contiguous())
        x0_f, _ = model.encode_window(x[:, nh:].contiguous())
    x0 = torch.cat([x0_h, x0_f], 1)
    obs, gen = latent_mask(B, train_ds.n_lat, train_ds.n_lat_hist, dev)
    tok, pool, ln = b["text_tokens"], b["text_pooled"], b["text_len"]
    if drop is not None and drop.any():                       # CFG dropout: swap in the CLIP("") features
        tok = torch.where(drop[:, None, None], E_TOK[None].expand_as(tok), tok)
        pool = torch.where(drop[:, None], E_POOL[None].expand_as(pool), pool)
        ln = torch.where(drop, torch.full_like(ln, E_LEN), ln)
    scal = torch.stack([b["progress"], b["total_len"] / 10.0], -1).float()
    return x0, obs, gen, (tok, pool, ln), scal


def intent_parts(b, text, scal, drop, generator=None, s_read=None):
    """HIP / IIP exactly as in route A (docs/07 §21): both predictors are flow-matched on the frozen VAE's
    latents, and the policy reads their hidden states under conditioning augmentation."""
    B = b["x"].shape[0]
    I_H, I_I, I_h = model.encode_latent(b["holi"]), model.encode_latent(b["fut"]), model.encode_latent(b["hist"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mem, mv = model.adapter(text[0].float(), text[2])
        t_h = logit_normal_t(B, dev, generator)
        z_h = build_state_elem(I_H, torch.ones_like(I_H), t_h, generator=generator)
        v_h_hat, v_h = velocity_pair(model.hip(z_h, t_h, mem, mv)[0].float(), I_H, z_h, t_h, args.v_eps)
        t_i = logit_normal_t(B, dev, generator)
        z_i = build_state_elem(I_I, torch.ones_like(I_I), t_i, generator=generator)
        hH = ifl.intent_hidden(model.hip, I_H, s_read if s_read is not None else
                               float(np.random.uniform(args.cond_aug, 1.0)), generator, mem=mem, mem_valid=mv)
        v_i_hat, v_i = velocity_pair(model.iip(z_i, t_i, mem, mv, prefix_latent=I_h, scalars=scal,
                                               mem_extra=hH)[0].float(), I_I, z_i, t_i, args.v_eps)
        hI = ifl.intent_hidden(model.iip, I_I, s_read if s_read is not None else
                               float(np.random.uniform(args.cond_aug, 1.0)), generator, mem=mem, mem_valid=mv,
                               prefix_latent=I_h, scalars=scal, mem_extra=hH)
    toks, tv = model.intent_tokens(hH, hI, ~drop if drop is not None else
                                   torch.ones(B, dtype=torch.bool, device=dev))
    return ifl.latent_loss(v_h_hat, v_h), ifl.latent_loss(v_i_hat, v_i), toks, tv


def loss_on(b, t=None, generator=None, drop=None, s_read=None):
    x0, obs, gen, text, scal = prepare(b, drop)
    B = x0.shape[0]
    t = logit_normal_t(B, dev, generator) if t is None else t
    z = build_state_elem(x0, gen, t, generator=generator)
    kw, extra = {}, {}
    if args.intent:
        l_hip, l_iip, toks, tv = intent_parts(b, text, scal, drop, generator, s_read)
        kw = dict(extra_tokens=toks, extra_valid=tv)
        extra = dict(hip=l_hip, iip=l_iip)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x0_hat = model(z, obs, t, text[0], text[1], text[2], scal, **kw)
    v_hat, v = velocity_pair(x0_hat.float(), x0, z, t, args.v_eps)
    loss, per = level_loss(v_hat, v, gen, model.n_quant, model.code_dim)
    if args.intent:
        per = dict(per, **{k: v for k, v in extra.items()})
        loss = loss + extra["hip"] + extra["iip"]        # equal weights, as MIND trains them jointly
    return loss, per, x0_hat.float(), x0, gen


@torch.no_grad()
def evaluate():
    """teacher-forced flow loss on a FIXED set of test windows plus the snap accuracy at t = 0.5."""
    bak = {k: v.detach().clone() for k, v in trainable_mod.state_dict().items() if k in ema}
    trainable_mod.load_state_dict({k: v.to(bak[k].dtype) for k, v in ema.items()}, strict=False)
    trainable_mod.eval()
    tot, n = {}, 0
    for b in eval_loader:
        b = batch_to_dev(b)
        B = b["x"].shape[0]
        g = torch.Generator(device=dev); g.manual_seed(1234)
        t = torch.full((B,), 0.5, device=dev)
        loss, per, x0_hat, x0, gen = loss_on(b, t=t, generator=g, s_read=args.cond_aug_test if args.intent else None)
        # snap the GENERATED rows only: under --rvq_obs the observed rows belong to the other codebook
        nlh = train_ds.n_lat_hist
        codes_hat, _ = model.snap(x0_hat[:, nlh:])
        codes_gt, _ = model.snap(x0[:, nlh:])                 # x0 is already exactly on the codebook
        m = gen[:, nlh:, 0].bool()
        acc = float((codes_hat[m] == codes_gt[m]).float().mean())
        vals = dict(loss=float(loss), snap_acc=acc, **{k: float(v) for k, v in per.items()})
        for k, v in vals.items():
            tot[k] = tot.get(k, 0.0) + v * B
        n += B
    trainable_mod.load_state_dict(bak, strict=False); trainable_mod.train()
    return {k: v / n for k, v in tot.items()}


def save(path, tag):
    torch.save(dict(model={k: v for k, v in trainable_mod.state_dict().items() if k in ema}, ema=ema, opt=opt.state_dict(), step=step, best=best,
                    args=vars(args), tag=tag, rvq=args.rvq), path)
    log(f"saved {path} ({tag}) at step {step}")


trainable_mod.train(); t0 = time.time(); run, n_run = {}, 0
data_iter = iter(train_loader)
while step < args.steps:
    try:
        b = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader); b = next(data_iter)
    step += 1
    for gp in opt.param_groups:
        gp["lr"] = args.lr * min(1.0, step / max(args.warmup, 1))
    b = batch_to_dev(b)
    drop = torch.rand(b["x"].shape[0], device=dev) < args.p_uncond
    loss, per, _, _, _ = loss_on(b, drop=drop)
    if not torch.isfinite(loss):
        log("non-finite loss at step", step); save(os.path.join(args.out, f"nan_step_{step}.pt"), "non-finite"); raise SystemExit(1)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
    opt.step()
    with torch.no_grad():
        d = args.ema
        for k in ema:
            ema[k].mul_(d).add_(trainable_mod.state_dict()[k].detach().float(), alpha=1 - d)
    vals = dict(loss=float(loss), gn=float(gn), **{k: float(v) for k, v in per.items()})
    for k, v in vals.items():
        run[k] = run.get(k, 0.0) + v
    n_run += 1
    if step % args.log_every == 0:
        log(f"step {step} " + " ".join(f"{k}={v/n_run:.4f}" for k, v in run.items())
            + f" lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/n_run*1000:.0f}ms/it")
        run, n_run, t0 = {}, 0, time.time()
    if step % args.eval_every == 0 or step == args.steps:
        ev = evaluate()
        log(f"[{args.eval_split}] step {step} " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
        if ev["loss"] < best:
            best = ev["loss"]; save(os.path.join(args.out, f"best_{args.eval_split}.pt"), f"best={best:.5f}")
    if args.latest_every and step % args.latest_every == 0:
        save(os.path.join(args.out, "latest.pt"), "latest")
    if step % args.ckpt_every == 0 or step == args.steps:
        save(os.path.join(args.out, f"step_{step}.pt"), "periodic")
log("done")
