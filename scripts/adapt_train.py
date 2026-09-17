"""Train the ADAPT diffusion action prior on recorded tracker rollouts (plain PyTorch)."""
import argparse, os, sys, time, math, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from torch.utils.data import DataLoader
from adapt.data import RolloutClipDataset, ClipDatasetCfg
from adapt.model import DenoiserTransformer
from adapt.diffusion import TokenDiffusion

p = argparse.ArgumentParser()
p.add_argument("--rollouts", nargs="+", required=True)
p.add_argument("--val_rollouts", nargs="*", default=[])
p.add_argument("--text_dict", default="UniPhys/data/babel_state-action-text-pairs/text_embedding_dict_clip.pkl")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--n_frames", type=int, default=20); p.add_argument("--n_hist", type=int, default=5); p.add_argument("--stride", type=int, default=10)
p.add_argument("--K", type=int, default=20)
p.add_argument("--hist_noise_k", type=int, default=0); p.add_argument("--hist_loss_w", type=float, default=0.0)
p.add_argument("--gen", default="ddpm", choices=["ddpm", "flow"], help="generator: discrete-k v-pred diffusion or rectified flow")
p.add_argument("--flow_schedule", default="logit_normal"); p.add_argument("--flow_p_mean", type=float, default=0.0); p.add_argument("--flow_p_std", type=float, default=1.0)
p.add_argument("--hist_noise_t", type=float, default=0.0)
p.add_argument("--flow_pred", default="v", choices=["v", "x0"]); p.add_argument("--flow_loss_space", default="v", choices=["v", "x0"]); p.add_argument("--flow_v_eps", type=float, default=1e-2)
p.add_argument("--layers", type=int, default=8); p.add_argument("--d_model", type=int, default=512)
p.add_argument("--arch", default="xattn", choices=["xattn", "adaln"]); p.add_argument("--n_loops", type=int, default=1, help="adaln: loop the block stack this many times (weight-shared looped transformer)")
p.add_argument("--uncond_prob", type=float, default=0.1)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--log_every", type=int, default=20); p.add_argument("--val_every", type=int, default=200); p.add_argument("--ckpt_every", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--holdout_frac", type=float, default=0.0, help="hold out this fraction of motions (by name hash) as val when --val_rollouts is not given")
p.add_argument("--only_success", type=int, default=1); p.add_argument("--drop_tail", type=int, default=25)
p.add_argument("--init_ckpt", default=None, help="initialize model weights from this checkpoint (fine-tune)")
a = p.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
dev = "cuda"

cfg = ClipDatasetCfg(rollout_globs=a.rollouts, n_frames=a.n_frames, stride=a.stride, text_embedding_dict=a.text_dict, stats_path=os.path.join(a.out, "token_stats.npz"), holdout_frac=a.holdout_frac, only_success=bool(a.only_success), drop_tail_on_fail=a.drop_tail)
train = RolloutClipDataset(cfg, "train")
if a.val_rollouts:
    val = RolloutClipDataset(ClipDatasetCfg(rollout_globs=a.val_rollouts, n_frames=a.n_frames, stride=a.stride, text_embedding_dict=a.text_dict, stats_path=cfg.stats_path), "val")
elif a.holdout_frac > 0:
    val = RolloutClipDataset(ClipDatasetCfg(rollout_globs=a.rollouts, n_frames=a.n_frames, stride=a.stride, text_embedding_dict=a.text_dict, stats_path=cfg.stats_path, holdout_frac=a.holdout_frac), "val")
else:
    val = None
def collate(b):
    x = torch.stack([t[0] for t in b]); e = torch.stack([t[1]["text_embedding"] for t in b]); return x, e
dl = DataLoader(train, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate, persistent_workers=True)
vdl = DataLoader(val, batch_size=a.batch, shuffle=False, collate_fn=collate) if val else None

from adapt.model import AdaLNDenoiser
model = (AdaLNDenoiser(d_model=a.d_model, n_layers=a.layers, n_loops=a.n_loops) if a.arch == "adaln" else DenoiserTransformer(d_model=a.d_model, n_layers=a.layers)).to(dev)
if a.init_ckpt:
    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False); model.load_state_dict(ck["ema"]); print("[train] init from", a.init_ckpt)
if a.gen == "flow":
    from adapt.flow import TokenFlow
    dm = TokenFlow(model, n_hist=a.n_hist, uncond_prob=a.uncond_prob, time_schedule=a.flow_schedule, p_mean=a.flow_p_mean, p_std=a.flow_p_std, hist_noise_t=a.hist_noise_t, hist_loss_w=a.hist_loss_w, pred=a.flow_pred, v_eps=a.flow_v_eps, loss_space=a.flow_loss_space).to(dev)
else:
    dm = TokenDiffusion(model, K=a.K, n_hist=a.n_hist, uncond_prob=a.uncond_prob, hist_noise_k=a.hist_noise_k, hist_loss_w=a.hist_loss_w).to(dev)
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
print(f"[train] clips {len(train)} val {len(val) if val else 0} params {sum(p.numel() for p in model.parameters())/1e6:.1f}M batch {a.batch} steps {a.steps}")
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

def evaluate():
    if vdl is None: return float("nan")
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for x, e in vdl:
            l, _ = dm.loss(x.to(dev), e.to(dev)); tot += float(l) * x.shape[0]; n += x.shape[0]
    model.train(); return tot / max(n, 1)

step, t0, it = 0, time.time(), iter(dl)
logf = open(os.path.join(a.out, "log.csv"), "a"); logf.write("step,loss,val_loss,lr,sec\n")
run_loss = []
while step < a.steps:
    try: x, e = next(it)
    except StopIteration: it = iter(dl); x, e = next(it)
    loss, _ = dm.loss(x.to(dev), e.to(dev))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); ema.update_parameters(model); step += 1
    run_loss.append(float(loss))
    if step % a.log_every == 0 or step == a.steps:
        vl = evaluate() if (step % a.val_every == 0 or step == a.steps) else float("nan")
        m = sum(run_loss) / len(run_loss); run_loss = []
        print(f"[train] step {step:6d} loss {m:.4f} val {vl:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        logf.write(f"{step},{m:.5f},{vl:.5f},{sched.get_last_lr()[0]:.3e},{time.time()-t0:.0f}\n"); logf.flush()
    if step % a.ckpt_every == 0 or step == a.steps:
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "args": vars(a), "step": step,
                    "token_mean": train.mean, "token_std": train.std}, os.path.join(a.out, f"ckpt_{step}.pt"))
print("[train] done")
