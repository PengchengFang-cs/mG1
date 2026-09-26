"""Per-dimension mean / std of the frozen intent-VAE latents (posterior means) over TRAINING windows, pooled over the
history, immediate and holistic sequences (MIND: HIP and IIP share one latent space). docs/07 §21.1."""
import argparse, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.dataset import TextCache, load_env_constants, ROOT
from hml_phys.intent_policy_data import IntentPolicyDataset, collate_intent
from hml_phys.intent_vae import load_intent_vae

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="outputs/intent_vae_v2/best_test.pt")
ap.add_argument("--out", default=os.path.join(ROOT, "intent_latent_stats_vae_v2.npz"))
ap.add_argument("--n_batches", type=int, default=200); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--stats", default=os.path.join(ROOT, "token_stats_v3.npz")); ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()
dev = torch.device("cuda")
tc = TextCache()
ds = IntentPolicyDataset("train", F_act=4, stats_path=args.stats, text_cache=tc, env_constants=load_env_constants(),
                         train=True, seed=0)
dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=lambda b: collate_intent(b, tc),
                drop_last=True, generator=torch.Generator().manual_seed(0))
vae, _ = load_intent_vae(args.vae, dev)
acc = {k: [] for k in ("hist", "fut", "holi")}
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(dl):
        if i >= args.n_batches:
            break
        for k in acc:
            _, mu, _ = vae.encode(b[k].to(dev))
            acc[k].append(mu.reshape(-1, mu.shape[-1]).cpu().numpy())
lat = {k: np.concatenate(v) for k, v in acc.items()}
allz = np.concatenate(list(lat.values())).astype(np.float64)
mean, std = allz.mean(0), allz.std(0) + 1e-6
np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32), vae=args.vae,
         n=np.int64(len(allz)))
print(f"{len(allz)} latent vectors from {args.n_batches * args.batch} windows in {time.time()-t0:.0f}s -> {args.out}")
for k, v in lat.items():
    z = (v - mean) / std
    print(f"  {k:4s} raw |mu| {np.abs(v).mean():.3f} std {v.std(0).mean():.3f} | normalised mean {z.mean():+.3f} std {z.std():.3f}")
print(f"  pooled std per dim: min {std.min():.3f} median {np.median(std):.3f} max {std.max():.3f}")
