"""Latent statistics for the G1 intent space.

HIP and IIP regress onto the frozen VAE's latents, so those latents have to be normalised before they become
a regression target -- otherwise the flow loss is dominated by whichever latent dimension happens to have the
largest scale.  This is the G1 counterpart of `scripts/hml_phys/compute_intent_latent_stats.py`.

The statistics are fitted on the TRAIN split over all three streams the intent modules ever see (history,
immediate future, holistic), because HIP's target is a holistic latent while IIP's is an immediate one and
both must live on the same scale -- MIND shares one latent space between them.
"""
import argparse, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.g1_data import G1_DIR, G1WindowDataset, collate_g1
from hml_phys.intent_vae import load_intent_vae

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="outputs/g1_intent_vae/best_val.pt")
ap.add_argument("--split", default="train")
ap.add_argument("--stride", type=int, default=29, help="window stride; the statistics do not need every window")
ap.add_argument("--max_windows", type=int, default=200000)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--out", default=os.path.join(G1_DIR, "g1_intent_latent_stats.npz"))
args = ap.parse_args()
dev = "cuda"

vae, vargs = load_intent_vae(args.vae, dev)
vae.eval()
print(f"VAE {args.vae}: input_dim={vargs.get('input_dim')} latent_dim={vargs.get('latent_dim')} "
      f"step={vargs.get('iters')}", flush=True)

ds = G1WindowDataset(args.split, intent=True, stride=args.stride)
sel = np.arange(len(ds))
if args.max_windows and len(ds) > args.max_windows:
    sel = np.sort(np.random.RandomState(0).choice(len(ds), args.max_windows, replace=False))
loader = DataLoader(torch.utils.data.Subset(ds, sel), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, collate_fn=lambda b: collate_g1(b))
print(f"{ds.n_rollouts} rollouts -> {len(ds)} windows -> {len(sel)} sampled", flush=True)

n = 0
s1 = s2 = None
with torch.no_grad():
    for k, b in enumerate(loader):
        x = torch.cat([b["hist"], b["fut"], b["holi"]], 0).to(dev).float()
        z = vae.encode(x)[1]                      # [B*3, T', latent]
        f = z.reshape(-1, z.shape[-1]).double()
        if s1 is None:
            s1 = torch.zeros(f.shape[-1], dtype=torch.float64, device=dev)
            s2 = torch.zeros_like(s1)
        n += f.shape[0]; s1 += f.sum(0); s2 += (f * f).sum(0)
        if k % 50 == 0:
            print(f"  {k * args.batch}/{len(sel)}", flush=True)

mean = (s1 / n).float().cpu().numpy()
var = torch.clamp(s2 / n - (s1 / n) ** 2, min=0.0)
std = torch.clamp(var.sqrt(), min=1e-4).float().cpu().numpy()
print(f"\n{n} latent vectors, dim {mean.shape[0]}")
print(f"  |mean| max {np.abs(mean).max():.4f}   std min {std.min():.4f}   std max {std.max():.4f}")
np.savez(args.out, mean=mean, std=std, n=np.int64(n),
         meta=json.dumps(dict(vae=args.vae, split=args.split, stride=args.stride, n_windows=int(len(sel)),
                              streams=["hist", "fut", "holi"])))
print(f"wrote {args.out}")
