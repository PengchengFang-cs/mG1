"""Offline one-step check: given true history, predict first future token; compare action to ground truth."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adapt.data import RolloutClipDataset, ClipDatasetCfg, ACT_DIM
from adapt.policy import DiffusionPolicy
ckpt, roll, tdict = sys.argv[1], sys.argv[2], sys.argv[3]
ds = RolloutClipDataset(ClipDatasetCfg(rollout_globs=[roll], text_embedding_dict=tdict, holdout_frac=0.1, stats_path="/scratch/pf2m24/tmp/_stats_tmp.npz"), "val")
pol = DiffusionPolicy(ckpt, steps=2, guidance=2.5)
mean, std = pol.mean, pol.std
idx = np.random.RandomState(0).choice(len(ds), 256, replace=False)
X = torch.stack([ds[i][0] for i in idx]).cuda(); E = torch.stack([ds[i][1]["text_embedding"] for i in idx]).cuda()
H = pol.n_hist
gt_fut = X[:, H]                       # normalized [a_t, o_{t+1}]
copy_prev = X[:, H - 1]                # previous token: a_{t-1}, o_t
def err(pred): return float(((pred[:, :ACT_DIM] - gt_fut[:, :ACT_DIM]) ** 2).mean()), float(((pred[:, ACT_DIM:] - gt_fut[:, ACT_DIM:]) ** 2).mean())
print("baseline copy-prev  action MSE %.4f  obs MSE %.4f" % err(copy_prev))
print("baseline zero       action MSE %.4f  obs MSE %.4f" % err(torch.zeros_like(gt_fut)))
for steps in (2, 5, 10, 20):
    for g in (0.0, 1.0, 2.5):
        outs = []
        for s in range(3):
            torch.manual_seed(s); outs.append(pol.dm.sample(X[:, :H], 15, E, steps=steps, guidance=g)[:, H])
        pred = torch.stack(outs).mean(0)
        a_mse, o_mse = err(outs[0])
        print(f"steps {steps:2d} guidance {g:3.1f}  action MSE {a_mse:.4f}  obs MSE {o_mse:.4f}   (3-sample mean: action {err(pred)[0]:.4f})")
# also check full-future rollout consistency: predicted future actions vs gt along 15 frames
out = pol.dm.sample(X[:, :H], 15, E, steps=10, guidance=2.5)
per_t = ((out[:, H:, :ACT_DIM] - X[:, H:, :ACT_DIM]) ** 2).mean((0, 2))
print("per-future-step action MSE (10 steps, g2.5):", [round(float(v), 3) for v in per_t])
