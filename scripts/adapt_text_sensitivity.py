"""How much does the prompt change predicted actions? Guidance sweep on held-out histories."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adapt.data import RolloutClipDataset, ClipDatasetCfg, ACT_DIM
from adapt.policy import DiffusionPolicy
cks = sys.argv[1:]
ds = RolloutClipDataset(ClipDatasetCfg(rollout_globs=["data/g1_rollouts/val_all_x1.pkl"], text_embedding_dict="data/text_embedding_dict_clip_merged.pkl", stats_path="/scratch/pf2m24/tmp/_stats_tmp.npz"), "val")
emb = ds.text_emb
X = torch.stack([ds[i][0] for i in range(0, len(ds), max(1, len(ds) // 96))][:96]).cuda(); H = X[:, :5]
for ck in cks:
    pol = DiffusionPolicy(ck, steps=10, guidance=2.5)
    def pred(p, g, seed=0):
        torch.manual_seed(seed)
        e = None if p is None else torch.from_numpy(np.asarray(emb[p], dtype=np.float32)).cuda()[None].expand(H.shape[0], -1)
        return pol.dm.sample(H, 15, e, steps=10, guidance=g)[:, 5:, :ACT_DIM]
    base = pred("stand", 0.0)
    print(ck, " (normalized action units, same seed)")
    print("   seed-to-seed noise floor (stand, g=0):", f"{(pred('stand', 0.0, 1) - base).abs().mean():.4f}", "  |gt future action|", f"{X[:, 5:, :ACT_DIM].abs().mean():.3f}")
    for g in (0.0, 2.5, 5.0, 15.0):
        d = {p: (pred(p, g) - pred("stand", g)).abs().mean().item() for p in ("walk", "run", "tpose", "sit")}
        print(f"   guidance {g:4.1f}: " + "  ".join(f"|{p}-stand| {v:.4f}" for p, v in d.items()))
