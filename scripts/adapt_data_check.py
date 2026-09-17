"""Smoke-test adapt.data on recorded rollouts (run on a compute node with the uniphys env)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adapt.data import RolloutClipDataset, ClipDatasetCfg
cfg = ClipDatasetCfg(rollout_globs=[sys.argv[1]], stats_path="/scratch/pf2m24/tmp/adapt_stats_test.npz")
ds = RolloutClipDataset(cfg, "train")
x, cond = ds[0]
print("clip tensor", tuple(x.shape), "text", cond["text"], "| vocab size", len(ds.vocab()))
print("vocab sample", ds.vocab()[:20])
print("token mean/std shapes", ds.mean.shape, ds.std.shape, " std min/max", ds.std.min().round(4), ds.std.max().round(3))
