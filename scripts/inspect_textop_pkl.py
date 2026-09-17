"""Print structure of TextOp retargeted G1 dataset pkl (RobotMDAR format). Run on a compute node."""
import sys, joblib, numpy as np
path = sys.argv[1]
d = joblib.load(path)
print("type", type(d), "len", len(d))
if isinstance(d, dict):
    keys = list(d.keys()); print("first keys:", keys[:3]); items=[d[k] for k in keys]
else:
    items = d; keys = list(range(len(d)))
m = items[0]
print("motion type", type(m))
if isinstance(m, dict):
    for k, v in m.items():
        if hasattr(v, "shape"): print(f"  {k:24s} ndarray{tuple(v.shape)} {v.dtype}")
        elif isinstance(v, (list, tuple)): print(f"  {k:24s} {type(v).__name__} len={len(v)} first={str(v[0])[:120] if len(v) else None}")
        else: print(f"  {k:24s} {type(v).__name__}: {str(v)[:120]}")
lens = [ (it["dof"].shape[0] if isinstance(it, dict) and "dof" in it else -1) for it in items[:2000]]
if isinstance(m, dict):
    for tk in ("text","texts","labels","frame_labels","seg_labels","caption"):
        if tk in m: print("TEXT", tk, str(m[tk])[:400])
    print("motion 1 keys same?", isinstance(items[1], dict) and list(items[1].keys())==list(m.keys()))
lens = [l for l in lens if l > 0]
if lens: print("frames per motion (first 2000): min/med/max", min(lens), int(np.median(lens)), max(lens))
