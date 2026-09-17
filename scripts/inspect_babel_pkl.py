"""Print the structure of a UniPhys BABEL state-action-text pickle. Run on a compute node."""
import sys, joblib, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else "data/babel_state-action-text-pairs/babel_val.pkl"
d = joblib.load(path)
print("top-level keys:", list(d.keys()))
n = len(d["action_all"])
print("num sequences:", n)
for k, v in d.items():
    e = v[0] if isinstance(v, (list, tuple)) else v
    if hasattr(e, "shape"):
        shp = f"ndarray{tuple(e.shape)} {e.dtype}"
    elif isinstance(e, (list, tuple)):
        shp = f"{type(e).__name__} len={len(e)}"
    else:
        shp = f"{type(e).__name__}: {str(e)[:60]}"
    ln = len(v) if hasattr(v, "__len__") else "-"
    print(f"  {k:20s} len={ln!s:6s} elem0 = {shp}")
i = 0
print("\n--- sequence 0 ---")
print("motion_file:", d["motion_file"][i] if "motion_file" in d else None)
print("is_succ:", d["is_succ_all"][i])
print("frames (root_state_all):", np.asarray(d["root_state_all"][i]).shape)
if "frame_labels_all" in d:
    fl = d["frame_labels_all"][i]
    print("frame_labels: n_segments =", len(fl))
    for seg in fl[:8]:
        print("   ", {k: seg[k] for k in seg if k in ("start_t", "end_t", "proc_label", "raw_label")})
lens = [np.asarray(x).shape[0] for x in d["root_state_all"]]
print("\nframes per seq: min/median/max =", min(lens), int(np.median(lens)), max(lens), " total frames =", sum(lens), " ~hours @30fps =", round(sum(lens)/30/3600, 2))
print("succ count:", int(np.sum(d["is_succ_all"])), "/", n, " succ_idxes len:", len(d.get("succ_idxes", [])))
