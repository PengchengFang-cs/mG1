"""Build a name->motion dict pkl (pklpack_to_npz format) from the RobotMDAR list-format dataset.
Usage: python make_motion_subset.py <in.pkl> <out.pkl> [n_motions] [min_len] [max_len]
"""
import sys, joblib, re
src, dst = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
min_len = int(sys.argv[4]) if len(sys.argv) > 4 else 100
max_len = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
d = joblib.load(src)
out, meta = {}, {}
for item in d:
    if not (min_len <= item["length"] <= max_len):
        continue
    name = re.sub(r"[^A-Za-z0-9_-]", "_", item["feat_p"].replace(".pkl", ""))[:100]
    if name in out:
        continue
    m = dict(item["motion"])
    m["fps"] = int(m.get("fps", 50))
    out[name] = m
    meta[name] = {"feat_p": item["feat_p"], "babel_sid": item["babel_sid"], "frame_ann": item["frame_ann"],
                  "length": item["length"], "duration": item["duration"]}
    if len(out) >= n:
        break
joblib.dump(out, dst)
joblib.dump(meta, dst.replace(".pkl", "_meta.pkl"))
print("wrote", len(out), "motions ->", dst)
for k, v in list(meta.items())[:5]:
    print(" ", k, v["length"], [a[2] for a in v["frame_ann"]])
