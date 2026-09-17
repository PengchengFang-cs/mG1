"""Step 1a: match HumanML3D index.csv source files to the UniPhys HF AMASS state-action pickles.

Writes:
  data/humanml3d_phys/match.csv      one row per HumanML3D clip: new_name, source_npy, hf_pkl (or ''), start, end
  data/humanml3d_phys/hf_download.txt  unique HF files to download
  data/humanml3d_phys/match_stats.json
"""
import csv, json, os, sys, collections
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/humanml3d_phys"
files = set(json.load(open(f"{ROOT}/hf_file_list.json")))
rows = list(csv.DictReader(open(f"{ROOT}/index.csv")))
stats = collections.Counter(); per_sub = collections.defaultdict(collections.Counter)
out = []; need = set()
for r in rows:
    src = r["source_path"]            # ./pose_data/KIT/3/kick_high_left02_poses.npy
    parts = src.split("/")            # ['.', 'pose_data', 'KIT', '3', 'kick_high_left02_poses.npy']
    sub = parts[2]
    rel = "/".join(parts[2:])[:-4]    # KIT/3/kick_high_left02_poses
    hf = f"amass_state-action-pairs/{rel}.pkl"
    ok = hf in files
    per_sub[sub]["clips"] += 1
    if ok:
        per_sub[sub]["matched"] += 1; need.add(hf); stats["matched"] += 1
    else:
        stats["missing"] += 1
        stats["missing_humanact12" if sub == "humanact12" else "missing_amass"] += 1
    out.append(dict(new_name=r["new_name"], source_npy=src, hf_pkl=hf if ok else "",
                    start_frame=r["start_frame"], end_frame=r["end_frame"]))
with open(f"{ROOT}/match.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
with open(f"{ROOT}/hf_download.txt", "w") as f:
    f.write("\n".join(sorted(need)) + "\n")
stats["clips_total"] = len(rows); stats["unique_hf_files"] = len(need)
stats["per_subset"] = {k: dict(v) for k, v in sorted(per_sub.items())}
json.dump(stats, open(f"{ROOT}/match_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
