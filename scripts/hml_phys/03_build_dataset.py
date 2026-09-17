"""Step 2: cut the PULSE-tracked AMASS state-action sequences into HumanML3D clips.

Inputs
  data/humanml3d_phys/match.csv            (from 01_match_index.py; HumanML3D index.csv joined with HF files)
  data/humanml3d_phys/uniphys_hf/...pkl    per-sequence joblib pickles: body_pos [T,24,3], dof_state [T,69,2],
                                           root_state [T,13], action [T,69], pulse_z [T,32], is_succ, fps=30
  HumanML3D: train/val/test.txt, texts/<id>.txt, new_joints/<id>.npy (only for sequence lengths)

Frame alignment
  HumanML3D frame k of clip <id> is pose_data[trim + start_frame + k] at 20 fps, i.e. time
  (trim + start_frame + k)/20 s from the AMASS sequence start, where trim is the dataset-specific head trim
  of raw_pose_processing.ipynb (Eyes_Japan/HDM05 3 s, TotalCapture/MPI_Limits 1 s, Transitions 0.5 s).  PHC converts AMASS with skip = int(src_fps / 30), so sources at 100 fps
  (KIT, MPI_mosh) are actually 33.3 fps although labelled 30.  We estimate the effective fps per source
  file by content: the longest clip of the file is converted to HumanML3D canonical joints under fps 30 and
  33.33 and its root-relative local joint positions (263-d block 4:67) are compared with the same block of
  new_joint_vecs/<id>.npy; the fps with the smaller mean joint error wins (drift-invariant; the error itself
  is stored per file as an alignment / tracking-quality measure).  Physics frames are then [round(t0 * fps_eff), round(t1 * fps_eff)).  Physics data is kept at its
  native (nominal 30 fps) rate; `time_stretch` = fps_eff / 30 records how much slower than real time the
  tracked motion was replayed (1.0 for 120/60 fps sources, 1.11 for 100 fps sources).

Outputs  data/humanml3d_phys/hml_phys_{train,val,test}.pkl  (joblib, dict of lists, one entry per clip)
  name (LOCAL id, the one texts/ new_joint_vecs/ splits use), official_id (index.csv id), split, source_file, hf_pkl, start_frame, end_frame, fps_eff, time_stretch, phys_i0, phys_i1,
  n_frames, texts=[{caption, tokens, f_tag, to_tag}], body_pos, dof_state, root_state, action, pulse_z
and data/humanml3d_phys/build_stats.json
"""
import argparse, csv, json, os, collections, time
import numpy as np, joblib

import sys
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from hml_phys.hml_ids import official_to_local
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/humanml3d_phys"
HML = "/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D"
HF = os.path.join(ROOT, "uniphys_hf")
KEYS = ["body_pos", "dof_state", "root_state", "action", "pulse_z"]

ap = argparse.ArgumentParser()
ap.add_argument("--min_frames", type=int, default=2, help="drop physics clips shorter than this")
ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
args = ap.parse_args()

t0 = time.time()
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "match.csv"))) if r["hf_pkl"]]
for r in rows:  # index.csv uses OFFICIAL ids; texts / splits / new_joint_vecs use the LOCAL renumbering
    r["official_id"] = r["new_name"].replace(".npy", "")
    r["local_id"] = official_to_local(r["official_id"])
n_no_local = sum(1 for r in rows if r["local_id"] is None)
rows = [r for r in rows if r["local_id"] is not None]
print(f"{len(rows)} matched rows ({n_no_local} official ids absent from the local copy)")
if args.limit:
    rows = rows[:args.limit]
split_of = {}
for sp in ["train", "val", "test"]:
    for l in open(os.path.join(HML, sp + ".txt")):
        if l.strip():
            split_of[l.strip()] = sp


def read_texts(name):
    out = []
    for line in open(os.path.join(HML, "texts", name + ".txt"), encoding="utf-8"):
        p = line.strip().split("#")
        if len(p) < 4:
            continue
        f, t = float(p[2]), float(p[3])
        out.append(dict(caption=p[0], tokens=p[1].split(" "), f_tag=0.0 if np.isnan(f) else f, to_tag=0.0 if np.isnan(t) else t))
    return out


def hml_len(name):
    return int(np.load(os.path.join(HML, "new_joint_vecs", name + ".npy"), mmap_mode="r").shape[0])


# ---- pass 1: effective fps per source file by content alignment against HumanML3D new_joints
# For each HF file take its longest clip, convert the physics slice to HumanML3D canonical joints under
# fps in {30, 33.33} and keep the fps whose mean joint error against new_joints/<id>.npy is smaller.
import sys
sys.path.insert(0, "/iridisfs/scratch/pf2m24/projects/motion_rebot")
from multiprocessing import Pool
from hml_phys.sim2hml import isaac_body_pos_to_hml_joints, joints_to_hml263

CANDIDATE_FPS = (30.0, 100.0 / 3.0, 31.25)  # 120/60 fps sources -> 30; 100 fps -> 33.33; 250 fps (SSM_synced) -> 31.25
# HumanML3D raw_pose_processing.ipynb trims the beginning of some datasets BEFORE applying index.csv
# (fps = 20 there): Eyes_Japan_Dataset / MPI_HDM05 3 s, TotalCapture / MPI_Limits 1 s, Transitions_mocap 0.5 s.
HML_HEAD_TRIM_FRAMES = {"Eyes_Japan_Dataset": 60, "MPI_HDM05": 60, "TotalCapture": 20, "MPI_Limits": 20, "Transitions_mocap": 10}


def head_trim(hf):
    return HML_HEAD_TRIM_FRAMES.get(hf.split("/")[1], 0)


def load(hf):
    return joblib.load(os.path.join(HF, hf))


def align_error(body_pos, T, s, e, n_hml, fps, hml_joints, trim):
    if e == -1:
        e = s + n_hml + 1
    i0 = int(round((s + trim) / 20.0 * fps)); i1 = int(round((e + trim) / 20.0 * fps))
    i0 = max(0, min(i0, T)); i1 = max(i0, min(i1, T))
    if i1 - i0 < 4:
        return float("inf"), i0, i1
    j20 = isaac_body_pos_to_hml_joints(body_pos[i0:i1], src_fps=fps)  # hypothesis also sets the true frame rate
    if len(j20) < 2:
        return float("inf"), i0, i1
    feat, _ = joints_to_hml263(j20)
    # drift-invariant comparison: root-relative, heading-normalised joint positions (263-d block 4:67)
    ric = feat[:, 4:67].reshape(-1, 21, 3)
    n = min(len(ric), len(hml_joints))
    err = float(np.linalg.norm(ric[:n] - hml_joints[:n], axis=-1).mean())
    return err, i0, i1


def decide_file(task):
    hf, rows_ = task
    d = joblib.load(os.path.join(HF, hf))
    T = len(d["body_pos"])
    res = dict(hf=hf, T=T, is_succ=bool(d["is_succ"]), errs={}, fps_eff=None, probe=None)
    if not res["is_succ"]:
        return res
    # longest clip, capped: use up to 10 s of it
    best = max(rows_, key=lambda r: (int(r["end_frame"]) if int(r["end_frame"]) != -1 else 10 ** 6) - int(r["start_frame"]))
    name = best["local_id"]
    hml_joints = np.load(os.path.join(HML, "new_joint_vecs", name + ".npy"))[:, 4:67].reshape(-1, 21, 3).astype(np.float64)
    s, e = int(best["start_frame"]), int(best["end_frame"])
    n_hml = len(hml_joints)
    if e == -1 or e - s > 200:
        e = s + min(200, n_hml + 1) if e == -1 else s + 200
        hml_joints = hml_joints[:e - s]
    max_end = max(int(r["end_frame"]) for r in rows_) + head_trim(hf)
    res["feasible"] = {f"{fps:.2f}": bool(max_end / 20.0 * fps <= T + 3) for fps in CANDIDATE_FPS}
    for fps in CANDIDATE_FPS:
        err, i0, i1 = align_error(d["body_pos"], T, s, e, n_hml, fps, hml_joints, head_trim(hf))
        res["errs"][f"{fps:.2f}"] = err if res["feasible"][f"{fps:.2f}"] else float("inf")
    if not any(np.isfinite(v) for v in res["errs"].values()):
        res["fps_eff"] = None; res["probe"] = name; res["margin"] = 0.0
        return res
    k = min(res["errs"], key=res["errs"].get)
    errs = sorted(res["errs"].values())
    res["fps_eff"] = float(k); res["probe"] = name
    res["margin"] = float((errs[1] - errs[0]) / max(errs[0], 1e-6)) if len(errs) > 1 and np.isfinite(errs[1]) else float("inf")
    return res


by_file = collections.defaultdict(list)
for r in rows:
    by_file[r["hf_pkl"]].append(r)
tasks = list(by_file.items())
with Pool(min(24, os.cpu_count())) as pool:
    decisions = pool.map(decide_file, tasks, chunksize=4)
fps_file, seq_len, align_err, subset_fps, undecided = {}, {}, {}, collections.defaultdict(list), []
for dres in decisions:
    seq_len[dres["hf"]] = dres["T"]
    if dres["fps_eff"] is not None:
        align_err[dres["hf"]] = dres["errs"]
        if dres["margin"] >= 0.2:
            fps_file[dres["hf"]] = dres["fps_eff"]
            subset_fps[dres["hf"].split("/")[1]].append(dres["fps_eff"])
        else:
            undecided.append(dres["hf"])
subset_mode = {sname: collections.Counter(v).most_common(1)[0][0] for sname, v in subset_fps.items()}
for hf in undecided:  # ambiguous probes: use the subset majority
    fps_file[hf] = subset_mode.get(hf.split("/")[1], 30.0)
print(f"pass1: {len(undecided)} files ambiguous (margin<0.2) -> subset majority", flush=True)
subset_counts = {sname: dict(collections.Counter(f"{x:.2f}" for x in v)) for sname, v in subset_fps.items()}
best_err = {hf: min(e.values()) for hf, e in align_err.items()}
finite_err = [v for v in best_err.values() if np.isfinite(v)]
misaligned = {hf: e for hf, e in align_err.items() if min(e.values()) > 0.25}
n_infeasible = sum(1 for dres in decisions if dres["is_succ"] and dres["fps_eff"] is None)
print(f"pass1: {len(fps_file)} files aligned ({n_infeasible} succ files where no fps hypothesis fits the index range); "
      f"per-subset confident fps votes: {subset_counts}; median best local joint err {np.median(finite_err):.3f} m; "
      f"{len(misaligned)} files with err>0.25 m  [{time.time()-t0:.0f}s]", flush=True)
json.dump(dict(fps_file=fps_file, align_err=align_err, subset_counts=subset_counts, seq_len=seq_len),
          open(os.path.join(ROOT, "fps_alignment.json"), "w"), indent=1)
unsnapped = {}

# ---- pass 2: slice
out = {sp: collections.defaultdict(list) for sp in ["train", "val", "test"]}
stats = collections.Counter(); per_subset = collections.defaultdict(collections.Counter)
frames = collections.Counter(); dropped = []
for hf, rs in by_file.items():
    subset = hf.split("/")[1]
    d = load(hf)
    T = len(d["body_pos"])
    if not bool(d["is_succ"]):
        for r in rs:
            stats["clip_dropped_tracking_failed"] += 1; per_subset[subset]["failed"] += 1
            dropped.append((r["new_name"], "is_succ=False"))
        continue
    fps_eff = fps_file.get(hf, subset_mode.get(subset, 30.0))
    a_err = best_err.get(hf, float("nan"))
    for r in rs:
        name = r["local_id"]
        sp = split_of.get(name)
        if sp is None:
            stats["clip_dropped_not_in_split"] += 1; dropped.append((name, "not in split")); continue
        s, e = int(r["start_frame"]), int(r["end_frame"])
        n_hml = hml_len(name)
        if e == -1:
            e = s + n_hml + 1
        trim = head_trim(hf)
        t_start, t_end = (s + trim) / 20.0, (e + trim) / 20.0
        i0 = int(round(t_start * fps_eff)); i1 = int(round(t_end * fps_eff))
        i0 = max(0, min(i0, T)); i1 = max(i0, min(i1, T))
        if i1 - i0 < args.min_frames:
            stats["clip_dropped_empty"] += 1; dropped.append((name, f"empty slice {i0}:{i1} of {T}")); continue
        cov = (i1 - i0) / max(1e-6, (t_end - t_start) * fps_eff)
        o = out[sp]
        o["name"].append(name); o["official_id"].append(r["official_id"]); o["split"].append(sp); o["source_file"].append(r["source_npy"]); o["hf_pkl"].append(hf)
        o["start_frame"].append(s); o["end_frame"].append(int(r["end_frame"])); o["fps_eff"].append(float(fps_eff)); o["head_trim_frames"].append(trim)
        o["time_stretch"].append(float(fps_eff / 30.0)); o["phys_i0"].append(i0); o["phys_i1"].append(i1)
        o["n_frames"].append(i1 - i0); o["coverage"].append(float(cov)); o["texts"].append(read_texts(name))
        o["file_align_err_m"].append(float(a_err))
        for k in KEYS:
            o[k].append(np.ascontiguousarray(d[k][i0:i1]).astype(np.float32))
        stats[f"clips_{sp}"] += 1; per_subset[subset]["kept"] += 1
        frames[sp] += i1 - i0
        if cov < 0.9:
            stats["clips_partial_coverage"] += 1
stats["clips_total"] = sum(stats[f"clips_{sp}"] for sp in ["train", "val", "test"])
stats["files_total"] = len(by_file)
stats["files_tracking_success"] = sum(1 for dres in decisions if dres["is_succ"])
stats["files_total_bytes"] = int(sum(os.path.getsize(os.path.join(HF, hf)) for hf in by_file))
stats["official_ids_absent_locally"] = n_no_local
stats["hours_at_30fps"] = {sp: frames[sp] / 30 / 3600 for sp in frames}
stats["hours_total"] = sum(stats["hours_at_30fps"].values())
stats["per_subset"] = {k: dict(v) for k, v in sorted(per_subset.items())}
stats["subset_fps_mode"] = subset_mode
stats["subset_fps_votes"] = subset_counts
stats["n_files_misaligned_gt_0p25m"] = len(misaligned)
stats["misaligned_examples"] = dict(list(misaligned.items())[:20])
stats["median_file_align_err_m"] = float(np.median(finite_err)) if finite_err else None
stats["n_succ_files_no_feasible_fps"] = n_infeasible
stats["n_files_ambiguous_fallback"] = len(undecided)
stats["index_rows_matched"] = len(rows)
print(json.dumps({k: v for k, v in stats.items() if k not in ("per_subset", "misaligned_examples")}, indent=1))
for sp in ["train", "val", "test"]:
    p = os.path.join(ROOT, f"hml_phys_{sp}.pkl")
    joblib.dump(dict(out[sp]), p, compress=3)
    print("saved", p, len(out[sp]["name"]), "clips")
json.dump(dict(stats=stats, dropped=dropped), open(os.path.join(ROOT, "build_stats.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s")
