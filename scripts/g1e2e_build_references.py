"""Build G1 21-DoF reference motions for the HumanML3D clips, with their captions.

This is step 1 of the end-to-end G1 line: text -> G1 action, trained by behaviour cloning on a tracker's
rollouts, the recipe every work in this area uses (docs/00 §数据). MIND conditions on HumanML3D's
sentence-level captions and that is what its holistic-intent predictor is for, so the references are the
HumanML3D clips rather than all of AMASS -- a single BABEL word has no holistic intent to extract.

The HumanML3D -> AMASS correspondence is not re-derived here. `data/humanml3d_phys/hml_phys_{train,test}.pkl`
already carries, per clip, the local id that `texts/` and the splits use, the AMASS source file, the crop,
and the captions; that matching was done and validated for the MIND line (scripts/hml_phys/03_build_dataset.py).

The crop convention, from that script's header: HumanML3D frame k of a clip is
`pose_data[trim + start_frame + k]` at 20 fps, where `trim` is HumanML3D's dataset-specific head trim
applied before index.csv. So the clip spans [(start+trim)/20, (end+trim)/20) seconds of the AMASS sequence,
and the raw-AMASS crop is that span times the file's own `mocap_framerate`. The 20 fps here is HumanML3D's
convention and has nothing to do with the 30 fps the motion library is resampled to.

Run on a compute node, one shard per GPU:
  srun --jobid=<id> --overlap --ntasks=1 bash -lc '
    source scripts/activate_h2h.sh
    python scripts/g1e2e_build_references.py --split train --shard 0 --nshards 2 \
      --out data/g1_e2e/refs_train_shard0.pkl'
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HML_PHYS = REPO / "data/humanml3d_phys"
AMASS = REPO / "data/amass"
# 03_build_dataset.py:85 -- HumanML3D's raw_pose_processing trims the head of some subsets BEFORE
# index.csv is applied, at 20 fps: Eyes_Japan / MPI_HDM05 3 s, TotalCapture / MPI_Limits 1 s,
# Transitions_mocap 0.5 s.
HEAD_TRIM_20FPS = {"Eyes_Japan_Dataset": 60, "MPI_HDM05": 60, "TotalCapture": 20,
                   "MPI_Limits": 20, "Transitions_mocap": 10}
HML_FPS = 20.0


def amass_path(source_file):
    """'./pose_data/KIT/3/jump_left02_poses.npy' -> data/amass/KIT/3/jump_left02_poses.npz"""
    rel = "/".join(source_file.split("/")[2:])
    return AMASS / (rel[:-4] + ".npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="fixed random subset of N clips (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=1000, help="Adam steps per clip (their default)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="write partial shard state every N clips, so a wall-clock kill costs at most "
                         "that many clips instead of the whole shard")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing .part.pkl and start this shard from scratch")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    d = joblib.load(HML_PHYS / f"hml_phys_{args.split}.pkl")
    n = len(d["name"])
    clips = [
        dict(name=str(d["name"][i]), official_id=str(d["official_id"][i]), split=str(d["split"][i]),
             source_file=str(d["source_file"][i]), start=int(d["start_frame"][i]),
             end=int(d["end_frame"][i]), texts=d["texts"][i])
        for i in range(n)
    ]
    print(f"{args.split}: {len(clips)} clips from {len({c['source_file'] for c in clips})} AMASS files")

    # An open-ended clip (end_frame == -1) would need the HumanML3D feature length to resolve; none of the
    # clips the MIND line kept are open-ended, so refuse rather than guess if that ever changes.
    open_ended = [c["name"] for c in clips if c["end"] == -1]
    assert not open_ended, f"{len(open_ended)} clips have end_frame == -1, e.g. {open_ended[:3]}"

    if args.sample and args.sample < len(clips):
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(len(clips), size=args.sample, replace=False).tolist())
        clips = [clips[i] for i in pick]
        manifest = out.with_name(out.stem + ".manifest.txt")
        manifest.write_text("\n".join(c["name"] for c in clips) + "\n")
        print(f"  sampled {len(clips)} (seed {args.seed}) -> {manifest.name}")
    if args.nshards > 1:
        clips = clips[args.shard::args.nshards]
        print(f"  shard {args.shard}/{args.nshards} -> {len(clips)} clips")

    import sys
    sys.path.insert(0, str(REPO))
    from hml_phys.g1_retarget import Retargeter

    rt = Retargeter(device=args.device)
    print(f"robot    : g1_21dof.xml, {len(rt.cfg.actuated_joint_names)} actuated joints, "
          f"order verified against the policy")
    print(f"layout   : 29-DoF, pose_aa rows = {len(rt.names29) + 3}")

    from tqdm import tqdm
    lib, text, skipped = {}, {}, {}
    # Checkpoint / resume. A shard takes hours and the allocation has a wall clock: on 2026-10-02 the job
    # hit its 2d12h limit with 20 shards at 23% and every one of them lost everything, because the shard
    # pkl was only written at the end. Partial state now lands every --checkpoint-every clips and a
    # restart skips what is already there.
    part = out.with_suffix(".part.pkl")
    if part.exists() and not args.restart:
        done = joblib.load(part)
        lib, text, skipped = done["lib"], done["text"], done["skipped"]
        print(f"resuming from {part.name}: {len(lib)} retargeted, {len(skipped)} skipped already")
    seen = set(lib) | set(skipped)
    clips = [c for c in clips if c["name"] not in seen]
    if seen:
        print(f"{len(clips)} clips left to do")

    def checkpoint():
        tmp = part.with_suffix(".tmp")
        joblib.dump(dict(lib=lib, text=text, skipped=skipped), tmp)
        tmp.replace(part)       # atomic: a kill mid-write leaves the previous checkpoint intact

    t0 = time.time()
    for n_done, c in enumerate(tqdm(clips, desc=f"retarget {args.split}"), 1):
        if n_done % args.checkpoint_every == 0:
            checkpoint()
        npz = amass_path(c["source_file"])
        if not npz.exists():
            skipped[c["name"]] = f"missing AMASS file {npz.name}"
            continue
        raw = dict(np.load(open(npz, "rb"), allow_pickle=True))
        fr = raw.get("mocap_framerate", raw.get("mocap_frame_rate"))
        if fr is None or "poses" not in raw or "trans" not in raw:
            skipped[c["name"]] = f"missing keys: {sorted(raw)}"
            continue
        fr = float(fr)

        trim = HEAD_TRIM_20FPS.get(c["source_file"].split("/")[2], 0)
        # HumanML3D strides the raw file by an INTEGER down_sample = int(fps/20) before applying the head
        # trim and index.csv, so its frame m is raw frame m * stride -- not round(m * fps / 20). The two
        # agree only when fps is a multiple of 20. Verified against HumanML3D's own new_joints root height
        # on 250 fps clips: stride int(250/20)=12 correlates 1.0000, while 250/20=12.5 gives 0.69 and
        # worse. 99 train and 13 test clips are affected, by up to 1.09 s, and the 59.99998 fps files
        # (int(fr/20)=2, i.e. HumanML3D sampled them at 30 fps) were off by a 1.5x time scale.
        # scripts/hml_phys/03_build_dataset.py uses the /20 form too; it escaped only because it picked
        # fps_eff per file by content alignment, so it does not validate the assumption.
        stride = max(1, int(fr / HML_FPS))
        T = raw["poses"].shape[0]
        j0 = max(0, min((c["start"] + trim) * stride, T))
        j1 = max(j0, min((c["end"] + trim) * stride, T))
        if (j1 - j0) / fr < 0.5:            # under half a second of source is not a usable clip
            skipped[c["name"]] = f"crop too short: {j1 - j0} raw frames at {fr:g} fps"
            continue

        entry = rt.fit(raw["poses"][j0:j1], raw["trans"][j0:j1], fr, max_iter=args.max_iter)
        if entry is None:
            skipped[c["name"]] = "fewer than 10 frames after resampling to 30 fps"
            continue
        lib[c["name"]] = entry
        text[c["name"]] = dict(split=c["split"], official_id=c["official_id"],
                               source_file=c["source_file"], captions=[t["caption"] for t in c["texts"]],
                               n_frames=int(entry["dof"].shape[0]), fit_err_m=entry["fit_err_m"],
                               fps=float(entry["fps"]), mocap_fps=float(fr), hml_stride=int(stride),
                               hml_fps_eff=float(fr) / stride)

    dt = time.time() - t0
    print(f"\nretargeted {len(lib)} / {len(clips)} in {dt:.0f}s  ({dt / max(1, len(lib)):.1f}s per clip)")
    if skipped:
        from collections import Counter
        print(f"skipped {len(skipped)}:")
        for why, k in Counter(v.split(":")[0] for v in skipped.values()).most_common():
            print(f"  {k:6d}  {why}")
    if lib:
        errs = np.array([v["fit_err_m"] for v in lib.values()])
        frames = np.array([v["dof"].shape[0] for v in lib.values()])
        ncap = np.array([len(v["captions"]) for v in text.values()])
        print(f"fit error (m): mean {errs.mean():.4f}  median {np.median(errs):.4f}  "
              f"p95 {np.percentile(errs, 95):.4f}  max {errs.max():.4f}")
        print(f"frames: {frames.sum()} total, {frames.sum() / 30 / 60:.1f} min; "
              f"per clip min {frames.min()} median {int(np.median(frames))} max {frames.max()}")
        print(f"captions per clip: min {ncap.min()} median {int(np.median(ncap))} max {ncap.max()}")

    joblib.dump(lib, out)
    out.with_suffix(".text.json").write_text(json.dumps(text, ensure_ascii=False, indent=1))
    joblib.dump(skipped, out.with_suffix(".skipped.pkl"))
    part.unlink(missing_ok=True)        # the shard is complete; the resume state is no longer needed
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"      {out.with_suffix('.text.json').name}")


if __name__ == "__main__":
    main()
