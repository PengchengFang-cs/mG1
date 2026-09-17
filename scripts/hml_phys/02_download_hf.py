"""Step 1b: download only the HF pickles needed for HumanML3D (list from 01_match_index.py).
Network I/O only (allowed on the login node). Retries per file; skips files already present.
"""
import os, sys, time, concurrent.futures as cf
from huggingface_hub import hf_hub_download
REPO = "yan0116/SMPL_Humanoid_offline_dataset"
ROOT = "/iridisfs/scratch/pf2m24/projects/motion_rebot/data/humanml3d_phys"
LOCAL = f"{ROOT}/uniphys_hf"
files = [l.strip() for l in open(f"{ROOT}/hf_download.txt") if l.strip()]
todo = [f for f in files if not os.path.exists(os.path.join(LOCAL, f))]
print(f"{len(files)} needed, {len(files)-len(todo)} present, {len(todo)} to download", flush=True)
def get(f):
    err = None
    for k in range(6):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=LOCAL)
            return f, None
        except Exception as e:
            err = repr(e)[:200]; time.sleep(2 + 5 * k)
    return f, err
done = 0; failed = []
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for f, err in ex.map(get, todo):
        done += 1
        if err: failed.append((f, err))
        if done % 200 == 0 or done == len(todo):
            print(f"{done}/{len(todo)} done, {len(failed)} failed, {time.time()-t0:.0f}s", flush=True)
with open(f"{ROOT}/hf_download_failed.txt", "w") as fo:
    for f, e in failed: fo.write(f"{f}\t{e}\n")
print("DOWNLOAD_DONE failed=%d" % len(failed), flush=True)
