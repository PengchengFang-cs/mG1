#!/bin/bash
# Convert a name->motion pkl to per-motion npz inside the container. Args: <in.pkl> <out_dir_under_TextOpTracker/artifacts>
IN=$1; OUT=$2
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-1}
bash scripts/isaaclab_exec.sh bash -lc "cd /iridisfs/scratch/pf2m24/projects/motion_rebot/TextOp/TextOpTracker && /workspace/isaaclab/_isaac_sim/python.sh scripts/pklpack_to_npz.py --input_file $IN --output_dir ./artifacts/$OUT --input_fps 50 --output_fps 50 --headless"
