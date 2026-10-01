#!/bin/bash
# The remaining FRoM-W1 pipeline, from retargeted clips to a number on the ADAPT Table-1 protocol.
#
#   clips (30 fps, per prompt)
#     -> fromw1_build_schedule.py   20 s schedules, prompt switched every 5-10 s, boundaries root-aligned
#     -> pklpack_to_npz.py          30 -> 50 fps and Kit forward kinematics, inside the Isaac Lab container
#     -> the TextOp tracker follows each schedule
#     -> g1_physical_protocol.py     contact-based falls, Eq. S10/S11, survivors-only quality
#
# usage: GPU=<idx> bash scripts/fromw1_pipeline.sh <n_episodes>
set -uo pipefail
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
N=${1:-2048}
SET=fromw1

echo "=== 1/3 schedules ==="
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/pf2m24/miniconda3/envs/fromw1
python "$R/scripts/fromw1_build_schedule.py" \
  --clips "$R/outputs/fromw1/g1_clips_130" \
  --episodes "$N" \
  --out "$R/outputs/fromw1/schedules_${N}.pkl" || exit 1
conda deactivate

echo "=== 2/3 resample to 50 fps + forward kinematics (container) ==="
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-0}
export ISAACLAB_HOME_TAG=fromw1pack
bash "$R/scripts/isaaclab_exec.sh" bash -lc "cd $R/TextOp/TextOpTracker && /workspace/isaaclab/_isaac_sim/python.sh scripts/pklpack_to_npz.py --headless --input_file $R/outputs/fromw1/schedules_${N}.pkl --output_dir $R/TextOp/TextOpTracker/artifacts/${SET} --input_fps 30 --output_fps 50" || exit 1

echo "=== 3/3 track + score ==="
echo "artifacts/${SET} 下的参考动作数: $(ls -d $R/TextOp/TextOpTracker/artifacts/${SET}/*/ 2>/dev/null | wc -l)"
echo "下一步由 scripts/g1_physical_protocol.py 以 --source tracker 形式跟踪并评测（见 README 注记）"
