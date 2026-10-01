#!/bin/bash
# Run H-GPT text-to-motion with our local configs. usage: GPU=<idx> bash scripts/fromw1_demo.sh <prompts.txt>
set -uo pipefail
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
PROMPTS=${1:-$R/data/fromw1_smoke_prompt.txt}
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/pf2m24/miniconda3/envs/fromw1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1
cd "$R/external/FRoM-W1/H-GPT"
echo "[hgpt] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES prompts=$PROMPTS"
python "$R/scripts/fromw1_gen.py" \
  --cfg_assets "$R/configs_fromw1/assets.yaml" \
  --cfg "$R/configs_fromw1/exp_motionx_cot_2k_local.yaml" \
  --task t2m --example "$PROMPTS"
echo "[hgpt] exit=$?"
