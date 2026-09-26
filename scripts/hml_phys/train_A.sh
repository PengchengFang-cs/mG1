#!/bin/bash
# Route A (docs/07 §21): HIP + IIP + action-only v5 policy, 200k steps, checkpoint every 50k. Approved by the user 2026-09-19.
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill; srun detached from tmux with setsid):
#   tmux new-session -d -s mc_train_A "setsid -w srun --jobid=1476691 --overlap bash"
#   tmux send-keys -t mc_train_A 'bash scripts/hml_phys/train_A.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=${OUT:-outputs/mc_A_v1}
STEPS=${STEPS:-200000}
LOG=logs/hml_phys/train_mc_A_v1.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_A start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=0
python scripts/hml_phys/train_intent_policy.py --out $OUT --steps $STEPS --ckpt_every 50000 --eval_every 5000 \
  --batch 256 --workers 12 ${RESUME:+--resume $RESUME}
echo "=== train_A exit $? $(date)"
