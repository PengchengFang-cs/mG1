#!/bin/bash
# v4 (docs/07 §17): MoGeFlow-style part-structured policy. Approved by the user 2026-09-19.
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill):
#   tmux new-session -d -s mc_train_v4 "setsid -w srun --jobid=1398710 --overlap bash"
#   tmux send-keys -t mc_train_v4 'bash scripts/hml_phys/train_v4.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=${OUT:-outputs/mc_v4}
STEPS=${STEPS:-50000}
LOG=logs/hml_phys/train_mc_v4.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_v4 start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
python scripts/hml_phys/train_mc.py \
  --out $OUT --arch part --hidden 512 --heads 12 --depth 3,6 \
  --t_dist logit_normal \
  --batch 256 --steps $STEPS --ckpt_every 10000 --eval_every 5000 --workers 8 ${RESUME:+--resume $RESUME}
echo "=== train_v4 exit $? $(date)"
