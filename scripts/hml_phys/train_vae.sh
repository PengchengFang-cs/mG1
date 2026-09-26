#!/bin/bash
# Intent VAE (docs/07 §20, MIND §4.2; route C approved by the user 2026-09-19).
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill, detached from tmux's session with setsid):
#   tmux new-session -d -s intent_vae "setsid -w srun --jobid=<job> --overlap bash"
#   tmux send-keys -t intent_vae 'bash scripts/hml_phys/train_vae.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=${OUT:-outputs/intent_vae_v1}
ITERS=${ITERS:-100000}
LOG=logs/hml_phys/train_intent_vae_v1.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_vae start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=0
python scripts/hml_phys/train_intent_vae.py --out $OUT --iters $ITERS --batch 128 --workers 8
echo "=== train_vae exit $? $(date)"
