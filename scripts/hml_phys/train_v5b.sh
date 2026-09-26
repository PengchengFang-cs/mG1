#!/bin/bash
# v5b (docs/07 §18.2): v5 without the sentence token -- no text in the joint attention or AdaLN; ALL text only via the
# gated cross-attention (CLIP start token excluded). Requested by the user 2026-09-19.
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill):
#   tmux new-session -d -s mc_train_v5b "setsid -w srun --jobid=1476691 --overlap bash"
#   tmux send-keys -t mc_train_v5b 'bash scripts/hml_phys/train_v5b.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=${OUT:-outputs/mc_v5b}
STEPS=${STEPS:-50000}
LOG=logs/hml_phys/train_mc_v5b.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_v5b start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=0
python scripts/hml_phys/train_mc.py \
  --out $OUT --arch part --hidden 512 --heads 12 --depth 3,6 \
  --t_dist logit_normal --text_xattn 1 --text_mode xattn_only \
  --batch 256 --steps $STEPS --ckpt_every 10000 --eval_every 5000 --workers 8 ${RESUME:+--resume $RESUME}
echo "=== train_v5b exit $? $(date)"
