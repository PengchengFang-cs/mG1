#!/bin/bash
# Route A v2 (docs/07 §21.8): route A + holistic-target augmentation, conditioning augmentation, chain-based selection,
# span scalars. 200k steps, checkpoint every 50k. Approved by the user 2026-09-19 ("用修好的跑").
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill; srun detached from tmux with setsid):
#   tmux new-session -d -s mc_train_A2 "setsid -w srun --jobid=1476691 --overlap bash"
#   tmux send-keys -t mc_train_A2 'bash scripts/hml_phys/train_A2.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=${OUT:-outputs/mc_A_v2}
STEPS=${STEPS:-200000}
LOG=logs/hml_phys/train_$(basename $OUT).log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_A2 start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=${GPU:-0}
python scripts/hml_phys/train_intent_policy.py --out $OUT --steps $STEPS --ckpt_every ${CKPT_EVERY:-200000} --eval_every ${EVAL_EVERY:-20000} \
  --batch 256 --workers 12 --F_act ${F_ACT:-4} --hip_aug 1 --span_scalars 1 --cond_aug 0.5 --cond_aug_test 0.75 --select chain ${RESUME:+--resume $RESUME}
echo "=== train_A2 exit $? $(date)"
