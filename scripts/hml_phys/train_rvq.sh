#!/bin/bash
# MoMask 6-layer RVQ-VAE, whole-body (docs/08). Approved 2026-09-21 ("三个h200…vq的abc都训练"); architecture fixed to
# MoMask's original by the user on 2026-09-21 ("就用momask的rvq版本…不要走mogeflow的rvq").
# Runs INSIDE a long-lived compute-node step (slurm-allocation skill; srun detached from tmux with setsid):
#   tmux new-session -d -s rvq_<variant> "setsid -w srun --jobid=<job> --overlap bash"
#   tmux send-keys -t rvq_<variant> 'VARIANT=<variant> GPU=0 bash scripts/hml_phys/train_rvq.sh' C-m
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
VARIANT=${VARIANT:?set VARIANT to action|token|state}
OUT=${OUT:-outputs/rvq_${VARIANT}}
STRUCTURE=${STRUCTURE:-whole}
CODE_DIM=${CODE_DIM:-512}
ITERS=${ITERS:-200000}
LOG=logs/hml_phys/train_rvq_${VARIANT}.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_rvq $VARIANT start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=${GPU:-0}
python scripts/hml_phys/train_rvq.py --out $OUT --variant $VARIANT --iters $ITERS \
  --batch 256 --workers ${WORKERS:-12} --nb_code ${NB_CODE:-2048} --code_dim ${CODE_DIM} --structure ${STRUCTURE} --eval_every 5000 --ckpt_every 50000 ${RESUME:+--resume $RESUME}
echo "=== train_rvq $VARIANT exit $? $(date)"
