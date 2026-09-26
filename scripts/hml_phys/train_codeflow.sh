#!/bin/bash
# CodeFlow policy over a frozen RVQ tokenizer (docs/08 §10). Approved by the user 2026-09-21
# ("版本一可以跑2个vq，然后版本二选一个vq来跑这个意图，总共三个训练").
#   tmux new-session -d -s cf_<name> "setsid -w srun --jobid=<job> --overlap bash"
#   tmux send-keys -t cf_<name> 'NAME=<name> RVQ=<ckpt> GPU=0 bash scripts/hml_phys/train_codeflow.sh' C-m
# 码本对照 control (user, 2026-09-22): add RVQ_OBS=outputs/rvq_token/iter_200000.pt with RVQ=the action
# tokenizer -- the policy then OBSERVES the full 435-channel state and GENERATES only the 69 action channels,
# which is route A's task exactly.
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
NAME=${NAME:?set NAME, e.g. v1_action}
RVQ=${RVQ:?set RVQ to a tokenizer checkpoint}
OUT=${OUT:-outputs/cf_${NAME}}
STEPS=${STEPS:-200000}
LOG=logs/hml_phys/train_cf_${NAME}.log
mkdir -p logs/hml_phys
exec > >(tee -a "$LOG") 2>&1
echo "=== train_codeflow $NAME start $(date) on $(hostname), job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}"
module load cuda/12.4.0 gcc/11.5.0 ffmpeg 2>/dev/null
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd /iridisfs/scratch/pf2m24/projects/motion_rebot   # the activate script changes directory
export CUDA_VISIBLE_DEVICES=${GPU:-0}
python scripts/hml_phys/train_codeflow.py --out $OUT --rvq $RVQ --steps $STEPS \
  --batch ${BATCH:-256} --workers ${WORKERS:-8} --eval_every 5000 --ckpt_every 50000 \
  ${RVQ_OBS:+--rvq_obs $RVQ_OBS} ${INTENT:+--intent 1} ${RESUME:+--resume $RESUME}
echo "=== train_codeflow $NAME exit $? $(date)"
