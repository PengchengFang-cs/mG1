#!/usr/bin/env bash
# Residual PPO + its single closed-loop evaluation, chained inside ONE persistent Slurm step.
#
# Everything the run needs lives on the compute node once this starts: nothing here waits on the login
# node, and all output goes to a file on shared storage from inside the step, so losing tmux or the
# login node cannot stop or hang the run (slurm-allocation skill, "Why a properly rooted run survives").
#
#   usage: g1e2e_ppo_launch.sh <tag> <gpu> <residual_scale> <max_hours> <iters> [extra args...]
set -uo pipefail
TAG=$1; GPU=$2; RSCALE=$3; HOURS=$4; ITERS=$5; shift 5
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=$REPO/outputs/g1e2e/ppo_$TAG
LOG=$REPO/logs/g1e2e_ppo_$TAG.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

module load cuda/12.4.0 >/dev/null 2>&1
module load gcc/11.5.0 >/dev/null 2>&1
source $REPO/scripts/activate_h2h.sh
cd $REPO
export CUDA_VISIBLE_DEVICES=$GPU

echo "### $TAG start $(date -Is) on $(hostname) gpu $GPU, residual-scale $RSCALE, max $HOURS h / $ITERS iters"
python -u scripts/g1e2e_train_residual_ppo.py \
  --policy outputs/g1e2e/push_G_lr25e6/best.pt \
  --refs data/g1_e2e/refs_train_part1.pkl \
  --text-cache data/g1_e2e/text_clipL14_full \
  --out "$OUT" --num-envs 512 --iters "$ITERS" --max-hours "$HOURS" \
  --residual-scale "$RSCALE" --save-every 25 --device cuda:0 "$@"
TRAIN=$?
echo "### $TAG TRAIN_EXIT=$TRAIN $(date -Is)"
[ $TRAIN -ne 0 ] && { echo "### $TAG training failed, skipping the evaluation"; exit $TRAIN; }

# Exactly ONE closed-loop rollout of the final residual, on the training prompt pool, per-clip
# episodes, K=1, cfg_action 1.0 -- the same protocol every number in STATUS.md §5.7 was measured
# under, so it is directly comparable to 0.0645 / 0.9576 and to the teacher's 0.0586 / 0.9604.
# One rollout, one computation (CLAUDE.md §4).
echo "### $TAG eval start $(date -Is)"
python -u scripts/g1e2e_eval_closed_loop.py \
  --policy outputs/g1e2e/push_G_lr25e6/best.pt \
  --refs data/g1_e2e/refs_train_part1.pkl \
  --text-cache data/g1_e2e/text_clipL14_full \
  --num-envs 512 --episode motion --hist-init rest --K 1 --seed 0 --cfg-action 1.0 \
  --residual "$OUT/latest.pt" \
  --out outputs/g1e2e/eval_ppo_$TAG.json --device cuda:0
echo "### $TAG EVAL_EXIT=$? $(date -Is)"
echo "### $TAG done $(date -Is)"
