#!/usr/bin/env bash
# Train the text<->G1-motion retrieval model, inside one persistent Slurm step.
set -uo pipefail
GPU=${1:-0}
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=$REPO/outputs/g1e2e/tmr
LOG=$REPO/logs/g1e2e_tmr.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1
module load gcc/11.5.0 >/dev/null 2>&1
source $REPO/scripts/activate_h2h.sh
cd $REPO
export CUDA_VISIBLE_DEVICES=$GPU
echo "### tmr start $(date -Is) on $(hostname) gpu $GPU"
python -u scripts/g1e2e_train_tmr.py \
  --rollouts data/g1_e2e/rollouts_train_part1.pkl,data/g1_e2e/rollouts_train_part2.pkl \
  --rollouts-test data/g1_e2e/rollouts_test.ref.pkl \
  --holdout-refs data/g1_e2e/refs_train_part1.pkl --holdout-n 512 \
  --out "$OUT" --steps 20000 --batch 128 --eval-every 500 --log-every 100 --device cuda:0
echo "### tmr TRAIN_EXIT=$? $(date -Is)"
echo "### tmr done $(date -Is)"
