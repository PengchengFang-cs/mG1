#!/usr/bin/env bash
# One retrieval model on one disjoint clip slice. Slices 0 and 1 become the reward ensemble (minimum,
# per Coste et al.); slice 2 is the judge, which never drives training.
set -uo pipefail
SL=$1; GPU=$2
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
exec > >(tee -a "$R/logs/g1e2e_tmr_s$SL.log") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1; module load gcc/11.5.0 >/dev/null 2>&1
source $R/scripts/activate_h2h.sh
cd $R
export CUDA_VISIBLE_DEVICES=$GPU
echo "### tmr slice $SL start $(date -Is) on $(hostname) gpu $GPU"
python -u scripts/g1e2e_train_tmr.py \
  --rollouts data/g1_e2e/rollouts_train_part1.pkl,data/g1_e2e/rollouts_train_part2.pkl \
  --rollouts-test data/g1_e2e/rollouts_test.ref.pkl \
  --holdout-refs data/g1_e2e/refs_train_part1.pkl --holdout-n 512 \
  --slice $SL/3 --out outputs/g1e2e/tmr_s$SL \
  --steps 6000 --batch 128 --eval-every 500 --log-every 500 --seed $((100+SL)) --device cuda:0
echo "### tmr slice $SL EXIT=$? $(date -Is)"
