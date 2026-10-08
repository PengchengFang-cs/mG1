#!/usr/bin/env bash
# Re-run the closed-loop evaluations that matter, to capture `proprio` -- the input the text<->robot
# retrieval model scores. Those rollouts never saved it, so this is the CLAUDE.md §4 exception:
# re-running only to capture a product that was never stored. Survival numbers from these runs do NOT
# update anything already reported.
set -uo pipefail
GPU=${1:-0}
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
LOG=$REPO/logs/g1e2e_recapture_proprio.log
exec > >(tee -a "$LOG") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1
module load gcc/11.5.0 >/dev/null 2>&1
source $REPO/scripts/activate_h2h.sh
cd $REPO
export CUDA_VISIBLE_DEVICES=$GPU
BASE=outputs/g1e2e/push_G_lr25e6/best.pt
COMMON="--refs data/g1_e2e/refs_train_part1.pkl --text-cache data/g1_e2e/text_clipL14_full \
        --num-envs 512 --episode motion --seed 0 --device cuda:0"

echo "### recapture start $(date -Is) on $(hostname) gpu $GPU"
python -u scripts/g1e2e_eval_closed_loop.py --teacher --hold 1 --policy $BASE $COMMON \
  --out outputs/g1e2e/tmr_teacher.json
echo "### teacher EXIT=$?"
for spec in "bc:" "ppoC:outputs/g1e2e/ppo_ppoC_scale05/latest.pt" \
            "ppoD:outputs/g1e2e/ppo_ppoD_noleash/latest.pt" \
            "ppoAlong:outputs/g1e2e/ppo_ppoA_long/latest.pt"; do
  TAG=${spec%%:*}; RES=${spec#*:}
  EXTRA=""
  [ -n "$RES" ] && EXTRA="--residual $RES"
  python -u scripts/g1e2e_eval_closed_loop.py --policy $BASE $COMMON \
    --hist-init rest --K 1 --cfg-action 1.0 $EXTRA --out outputs/g1e2e/tmr_$TAG.json
  echo "### $TAG EXIT=$?"
done
echo "### recapture done $(date -Is)"
