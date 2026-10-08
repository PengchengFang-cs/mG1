#!/usr/bin/env bash
# Phase 1: evaluation only, no training.
#   group A  repeat rollouts of two configurations, to measure the SEMANTIC jitter the way §5.12
#            measured the survival jitter. Permitted by CLAUDE.md §4a: necessary, because the
#            leg-masked residual's -0.05 R@1 cannot be read without a noise floor.
#   group B  ablate the MIND intent stream in the closed loop on the already-trained arm G.
#            Free -- no retraining -- and it answers whether the policy DEPENDS on the intent.
set -uo pipefail
GROUP=$1; GPU=$2
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
exec > >(tee -a "$R/logs/g1e2e_phase1_$GROUP.log") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1; module load gcc/11.5.0 >/dev/null 2>&1
source $R/scripts/activate_h2h.sh
cd $R
export CUDA_VISIBLE_DEVICES=$GPU
BASE=$R/outputs/g1e2e/push_G_lr25e6/best.pt
C="--refs $R/data/g1_e2e/refs_train_part1.pkl --text-cache $R/data/g1_e2e/text_clipL14_full \
   --num-envs 512 --episode motion --seed 0 --hist-init rest --K 1 --cfg-action 1.0 --device cuda:0"
run () { # name, extra args...
  local n=$1; shift
  python -u scripts/g1e2e_eval_closed_loop.py --policy $BASE $C "$@" --out $R/outputs/g1e2e/$n.json 2>&1 | tail -2
  echo "### $n EXIT=${PIPESTATUS[0]}"
}
echo "### phase1 $GROUP start $(date -Is) on $(hostname) gpu $GPU"
if [ "$GROUP" = A ]; then
  run p1_bc_r2
  run p1_ppoE_r2 --residual $R/outputs/g1e2e/ppo_ppoE_legs/latest.pt
else
  run p1_intent_all --ablate-intent all
  run p1_intent_hip --ablate-intent hip
  run p1_intent_iip --ablate-intent iip
fi
echo "### phase1 $GROUP done $(date -Is)"
