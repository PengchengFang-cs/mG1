#!/usr/bin/env bash
# TRAIN an intent ablation, then evaluate it end to end, inside one persistent Slurm step.
#
# Every training argument is copied from arm G (outputs/g1e2e/push_G_lr25e6/best.pt's own recorded
# args) so --ablate-intent is the ONLY difference. The inference-side ablation already showed this
# trained policy depends on the intent stream (R@1 0.32 -> 0.056 when it is masked, chance 0.031);
# this asks the complementary question -- trained from scratch WITHOUT intent, can the policy learn
# the semantics some other way from the text alone?
set -uo pipefail
ABL=$1; GPU=$2
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=$R/outputs/g1e2e/intent_abl_$ABL
mkdir -p "$OUT"
exec > >(tee -a "$R/logs/g1e2e_intent_abl_$ABL.log") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1; module load gcc/11.5.0 >/dev/null 2>&1
source $R/scripts/activate_h2h.sh
cd $R
export CUDA_VISIBLE_DEVICES=$GPU

echo "### intent_abl $ABL start $(date -Is) on $(hostname) gpu $GPU"
python -u scripts/g1e2e_train_policy.py \
  --rollouts data/g1_e2e/rollouts_train_part1.pkl,data/g1_e2e/rollouts_train_part2.pkl \
  --rollouts-eval data/g1_e2e/rollouts_test.ref.pkl \
  --text-cache data/g1_e2e/text_clipL14_full --vae outputs/g1e2e/vae_full/best.pt \
  --intent-target proprio --ablate-intent "$ABL" --out "$OUT" \
  --steps 30000 --lr 2.5e-5 --lr-schedule const --lr-warmup 2000 \
  --batch 128 --seed 123 --select chain --eval-every 1000 --log-every 500 --device cuda:0
echo "### $ABL TRAIN_EXIT=$? $(date -Is)"
[ -f "$OUT/best.pt" ] || { echo "### $ABL no best.pt, stopping"; exit 1; }

# Closed loop, same protocol as every other row. The eval script reads `ablate_intent` out of the
# checkpoint and matches it, so the policy is not handed tokens it never learned to read.
echo "### $ABL eval start $(date -Is)"
python -u scripts/g1e2e_eval_closed_loop.py --policy "$OUT/best.pt" \
  --refs $R/data/g1_e2e/refs_train_part1.pkl --text-cache $R/data/g1_e2e/text_clipL14_full \
  --num-envs 512 --episode motion --seed 0 --hist-init rest --K 1 --cfg-action 1.0 \
  --out $R/outputs/g1e2e/intent_abl_$ABL.json --device cuda:0
echo "### $ABL EVAL_EXIT=$? $(date -Is)"

# And the semantic score, against our own robot-motion retrieval model.
python -u scripts/g1e2e_eval_semantic_tmr.py --tmr outputs/g1e2e/tmr/best.pt \
  --npz bc=$R/outputs/g1e2e/tmr_bc.bodypos.npz \
        intent_abl_$ABL=$R/outputs/g1e2e/intent_abl_$ABL.bodypos.npz \
  --real $R/data/g1_e2e/rollouts_test.ref.pkl \
  --out $R/outputs/g1e2e/semantic_intent_abl_$ABL.json --device cuda
echo "### $ABL SEM_EXIT=$? $(date -Is)"
echo "### intent_abl $ABL done $(date -Is)"
