#!/usr/bin/env bash
# DGPO on the flow action policy, then its single closed-loop evaluation and semantic score,
# all inside ONE persistent Slurm step so losing tmux or the login node cannot stop it.
#
#   usage: g1e2e_dgpo_launch.sh <tag> <gpu> <max_hours> <iters> [extra args...]
set -uo pipefail
TAG=$1; GPU=$2; HOURS=$3; ITERS=$4; shift 4
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
OUT=$R/outputs/g1e2e/dgpo_$TAG
LOG=$R/logs/g1e2e_dgpo_$TAG.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
module load cuda/12.4.0 >/dev/null 2>&1
module load gcc/11.5.0 >/dev/null 2>&1
source $R/scripts/activate_h2h.sh
cd $R
export CUDA_VISIBLE_DEVICES=$GPU

echo "### dgpo $TAG start $(date -Is) on $(hostname) gpu $GPU, max $HOURS h / $ITERS iters, extra: $*"
python -u scripts/g1e2e_train_dgpo.py \
  --policy $R/outputs/g1e2e/push_G_lr25e6/best.pt \
  --refs $R/data/g1_e2e/refs_train_part1.pkl \
  --text-cache $R/data/g1_e2e/text_clipL14_full \
  --tmr $R/outputs/g1e2e/tmr_s0/best.pt,$R/outputs/g1e2e/tmr_s1/best.pt \
  --tmr-judge $R/outputs/g1e2e/tmr_s2/best.pt \
  --out "$OUT" --iters "$ITERS" --max-hours "$HOURS" --device cuda:0 "$@"
echo "### $TAG TRAIN_EXIT=$? $(date -Is)"
[ -f "$OUT/latest.pt" ] || { echo "### $TAG no checkpoint, stopping"; exit 1; }

# One closed-loop rollout under the SAME protocol as every number in STATUS.md section 5.7:
# the 512-clip training prompt pool, per-clip episodes, hist-init rest, K=1, cfg_action 1.0.
# The 150-step training episodes are a training choice; the report stays on the standard protocol.
echo "### $TAG eval start $(date -Is)"
python -u scripts/g1e2e_eval_closed_loop.py --policy "$OUT/latest.pt" \
  --refs $R/data/g1_e2e/refs_train_part1.pkl --text-cache $R/data/g1_e2e/text_clipL14_full \
  --num-envs 512 --episode motion --seed 0 --hist-init rest --K 1 --cfg-action 1.0 \
  --out $R/outputs/g1e2e/eval_dgpo_$TAG.json --device cuda:0
echo "### $TAG EVAL_EXIT=$? $(date -Is)"

# Scored by the JUDGE (slice 2), which never entered the reward, next to pure cloning. The manifold
# monitor runs in the judge's space for the reason section 5.18 measured: in the reward model's own
# space the outlier statistic carries no signal.
python -u scripts/g1e2e_eval_semantic_tmr.py --tmr $R/outputs/g1e2e/tmr_s2/best.pt \
  --npz bc=$R/outputs/g1e2e/tmr_bc.bodypos.npz \
        dgpo_$TAG=$R/outputs/g1e2e/eval_dgpo_$TAG.bodypos.npz \
  --real $R/data/g1_e2e/rollouts_test.ref.pkl \
  --out $R/outputs/g1e2e/semantic_dgpo_$TAG.json --device cuda
echo "### $TAG SEM_EXIT=$? $(date -Is)"
echo "### dgpo $TAG done $(date -Is)"
