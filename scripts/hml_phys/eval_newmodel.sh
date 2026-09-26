#!/bin/bash
# Evaluate ONE newly trained model the way 2026-09-23's sweep showed is necessary:
#   * the optimal CFG MOVES with model quality (10万 ckpt wants <=2.5, 20万 wants 3.5, 100万 wants >=6.5),
#     so a new model must get its own small CFG scan -- reusing 3.5 would be arbitrary.
#   * sampling 10 steps beat 32 on every metric and is 3.2x faster, so 10 steps is the default here.
#   * intent read strength s=0.5 strictly dominated 0.75 and 1.0.
#   * every number is reported under the TRUNCATE convention (CLAUDE.md §2): "exclude fallen" only scores
#     surviving episodes, so configurations with different fall rates are not comparable under it.
# usage: NAME=<outputs subdir> TAG=<ckpt tag> GPU=<n> bash eval_newmodel.sh
set -o pipefail
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
cd $D
NAME=${NAME:?set NAME, e.g. mc_A_nosparse}
TAG=${TAG:-step_200000}
CK=$D/outputs/${NAME}/${TAG}.pt
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
CFGS=${CFGS:-"2.0 2.5 3.5"}
for c in $CFGS; do
  t="${NAME}_$(echo $c | tr -d .)"
  GPU=${GPU:-0} CK=$CK TAG=$t CFG=$c STEPS=${STEPS:-10} S_READ=${S_READ:-0.5} K=${K:-2} \
    bash scripts/hml_phys/eval_gen.sh 2>&1 | grep -a --line-buffered -E "^=== |episodes |^top[123] |^fid |duration_completion"
  # the reporting number: truncate convention, single metric pass
  python scripts/hml_phys/07_eval_rollouts.py --rollouts $D/data/humanml3d_phys/rollouts_${t}_test.pkl \
    --fallen truncate --replications 1 --out $D/data/humanml3d_phys/eval_${t}_test_truncate.json 2>&1 \
    | grep -a --line-buffered -E "^top[123] |^fid " | awk -v n="$t(截断)" '{printf "%s %s=%s\n", n, $1, $2}'
done
echo "==== $NAME $TAG ALL DONE"
