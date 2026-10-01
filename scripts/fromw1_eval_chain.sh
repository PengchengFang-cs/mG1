#!/usr/bin/env bash
# Wait for the retarget shards, build the 29-DoF library (world-frame check included), then evaluate
# both released G1 students on it. Each stage aborts the chain on failure so a bad library can never
# reach the evaluation.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-1678103}
cd "$REPO"

run() { srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "$1"; }

echo "[chain] waiting for the retarget shards"
while pgrep -f "fromw1_amass_to_g1_21dof.py --amass" >/dev/null; do sleep 30; done
for s in 0 1; do
  [ -s "data/g1_21dof/v2_shard$s.pkl" ] || { echo "[chain] FAIL: v2_shard$s.pkl missing"; exit 1; }
done
echo "[chain] shards done:"
for s in 0 1; do
  tr '\r' '\n' < "logs/g1_retarget_v2_shard$s.log" | grep -E "retargeted|fit error" | sed "s/^/  shard$s /"
done

echo "[chain] relayout 21 -> 29 (+ world-frame check)"
run "source scripts/activate_h2h.sh && cd $REPO && python -u scripts/fromw1_relayout_21to29.py \
  --inputs data/g1_21dof/v2_shard0.pkl data/g1_21dof/v2_shard1.pkl \
  --out data/g1_21dof/v2_29layout.pkl" || { echo "[chain] FAIL: relayout"; exit 1; }

N=$(ls -l data/g1_21dof/v2_29layout.pkl | awk '{print $5}')
[ "$N" -gt 1000000 ] || { echo "[chain] FAIL: library too small"; exit 1; }

# One env per motion means a single pass. Their config_eval uses 406, which presumably matched the size
# of their own (unreleased) eval set.
NM=$(run "source scripts/activate_h2h.sh && python -c \"
import joblib; print(len(joblib.load('$REPO/data/g1_21dof/v2_29layout.pkl')))\"" 2>/dev/null | tail -1)
echo "[chain] library holds $NM clips"

for spec in "full:25_12_11_18-16-37_OmniH2O_STUDENT" "clean:25_12_11_18-18-10_OmniH2O_STUDENT_FILTER"; do
  tag=${spec%%:*}; run_name=${spec#*:}
  echo "[chain] evaluating $tag ($run_name)"
  run "source scripts/activate_h2h.sh && cd $REPO && export CUDA_VISIBLE_DEVICES=0 && \
    python -u scripts/fromw1_eval_g1_policy.py \
      --motion-file data/g1_21dof/v2_29layout.pkl \
      --load-run $run_name --num-envs $NM --device cuda:0 \
      --out outputs/fromw1/eval_g1_${tag}_v2.json" > "logs/eval_g1_${tag}_v2.log" 2>&1
  if grep -qE "^Success Rate" "logs/eval_g1_${tag}_v2.log"; then
    echo "[chain] $tag:"
    grep -E "^Success Rate|^All: |^Succ: " "logs/eval_g1_${tag}_v2.log" | sed 's/^/    /'
  else
    echo "[chain] FAIL: $tag produced no metrics; tail of log:"
    grep -vE "^Importing module|^Setting GYM|generating randomized" "logs/eval_g1_${tag}_v2.log" | tail -15
  fi
done
echo "[chain] done"
