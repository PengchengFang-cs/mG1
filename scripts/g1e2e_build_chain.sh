#!/usr/bin/env bash
# Build G1 references for the end-to-end line: a 2000-clip train subset first, then the full test split.
#
# The train subset is deliberately not the full 8734: at ~22 s/clip on one GPU the full set is ~32 GPU-hours,
# and there is no point spending that before the recorder and the training path are known to work. The test
# split is built in full because it is the evaluation set and will be reused unchanged.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-1678103}
TRAIN_N=${TRAIN_N:-2000}
cd "$REPO"; mkdir -p data/g1_e2e logs

stage() {   # stage <split> <sample:0=all>
  local split=$1 sample=$2 pids=()
  echo "[chain] $split (sample=$sample) on 2 GPUs"
  for sh in 0 1; do
    srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "
      source scripts/activate_h2h.sh
      cd $REPO
      export CUDA_VISIBLE_DEVICES=$sh
      python -u scripts/g1e2e_build_references.py --split $split --sample $sample --seed 0 \
        --shard $sh --nshards 2 --device cuda:0 \
        --out data/g1_e2e/refs_${split}_shard${sh}.pkl" > "logs/g1e2e_refs_${split}_${sh}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "[chain] FAIL: $split shard exited nonzero"; return 1; }; done
  for sh in 0 1; do
    [ -s "data/g1_e2e/refs_${split}_shard${sh}.pkl" ] || { echo "[chain] FAIL: $split shard$sh produced nothing"; return 1; }
    grep -E "retargeted|fit error|frames:|captions" "logs/g1e2e_refs_${split}_${sh}.log" | sed "s/^/  $split.$sh /"
  done
}

stage train "$TRAIN_N" || exit 1
stage test 0 || exit 1

echo "[chain] merging and checking the world frame"
srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "
  source scripts/activate_h2h.sh
  cd $REPO
  python -u scripts/g1e2e_merge_refs.py --out-dir data/g1_e2e" || { echo "[chain] FAIL: merge"; exit 1; }
echo "[chain] done"
