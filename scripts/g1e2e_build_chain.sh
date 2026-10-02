#!/usr/bin/env bash
# Build G1 references for the end-to-end line: a train subset first, then the full test split.
#
# Parallelism is set by a MEASURED bottleneck, not a guess. On a clean card each fit process uses 840 MB of
# GPU memory and 9% of the GPU, but pegs ONE CPU core at 96% -- the gradient fit is launch-overhead bound, so
# the limit is cores, not the GPU. The job has 24; NSHARDS=20 leaves four for overhead and puts ten processes
# on each card (~90% GPU each). That takes the 3,640 clips from ~11 h at two shards to about an hour.
#
# (An earlier reading of "9% GPU, 26 GB" was contaminated: the 26 GB belonged to another project's processes
# sharing these cards, and the parallelism decision drawn from it was wrong. CLAUDE.md §10.)
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-1678103}
NSHARDS=${NSHARDS:-20}
NGPU=${NGPU:-2}
TRAIN_N=${TRAIN_N:-2000}
cd "$REPO"; mkdir -p data/g1_e2e logs

stage() {   # stage <split> <sample:0=all>
  local split=$1 sample=$2 pids=() sh gpu
  echo "[chain] $split (sample=$sample): $NSHARDS shards over $NGPU GPUs"
  for ((sh = 0; sh < NSHARDS; sh++)); do
    gpu=$((sh % NGPU))
    srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "
      source scripts/activate_h2h.sh
      cd $REPO
      export CUDA_VISIBLE_DEVICES=$gpu
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      python -u scripts/g1e2e_build_references.py --split $split --sample $sample --seed 0 \
        --shard $sh --nshards $NSHARDS --device cuda:0 \
        --out data/g1_e2e/refs_${split}_shard${sh}.pkl" > "logs/g1e2e_refs_${split}_${sh}.log" 2>&1 &
    pids+=($!)
  done
  local bad=0
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [ "$bad" = 0 ] || { echo "[chain] FAIL: a $split shard exited nonzero"; return 1; }
  local got=0
  for ((sh = 0; sh < NSHARDS; sh++)); do
    [ -s "data/g1_e2e/refs_${split}_shard${sh}.pkl" ] || { echo "[chain] FAIL: $split shard$sh produced nothing"; return 1; }
    got=$((got + $(grep -oE "retargeted [0-9]+" "logs/g1e2e_refs_${split}_${sh}.log" | tail -1 | awk '{print $2}')))
  done
  echo "[chain] $split: $got clips retargeted"
}

stage train "$TRAIN_N" || exit 1
stage test 0 || exit 1

echo "[chain] merging and checking the world frame"
srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "
  source scripts/activate_h2h.sh
  cd $REPO
  python -u scripts/g1e2e_merge_refs.py --out-dir data/g1_e2e" || { echo "[chain] FAIL: merge"; exit 1; }
echo "[chain] done"
