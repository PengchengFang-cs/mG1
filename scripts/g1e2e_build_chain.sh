#!/usr/bin/env bash
# Build G1 references for the end-to-end line: the train split first, then the test split.
#
# Parallelism is set by a MEASURED bottleneck, not a guess. On a clean card each fit process uses 840 MB of
# GPU memory and 9% of the GPU, but pegs ONE CPU core at 96% -- the gradient fit is launch-overhead bound, so
# the limit is cores, not the GPU. The job has 24; NSHARDS=20 leaves four for overhead and puts ten processes
# on each card (~90% GPU each). Measured RSS: 20 shards sit at ~70 GB and do not grow, well inside the 200 GB.
#
# (An earlier reading of "9% GPU, 26 GB" was contaminated: the 26 GB belonged to another project's processes
# sharing these cards, and the parallelism decision drawn from it was wrong. CLAUDE.md §10.)
#
# RESUMABLE, because the allocation has a wall clock. On 2026-10-02 job 1678103 hit its 2d12h limit with all
# 20 shards at 23% and every shard lost everything. Now each shard checkpoints every --checkpoint-every clips
# and a rerun of this script skips finished shards and resumes partial ones. Re-running it is always safe.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
# The job id changes every time the allocation is replaced, so take the running one rather than a constant.
JOB=${JOB:-$(squeue -u "$USER" -h -t R -o %i | head -1)}
NSHARDS=${NSHARDS:-20}
NGPU=${NGPU:-2}
# Explicit card list, so a build can be confined to the cards this project should use. The job's cards
# may carry another project (CLAUDE.md §10): on 2026-10-05 card 0 ran eval_supp_metrics_*.py (BraTS /
# ISLE) while this build was confined to card 1. `sh % NGPU` alone always starts at card 0.
GPU_LIST=${GPU_LIST:-}
# Cards whose first shard must wait until the card has no compute apps left. Used when a card still
# carries another project (CLAUDE.md §10): the shards bound to it queue instead of competing.
WAIT_CARDS=${WAIT_CARDS:-}
TRAIN_N=${TRAIN_N:-0}        # 0 = every clip; the 2000-clip subset was only to validate the chain
CKPT_EVERY=${CKPT_EVERY:-25}
cd "$REPO"; mkdir -p data/g1_e2e logs
[ -n "$JOB" ] || { echo "[chain] FAIL: no RUNNING job for $USER; squeue shows nothing to --overlap into"; exit 1; }
echo "[chain] job $JOB, $NSHARDS shards over $NGPU GPUs, checkpoint every $CKPT_EVERY clips"

wait_card_free() {   # block until card $1 reports no compute apps
  local c=$1 n
  while true; do
    n=$(srun --jobid="$JOB" --overlap --ntasks=1 bash -lc \
          "nvidia-smi -i $c --query-compute-apps=pid --format=csv,noheader" 2>/dev/null | grep -c .)
    [ "${n:-1}" -eq 0 ] && break
    echo "[chain] card $c busy ($n compute apps), waiting"
    sleep 120
  done
  echo "[chain] card $c free, launching its shards"
}

stage() {   # stage <split> <sample:0=all>
  local split=$1 sample=$2 pids=() sh gpu n_skip=0 waited=""
  for ((sh = 0; sh < NSHARDS; sh++)); do
    if [ -s "data/g1_e2e/refs_${split}_shard${sh}.pkl" ]; then n_skip=$((n_skip + 1)); continue; fi
    if [ -n "$GPU_LIST" ]; then
      # CONTIGUOUS blocks, not round-robin: with GPU_LIST="1 0" shards 0..9 go to card 1 and 10..19 to
      # card 0. Round-robin would put shard 1 on the card we are waiting for, and the launch loop would
      # block there -- leaving the free card's remaining shards unstarted.
      set -- $GPU_LIST
      shift $(( sh * $# / NSHARDS ))
      gpu=$1
      set --
    else
      gpu=$((sh % NGPU))
    fi
    # The first shard bound to a WAIT_CARDS card blocks until that card is free, so shards queue
    # behind another project instead of competing with it for cores and memory.
    case " $WAIT_CARDS " in
      *" $gpu "*) case " $waited " in
                    *" $gpu "*) ;;
                    *) wait_card_free "$gpu"; waited="$waited $gpu" ;;
                  esac ;;
    esac
    srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "
      source scripts/activate_h2h.sh
      cd $REPO
      export CUDA_VISIBLE_DEVICES=$gpu
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      python -u scripts/g1e2e_build_references.py --split $split --sample $sample --seed 0 \
        --shard $sh --nshards $NSHARDS --device cuda:0 --checkpoint-every $CKPT_EVERY \
        --out data/g1_e2e/refs_${split}_shard${sh}.pkl" >> "logs/g1e2e_refs_${split}_${sh}.log" 2>&1 &
    pids+=($!)
  done
  echo "[chain] $split (sample=$sample): ${#pids[@]} shards launched, $n_skip already complete"
  local bad=0
  for p in "${pids[@]+${pids[@]}}"; do wait "$p" || bad=1; done
  local missing=()
  for ((sh = 0; sh < NSHARDS; sh++)); do
    [ -s "data/g1_e2e/refs_${split}_shard${sh}.pkl" ] || missing+=("$sh")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "[chain] FAIL: $split shards ${missing[*]} produced nothing (bad=$bad)."
    echo "[chain] Their .part.pkl files are kept -- rerun this script to resume from them."
    return 1
  fi
  local got=0 n
  for ((sh = 0; sh < NSHARDS; sh++)); do
    n=$(grep -oE "retargeted [0-9]+" "logs/g1e2e_refs_${split}_${sh}.log" 2>/dev/null | tail -1 | awk '{print $2}')
    got=$((got + ${n:-0}))
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
