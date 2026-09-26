#!/bin/bash
# Launch one G1 policy training on a given GPU of an already-allocated job.
# usage: g1_run_policy.sh <gpu> <out-dir> <--intent 0|1> [extra args...]
set -uo pipefail
GPU=$1; OUT=$2; shift 2
ROOT=/scratch/pf2m24/projects/motion_rebot
cd "$ROOT"
source scripts/activate_uniphys.sh >/dev/null 2>&1
cd "$ROOT"                      # activate_uniphys.sh leaves the shell in the UniPhys tree
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
mkdir -p logs/hml_phys
echo "[g1] host=$(hostname) gpu=$GPU out=$OUT start=$(date '+%F %T') args=$*"
python scripts/hml_phys/g1_train_policy.py --out "$OUT" "$@" 2>&1 | tee -a "logs/hml_phys/$(basename "$OUT").log"
echo "[g1] exit=$? end=$(date '+%F %T')"
