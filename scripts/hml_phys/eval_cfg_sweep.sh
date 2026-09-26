#!/bin/bash
# CFG sweep on ONE checkpoint, test split, full protocol. Requested by the user 2026-09-18: 1.5 / 5 / 7.5.
set -o pipefail
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
CKPT=$1; BASE=$2; shift 2
for CFG in "$@"; do
  TAG=${BASE}_cfg${CFG}
  bash $D/scripts/hml_phys/eval_one.sh $CKPT $TAG 32 $CFG
done
echo "==== CFG SWEEP DONE"
