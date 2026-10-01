#!/usr/bin/env bash
# Extract the downloaded AMASS archives. Extraction is CPU work, so it runs on a compute node:
#   srun --jobid=<id> --overlap --ntasks=1 bash scripts/fromw1_amass_extract.sh
#
# The archives unpack to the dataset folder names that human2humanoid's splits match on
# (scripts/data_process/process_amass_db.py:242-261), which differ from the archive names.
set -euo pipefail
RAW="${1:-/iridisfs/scratch/pf2m24/projects/motion_rebot/data/amass_raw}"
OUT="${2:-/iridisfs/scratch/pf2m24/projects/motion_rebot/data/amass}"
mkdir -p "$OUT"

for f in "$RAW"/*.tar.bz2; do
  b=$(basename "$f" .tar.bz2)
  if [ -f "$OUT/.done_$b" ]; then echo "[skip] $b"; continue; fi
  echo "[tar ] $b"
  tar -xjf "$f" -C "$OUT"
  touch "$OUT/.done_$b"
done

echo "=== extracted dataset folders ==="
find "$OUT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
echo "=== npz count ==="
find "$OUT" -name "*_poses.npz" | wc -l
du -sh "$OUT"
