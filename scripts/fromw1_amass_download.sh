#!/usr/bin/env bash
# Download the AMASS subsets human2humanoid's AMASS preparation expects (SMPL+H, gender-specific).
#
# The archive names on amass.is.tue.mpg.de differ from the folder names the processing scripts match on
# (scripts/data_process/process_amass_db.py:242-261), so the mapping is spelled out below.
#
# Credentials come from a file OUTSIDE the repo so they are never committed:
#   AMASS_CREDS=/path/to/creds  (two lines: AMASS_USER=... / AMASS_PASS=...)
# Downloads run on the login node -- compute nodes have no outbound network. Extraction is CPU work and
# must go through srun; this script only fetches.
set -euo pipefail

: "${AMASS_CREDS:?set AMASS_CREDS to a file holding AMASS_USER / AMASS_PASS}"
set -a; . "$AMASS_CREDS"; set +a
: "${AMASS_USER:?}"; : "${AMASS_PASS:?}"

OUT="${1:-/iridisfs/scratch/pf2m24/projects/motion_rebot/data/amass_raw}"
mkdir -p "$OUT"; cd "$OUT"

BASE="https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results"

# archive name on the download page -> folder name after extraction (what their splits match on)
ARCHIVES=(
  CMU                 # CMU
  PosePrior           # MPI_Limits
  TotalCapture        # TotalCapture
  EyesJapanDataset    # Eyes_Japan_Dataset
  KIT                 # KIT
  BMLrub              # BioMotionLab_NTroje  (their "BML" / "BioMotionLab")
  EKUT                # EKUT
  TCDHands            # TCD_handMocap
  BMLhandball         # BMLhandball
  DanceDB             # DanceDB
  ACCAD               # ACCAD
  BMLmovi             # BMLmovi
  DFaust              # DFaust_67
  HumanEva            # HumanEva
  HDM05               # MPI_HDM05
  SFU                 # SFU
  MoSh                # MPI_mosh
  Transitions         # Transitions_mocap
  SSM                 # SSM_synced
)

# download.is.tue.mpg.de authenticates per request: the form on its sign-in page posts back to the same
# URL, so the credentials ride along with the file request itself.
for name in "${ARCHIVES[@]}"; do
  f="$name.tar.bz2"
  if [ -s "$f" ] && bzip2 -t "$f" 2>/dev/null; then
    echo "[skip] $f already complete"
    continue
  fi
  echo "[get ] $f"
  curl -fSL --retry 5 --retry-delay 10 --connect-timeout 30 \
    --cookie-jar .dl_cookies --cookie .dl_cookies -X POST \
    --data-urlencode "username=$AMASS_USER" \
    --data-urlencode "password=$AMASS_PASS" \
    --data-urlencode "commit=Log in" \
    "$BASE/$f" -o "$f.part"
  mv "$f.part" "$f"
  printf '[ok  ] %s  %s\n' "$f" "$(du -h "$f" | cut -f1)"
done

echo "=== done ==="
du -sh "$OUT"; ls -l "$OUT"/*.tar.bz2 | wc -l
