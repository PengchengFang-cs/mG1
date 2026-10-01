#!/bin/bash
# Build the H-GPT inference environment for the FRoM-W1 reproduction.
#
# Two things learned the hard way:
#  * A freshly created conda env ships a pip whose vendored urllib3 clashes with PySocks, so every install
#    through the SSH tunnel dies with "PoolKey.__new__() got an unexpected keyword argument". The project's
#    existing `uniphys` env has pip 24.2, which works through the same tunnel, so clone that instead of
#    building from scratch -- it also already carries torch, numpy 1.23, pytorch_lightning, smplx, chumpy,
#    omegaconf, trimesh and scipy, leaving only three packages to add.
#  * Clone rather than install into `uniphys` directly: that env runs the ADAPT reproduction and a
#    transformers install can drag shared dependencies with it.
set -uo pipefail
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
BASE=$(conda env list | awk '/uniphys/{print $NF; exit}')
ENV=/scratch/pf2m24/miniconda3/envs/fromw1

echo "[env] host=$(hostname)  base=$BASE"
if [ ! -x "$ENV/bin/python" ]; then
  rm -rf "$ENV"
  conda create -y --clone "$BASE" -p "$ENV" || { echo "[env] clone FAILED"; exit 1; }
fi
conda activate "$ENV"
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh >/dev/null 2>&1
export PIP_CACHE_DIR=/scratch/pf2m24/.cache/pip
echo "[env] $(python -V 2>&1)  pip $(pip -V | cut -d' ' -f2)"

pip install --no-input "transformers==4.45.1" peft accelerate || echo "[env] pip FAILED"

echo "[env] ---- versions ----"
python "$R/scripts/fromw1_envcheck.py"
echo "[env] done"
