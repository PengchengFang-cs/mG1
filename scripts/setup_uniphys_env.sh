#!/bin/bash
# Build the `uniphys` conda env on a compute node (needs the SOCKS tunnel for internet).
# Usage: srun --jobid=<job> --overlap --ntasks=1 bash -lc 'bash scripts/setup_uniphys_env.sh [stage]'
# stages: torch | deps | all (default all)
set -o pipefail
STAGE="${1:-all}"
ROOT=/iridisfs/scratch/pf2m24/projects/motion_rebot
REPO=$ROOT/UniPhys
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh
tunnel_check
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/scratch/pf2m24/miniconda3/pkgs
export PIP_NO_CACHE_DIR=0

if [[ "$STAGE" == "torch" || "$STAGE" == "all" ]]; then
  echo "=== [torch] create env"
  conda env list | grep -q '^uniphys ' || conda create -y -n uniphys python=3.8 || exit 1
  conda activate uniphys
  conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia || exit 1
  python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
fi

if [[ "$STAGE" == "deps" || "$STAGE" == "all" ]]; then
  echo "=== [deps] poselib / isaac_utils / requirements"
  conda activate uniphys
  cd $REPO/poselib && pip install -e . || exit 1
  cd $REPO/isaac_utils && pip install -e . || exit 1
  cd $REPO && pip install -r requirements.txt || exit 1
  python -c "import lightning, diffusers, clip, smplx, hydra; print('deps ok')" || exit 1
fi
echo "=== done stage $STAGE"
tunnel_down
