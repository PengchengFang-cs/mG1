#!/bin/bash
# source this on a compute node before running anything in UniPhys
module load gcc/11.5.0
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
conda activate uniphys
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/scratch/pf2m24/torch_extensions/uniphys
export CC=$(which gcc) CXX=$(which g++)
export HF_HOME=/scratch/pf2m24/.cache/huggingface
export TMPDIR=/scratch/pf2m24/tmp
export WANDB_MODE=offline
cd /iridisfs/scratch/pf2m24/projects/motion_rebot/UniPhys
