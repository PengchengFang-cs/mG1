#!/bin/bash
# source this on a compute node before running anything in human2humanoid / H-ACT
module load gcc/11.5.0
source /scratch/pf2m24/miniconda3/etc/profile.d/conda.sh
conda activate h2h
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/scratch/pf2m24/torch_extensions/h2h
export CC=$(which gcc) CXX=$(which g++)
export TMPDIR=/scratch/pf2m24/tmp
export WANDB_MODE=offline
# Isaac Gym needs this on headless nodes
export MESA_GL_VERSION_OVERRIDE=4.6
# human2humanoid imports `poselib` but neither ships nor declares it. The h2h env inherited UniPhys's
# editable poselib from the clone, whose from_mjcf requires every MJCF body to carry a `pos` attribute;
# g1_21dof.xml omits it on waist_yaw_link and torso_link, exactly as Unitree's official g1_29dof.xml
# does. PYTHONPATH is searched before .pth entries, so this shadows that copy with a fixed one and
# leaves UniPhys's own environment untouched. See external/VENDOR_PATCHES.md.
export PYTHONPATH=/iridisfs/scratch/pf2m24/projects/motion_rebot/external/vendored_for_h2h/poselib:${PYTHONPATH:-}
