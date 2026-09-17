#!/bin/bash
# Usage: bash scripts/adapt_play_grid.sh <ckpt> <tag>   (runs inside srun on a compute node)
CK=$1; TAG=$2; ROOT=/iridisfs/scratch/pf2m24/projects/motion_rebot
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-1}
bash $ROOT/scripts/isaaclab_exec.sh bash -lc "cd $ROOT/TextOp/TextOpTracker && PY=/workspace/isaaclab/_isaac_sim/python.sh; for P in stand walk; do for S in 2 10; do for E in 1 5; do echo \"=== prompt=\$P ddim=\$S exec=\$E\"; \$PY $ROOT/scripts/adapt_play.py --headless --num_envs 32 --steps 400 --prompt \$P --switch_s 0,0 --ddim_steps \$S --exec_steps \$E --text_dict $ROOT/data/text_embedding_dict_clip_merged.pkl --motion_glob artifacts/val_subset20/KIT_348_bend_left01_poses/motion.npz --ckpt $CK env.commands.motion.anchor_body_name=pelvis env.commands.motion.future_steps=5 2>&1 | grep -E '^\[play\] prompts|Traceback'; done; done; done" 2>&1 | grep -E "^===|^\[play\]|Traceback" | tee $ROOT/logs/play_grid_$TAG.log
