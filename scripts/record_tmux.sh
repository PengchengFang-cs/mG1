#!/bin/bash
# Usage: GPU=<idx> bash scripts/record_tmux.sh <artifacts_subdir> <meta.pkl> <out.pkl> <num_envs> <repeats>
SET=$1; META=$2; OUT=$3; NENV=${4:-128}; REP=${5:-1}
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-1}
bash scripts/isaaclab_exec.sh bash -lc "cd /iridisfs/scratch/pf2m24/projects/motion_rebot/TextOp/TextOpTracker && /workspace/isaaclab/_isaac_sim/python.sh /iridisfs/scratch/pf2m24/projects/motion_rebot/scripts/record_tracker_rollouts.py --headless --num_envs $NENV --repeats $REP --resume_path logs/rsl_rl/Pretrained/checkpoints/model_75000.pt --motion_glob 'artifacts/$SET/*/motion.npz' --meta_pkl $META --out $OUT env.commands.motion.anchor_body_name=pelvis env.commands.motion.future_steps=5 agent.policy.actor_hidden_dims=[2048,1024,512] agent.policy.critic_hidden_dims=[2048,1024,512]"
