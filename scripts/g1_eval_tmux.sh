#!/bin/bash
# Run the G1 closed-loop evaluation inside the Isaac Lab container.
# Usage: GPU=<idx> bash scripts/g1_eval_tmux.sh <out.pkl> [extra args to scripts/g1_eval_rollout.py...]
OUT=$1; shift
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
cd $R
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-0}
bash scripts/isaaclab_exec.sh bash -lc "cd $R/TextOp/TextOpTracker && /workspace/isaaclab/_isaac_sim/python.sh $R/scripts/g1_eval_rollout.py --headless --out $OUT $* env.commands.motion.anchor_body_name=pelvis env.commands.motion.future_steps=5 agent.policy.actor_hidden_dims=[2048,1024,512] agent.policy.critic_hidden_dims=[2048,1024,512]"
