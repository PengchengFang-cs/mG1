#!/bin/bash
# Re-collect tracker rollouts with ADAPT Table S4's full domain randomisation (including the actuator gains
# the TextOp EventCfg lacks).
# usage: GPU=<idx> bash scripts/record_dr_tmux.sh <artifacts_subdir> <meta.pkl> <out.pkl> [num_envs]
SET=$1; META=$2; OUT=$3; NENV=${4:-256}
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
cd $R
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPU:-0}
export ISAACLAB_HOME_TAG=${TAG:-$(basename "$OUT" .pkl)}
bash scripts/isaaclab_exec.sh bash -lc "cd $R/TextOp/TextOpTracker && /workspace/isaaclab/_isaac_sim/python.sh $R/scripts/record_tracker_rollouts.py --headless --num_envs $NENV --repeats 1 --actuator_dr ${ADR:-1} ${NODR:+--no_dr 1} --dr_mode ${DRMODE:-startup} ${PUSH:+--keep_pushes} --resume_path logs/rsl_rl/Pretrained/checkpoints/model_75000.pt --motion_glob 'artifacts/$SET/*/motion.npz' --meta_pkl $META --out $OUT env.commands.motion.anchor_body_name=pelvis env.commands.motion.future_steps=5 agent.policy.actor_hidden_dims=[2048,1024,512] agent.policy.critic_hidden_dims=[2048,1024,512]"
