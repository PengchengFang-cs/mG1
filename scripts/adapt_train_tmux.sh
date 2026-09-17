#!/bin/bash
# Usage: GPU=<idx> JOB=<jobid> bash scripts/adapt_train_tmux.sh <run_name> <train_rollouts_glob> <val_rollouts_glob> [extra adapt_train.py args...]
NAME=$1; TR=$2; VA=$3; shift 3
ROOT=/iridisfs/scratch/pf2m24/projects/motion_rebot
tmux new-session -d -s train_$NAME "srun --jobid=${JOB:-1476768} --overlap bash"
sleep 5
tmux send-keys -t train_$NAME "hostname; export CUDA_VISIBLE_DEVICES=${GPU:-1}; source $ROOT/scripts/activate_uniphys.sh; cd $ROOT; python scripts/adapt_train.py --rollouts $TR ${VA:+--val_rollouts $VA} --out outputs/$NAME $* 2>&1 | tee logs/train_$NAME.log; exit \${PIPESTATUS[0]}" C-m
sleep 3; tmux ls | grep train_$NAME
