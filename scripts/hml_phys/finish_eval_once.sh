#!/bin/bash
# Finish an evaluation whose rollout is already running: wait for the rollout process to exit, then compute the
# metrics ONCE (project CLAUDE.md §4) and run the two probes once. usage: finish_eval_once.sh <rollout_pid> <ckpt> <tag>
PID=$1; CK=$2; TAG=$3
D=/iridisfs/scratch/pf2m24/projects/motion_rebot; cd $D
exec > >(tee -a logs/hml_phys/eval_${TAG}.log) 2>&1
echo "=== $TAG: waiting for rollout pid $PID ($(date))"
while kill -0 $PID 2>/dev/null; do sleep 30; done
echo "=== $TAG: rollout finished $(date); metrics computed once"
source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null; cd $D
export CUDA_VISIBLE_DEVICES=0
python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --fallen exclude --replications 1 --out data/humanml3d_phys/eval_${TAG}_test_exclude.json 2>&1 | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|^mm_dist \|^diversity \|duration_completion\|Traceback"
python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --fallen truncate --replications 1 --out data/humanml3d_phys/eval_${TAG}_test_truncate.json 2>&1 | grep -a "^top1 \|^fid \|Traceback"
python scripts/hml_phys/08_paper_metrics.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --out data/humanml3d_phys/paper_metrics_${TAG}.json 2>&1 | grep -a "Float\|Traceback"
python scripts/hml_phys/probe_text_attention.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
python scripts/hml_phys/probe_history_use.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
echo "=== $TAG: all done $(date)"
