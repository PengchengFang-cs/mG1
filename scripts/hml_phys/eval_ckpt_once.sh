#!/bin/bash
# One evaluation of one checkpoint: single rollout + single metric pass (project CLAUDE.md §4) + the matching probe.
# usage (inside a long-lived compute-node step): eval_ckpt_once.sh <ckpt> <tag>
CK=$1; TAG=$2
D=/iridisfs/scratch/pf2m24/projects/motion_rebot; cd $D
exec > >(tee -a logs/hml_phys/eval_${TAG}.log) 2>&1
echo "=== $TAG: eval start $(date) on $(hostname), job ${SLURM_JOB_ID:-?}"
bash scripts/hml_phys/eval_one.sh "$CK" "$TAG"
source scripts/activate_uniphys.sh >/dev/null 2>&1; cd $D
export CUDA_VISIBLE_DEVICES=0
if python -c "import torch,sys; sys.exit(0 if torch.load('$CK', map_location='cpu')['args'].get('arch')=='intent' else 1)" 2>/dev/null; then
  python scripts/hml_phys/probe_intent_use.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
else
  python scripts/hml_phys/probe_text_attention.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
  python scripts/hml_phys/probe_history_use.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
fi
echo "=== $TAG: all done $(date)"
