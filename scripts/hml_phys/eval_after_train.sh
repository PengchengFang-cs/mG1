#!/bin/bash
# Wait until a training run has written step_<STEP>.pt AND its process has exited, then evaluate exactly that one
# checkpoint on the test split (full closed-loop protocol, eval_one.sh) and run the two offline probes.
# usage (inside a long-lived `srun --jobid=<id> --overlap bash` step): eval_after_train.sh <out_dir> <tag> [step]
OUT=$1; TAG=$2; STEP=${3:-50000}
D=/iridisfs/scratch/pf2m24/projects/motion_rebot; cd $D
LOG=logs/hml_phys/eval_${TAG}.log
exec > >(tee -a "$LOG") 2>&1
CK=$D/$OUT/step_$STEP.pt
echo "=== $TAG: waiting for $CK and for the training of $OUT to exit ($(date), $(hostname), job ${SLURM_JOB_ID:-?})"
until [ -f "$CK" ] && ! pgrep -f "train_[a-z_]*\.py --out $OUT" >/dev/null; do sleep 60; done
echo "=== $TAG: closed-loop eval start $(date)"
bash scripts/hml_phys/eval_one.sh "$CK" "$TAG"
echo "=== $TAG: probes $(date)"
source scripts/activate_uniphys.sh >/dev/null 2>&1; cd $D
if grep -q intent_policy <(python -c "import torch; print(\"intent_policy\" if torch.load(\"$CK\", map_location=\"cpu\")[\"args\"].get(\"arch\") == \"intent\" else \"\")" 2>/dev/null); then
  python scripts/hml_phys/probe_intent_use.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
else
  python scripts/hml_phys/probe_text_attention.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
  python scripts/hml_phys/probe_history_use.py --ckpt "$CK" 2>&1 | grep -vE "Warning|return q /"
fi
echo "=== $TAG: all done $(date)"
