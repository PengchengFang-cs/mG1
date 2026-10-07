#!/usr/bin/env bash
# Rebuild everything downstream of the 2026-10-02/03 code-review fixes, on the 2000-clip subset.
#
# Why a rebuild and not just a retrain:
#   * the head extend is now parented to the PELVIS, as both of the env's own configs do
#     (extra/extra_base.yaml:14, phc/phc_base.yaml:4), so the retargeted references change;
#   * the recorder now stores default_dof_pos, which the dataset's start-rest windows need.
# The previous artefacts were moved to data/g1_e2e/superseded_oldhead/ rather than deleted.
#
# Resumable at every stage: finished artefacts are skipped and each reference shard resumes from its
# own .part.pkl, so re-running this after a wall-clock kill continues instead of starting over.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-$(squeue -u "$USER" -h -t R -o "%i %N" | awk '$2=="rose04"{print $1; exit}')}
TRAIN_N=${TRAIN_N:-2000}
VAE_STEPS=${VAE_STEPS:-60000}
POLICY_STEPS=${POLICY_STEPS:-100000}
cd "$REPO"
[ -n "$JOB" ] || { echo "[chain] FAIL: no RUNNING job on rose04"; exit 1; }
echo "[chain] job $JOB, train subset $TRAIN_N clips, vae $VAE_STEPS steps, policy $POLICY_STEPS steps"
run() { srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "source scripts/activate_h2h.sh && cd $REPO && $1"; }

# --- 1. references --------------------------------------------------------------------------------
# g1e2e_build_chain.sh is itself resumable: it skips any shard whose final .pkl exists, and each shard
# resumes from its .part.pkl, so calling it unconditionally is both safe and the way to continue.
TRAIN_N="$TRAIN_N" JOB="$JOB" bash scripts/g1e2e_build_chain.sh >> logs/g1e2e_rebuild_refs.log 2>&1 \
  || { echo "[chain] FAIL refs"; tail -5 logs/g1e2e_rebuild_refs.log; exit 1; }
echo "[chain] refs rebuilt: $(grep -o '"[a-z_]*": [0-9.]*' data/g1_e2e/refs_summary.json | tr '\n' ' ')"

# --- 2. record teacher rollouts -------------------------------------------------------------------
for split in train test; do
  if [ -s "data/g1_e2e/rollouts_${split}.new.pkl" ]; then
    echo "[chain] rollouts_$split already recorded"
    continue
  fi
  run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_record_rollouts.py \
        --refs data/g1_e2e/refs_${split}.pkl --text data/g1_e2e/refs_${split}.text.json \
        --out data/g1_e2e/rollouts_${split}.new.pkl \
        --num-envs 512 --device cuda:0" >> "logs/g1e2e_record_${split}.log" 2>&1 \
    || { echo "[chain] FAIL record $split"; tail -20 "logs/g1e2e_record_${split}.log"; exit 1; }
  echo "[chain] recorded $split: $(grep -o '"n_clips": [0-9]*' "data/g1_e2e/rollouts_${split}.new.meta.json")"
done

# --- 3. stage 1: intent VAE -----------------------------------------------------------------------
if [ -s outputs/g1e2e/vae2/best.pt ]; then
  echo "[chain] vae already trained"
else
  run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_train_vae.py \
        --rollouts data/g1_e2e/rollouts_train.new.pkl --rollouts-eval data/g1_e2e/rollouts_test.new.pkl \
        --out outputs/g1e2e/vae2 --steps $VAE_STEPS --eval-every 2000 --log-every 500 --device cuda:0" \
    > logs/g1e2e_vae2.log 2>&1 || { echo "[chain] FAIL vae"; tail -20 logs/g1e2e_vae2.log; exit 1; }
fi
echo "[chain] vae: $(grep -E '^done' logs/g1e2e_vae2.log)"

# --- 4. stage 2: HIP + IIP + policy ---------------------------------------------------------------
if [ -s outputs/g1e2e/policy2/best.pt ]; then
  echo "[chain] policy already trained"
else
  run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_train_policy.py \
        --rollouts data/g1_e2e/rollouts_train.new.pkl --rollouts-eval data/g1_e2e/rollouts_test.new.pkl \
        --vae outputs/g1e2e/vae2/best.pt --out outputs/g1e2e/policy2 \
        --steps $POLICY_STEPS --eval-every 5000 --log-every 500 --device cuda:0" \
    > logs/g1e2e_policy2.log 2>&1 || { echo "[chain] FAIL policy"; tail -20 logs/g1e2e_policy2.log; exit 1; }
fi
echo "[chain] policy: $(grep -E '^done' logs/g1e2e_policy2.log)"

# --- 5. closed loop, ONCE (CLAUDE.md §4) ----------------------------------------------------------
run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_eval_closed_loop.py \
      --policy outputs/g1e2e/policy2/best.pt --refs data/g1_e2e/refs_train.pkl \
      --text-cache data/g1_e2e/text_clipL14 --num-envs 512 --episode motion --hist-init rest \
      --out outputs/g1e2e/eval_fixed.json --device cuda:0" > logs/g1e2e_eval_fixed2.log 2>&1 \
  || { echo "[chain] FAIL eval"; tail -20 logs/g1e2e_eval_fixed2.log; exit 1; }
echo "[chain] CLOSED LOOP: $(tr '\r' '\n' < logs/g1e2e_eval_fixed2.log | grep 'fall rate')"
echo "[chain] all done"
