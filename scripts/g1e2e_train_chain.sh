#!/usr/bin/env bash
# Stage 1 (intent VAE) then stage 2 (HIP + IIP + policy), on the two cards this project holds.
# Stage 2 cannot start before stage 1 finishes: it loads the frozen VAE and takes its normalisation
# statistics from that checkpoint, so the two must come from the same run.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-1678103}
cd "$REPO"
run() { srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "source scripts/activate_h2h.sh && cd $REPO && $1"; }

echo "STAGE vae"
run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_train_vae.py \
      --out outputs/g1e2e/vae --steps 60000 --eval-every 2000 --log-every 500 --device cuda:0" \
  > logs/g1e2e_vae.log 2>&1 || { echo "FAIL vae"; tail -20 logs/g1e2e_vae.log; exit 1; }
echo "OK vae: $(grep -E '^done' logs/g1e2e_vae.log)"

echo "STAGE policy"
run "export CUDA_VISIBLE_DEVICES=0,1 && python -u scripts/g1e2e_train_policy.py \
      --vae outputs/g1e2e/vae/best.pt --out outputs/g1e2e/policy \
      --steps 200000 --eval-every 5000 --log-every 500 --device cuda:0" \
  > logs/g1e2e_policy.log 2>&1 || { echo "FAIL policy"; tail -20 logs/g1e2e_policy.log; exit 1; }
echo "OK policy: $(grep -E '^done' logs/g1e2e_policy.log)"
echo "STAGE done"
