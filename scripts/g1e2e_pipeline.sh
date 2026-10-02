#!/usr/bin/env bash
# Rebuild everything downstream of the review fixes, then smoke-test both training stages.
#
# Stops before the long training run on purpose: the critical fix (the IIP's target) is only verifiable
# by looking at whether l_iip stays non-zero, so a human reads the smoke numbers before 200k steps start.
# Every stage prints STAGE/OK/FAIL markers so a monitor can follow it without tailing the whole log.
set -uo pipefail
REPO=/iridisfs/scratch/pf2m24/projects/motion_rebot
JOB=${JOB:-1678103}
cd "$REPO"

run() { srun --jobid="$JOB" --overlap --ntasks=1 bash -lc "source scripts/activate_h2h.sh && cd $REPO && $1"; }

echo "STAGE waiting for the reference rebuild"
# Wait for the CHAIN, not for the shard processes: the chain merges and runs the world-frame check after
# the last shard exits, so watching the shards alone sees a gap and advances before refs_*.pkl exist.
while pgrep -f "g1e2e_build_chain.sh" >/dev/null || pgrep -f "g1e2e_build_references.py --split" >/dev/null       || pgrep -f "g1e2e_merge_refs.py" >/dev/null; do sleep 30; done
for sp in train test; do
  [ -s "data/g1_e2e/refs_${sp}.pkl" ] || { echo "FAIL references: refs_${sp}.pkl missing"; exit 1; }
done
python3 - <<'PY' || { echo "FAIL references: fps still constant"; exit 1; }
import json
t = json.load(open("/iridisfs/scratch/pf2m24/projects/motion_rebot/data/g1_e2e/refs_train.text.json"))
fps = sorted({round(v["fps"], 3) for v in t.values()})
strides = sorted({v["hml_stride"] for v in t.values()})
print(f"OK references: {len(t)} train clips, fps values {fps}, HumanML3D strides {strides}")
assert len(fps) > 1, "every clip still has the same fps -- the per-clip rate fix did not take effect"
PY

echo "STAGE recording rollouts"
for sp in train test; do
  gpu=$([ "$sp" = train ] && echo 0 || echo 1)
  run "export CUDA_VISIBLE_DEVICES=$gpu && python -u scripts/g1e2e_record_rollouts.py \
        --refs data/g1_e2e/refs_${sp}.pkl --text data/g1_e2e/refs_${sp}.text.json \
        --num-envs 512 --device cuda:0 --out data/g1_e2e/rollouts_${sp}.pkl" \
    > "logs/g1e2e_rec_${sp}.log" 2>&1 &
done
wait
for sp in train test; do
  grep -q "^kept " "logs/g1e2e_rec_${sp}.log" || { echo "FAIL rollouts $sp"; tail -20 "logs/g1e2e_rec_${sp}.log"; exit 1; }
  echo "OK rollouts $sp: $(grep '^kept ' logs/g1e2e_rec_${sp}.log)"
done

echo "STAGE building the text cache"
run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_build_text_cache.py \
      --text-json data/g1_e2e/refs_train.text.json data/g1_e2e/refs_test.text.json \
      --out data/g1_e2e/text_clipL14 --device cuda:0" > logs/g1e2e_text.log 2>&1 \
  || { echo "FAIL text cache"; tail -15 logs/g1e2e_text.log; exit 1; }
echo "OK text cache: $(grep -E "^clips |unconditional" logs/g1e2e_text.log | tr '\n' ' ')"

echo "STAGE VAE smoke"
rm -rf /scratch/pf2m24/tmp/vae_s3
run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_train_vae.py \
      --out /scratch/pf2m24/tmp/vae_s3 --steps 600 --eval-every 300 --log-every 200 \
      --max-clips 300 --device cuda:0" > logs/g1e2e_vae_smoke.log 2>&1 \
  || { echo "FAIL VAE smoke"; tail -15 logs/g1e2e_vae_smoke.log; exit 1; }
echo "OK VAE smoke: $(grep -E '\[eval\]|^done' logs/g1e2e_vae_smoke.log | tail -2 | tr '\n' ' ')"

echo "STAGE policy smoke"
rm -rf /scratch/pf2m24/tmp/pol_s3
run "export CUDA_VISIBLE_DEVICES=0 && python -u scripts/g1e2e_train_policy.py \
      --vae /scratch/pf2m24/tmp/vae_s3/best.pt --out /scratch/pf2m24/tmp/pol_s3 \
      --steps 300 --eval-every 150 --log-every 50 --max-clips 200 --batch 16 \
      --lat-stat-rows 20000 --device cuda:0" > logs/g1e2e_pol_smoke.log 2>&1 \
  || { echo "FAIL policy smoke"; tail -20 logs/g1e2e_pol_smoke.log; exit 1; }
echo "OK policy smoke"
grep -E "intent latents:|unconditional CLIP|^step|\[eval\]" logs/g1e2e_pol_smoke.log | tail -8

echo "STAGE done -- everything rebuilt and both stages smoke-tested; training not started"
