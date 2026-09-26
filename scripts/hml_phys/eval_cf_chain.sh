#!/bin/bash
# TEST-split closed-loop evaluation of CodeFlow checkpoints (docs/08 §10), full protocol:
# Euler 32, CFG 3.5, fixed standing start, random caption, ONE rollout and ONE metric pass (CLAUDE.md §4).
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
export CUDA_VISIBLE_DEVICES=${GPU:-0}
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
NAME=${NAME:?set NAME, e.g. action}
TAG=${TAG:-step_200000}
ITEMS=${ITEMS:-$D/data/humanml3d_phys/rollout_items_test_random.json}
ENVS=${ENVS:-256}
CK=$D/outputs/cf_${NAME}/${TAG}.pt
OUT=$D/data/humanml3d_phys/rollouts_cf_${NAME}_${TAG}_test.pkl
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
cd $D/UniPhys && python main_hml_rollout.py phc/env=env_im_vae phc.env.num_envs=$ENVS phc.headless=True \
  phc.env.episode_length=320 phc.env.stateInit=Start \
  diffusion_forcing/algorithm=df_humanoid diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt \
  diffusion_forcing.algorithm.diffusion.use_ema=False diffusion_forcing.task=interact \
  +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=cf_${NAME}_${TAG} \
  +hml.policy=mc +hml.ckpt=$CK +hml.items=$ITEMS +hml.out=$OUT +hml.num_steps=32 +hml.cfg=3.5 2>&1 \
  | grep -a "MC policy\|done:\|Traceback\|Error\|assert" | tr '\r' '\n'
cd $D
python scripts/hml_phys/07_eval_rollouts.py --rollouts $OUT --fallen exclude --replications 1 \
  --out $D/data/humanml3d_phys/eval_cf_${NAME}_${TAG}_test_exclude.json 2>&1 \
  | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|duration_completion\|Traceback"
echo "==== cf_$NAME $TAG DONE"
