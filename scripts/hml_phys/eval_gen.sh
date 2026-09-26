#!/bin/bash
# ONE closed-loop evaluation of ONE checkpoint under ONE sampling configuration (project CLAUDE.md §4:
# a single rollout and a single metric pass; never re-run a configuration that already has a number).
# usage: CK=<ckpt> TAG=<tag> [STEPS=32] [CFG=3.5] [K=4] [GPU=0] bash eval_gen.sh
set -o pipefail
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
cd $D; source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null
cd $D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CK=${CK:?set CK}; TAG=${TAG:?set TAG}; STEPS=${STEPS:-32}; CFG=${CFG:-3.5}; K=${K:-4}
case "$CK" in /*) ;; *) CK="$D/$CK";; esac      # main_hml_rollout.py runs from UniPhys/, so the path must be absolute
OUT=$D/data/humanml3d_phys/rollouts_${TAG}_test.pkl
if [ -f "$OUT" ]; then echo "SKIP $TAG: rollout already exists (CLAUDE.md §4, never repeat)"; exit 0; fi
echo "=== $TAG start $(date) on $(hostname) gpu $CUDA_VISIBLE_DEVICES: ckpt=$CK steps=$STEPS cfg=$CFG K=$K"
cd $D/UniPhys && python main_hml_rollout.py phc/env=env_im_vae phc.env.num_envs=256 phc.headless=True \
  phc.env.episode_length=320 phc.env.stateInit=Start \
  diffusion_forcing/algorithm=df_humanoid diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt \
  diffusion_forcing.algorithm.diffusion.use_ema=False diffusion_forcing.task=interact \
  +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=eval_$TAG \
  +hml.policy=mc +hml.ckpt=$CK +hml.items=$D/data/humanml3d_phys/rollout_items_test_random.json \
  +hml.out=$OUT +hml.num_steps=$STEPS +hml.cfg=$CFG +hml.K=$K \
  ${H_SPARSE:+ +hml.h_sparse=$H_SPARSE} ${ALPHA:+ +hml.alpha=$ALPHA} ${L_MAX:+ +hml.l_max=$L_MAX} ${S_READ:+ +hml.s_read=$S_READ} 2>&1 \
  | grep -a "MC policy\|done:\|Traceback\|Error" | tr '\r' '\n'
cd $D
python scripts/hml_phys/07_eval_rollouts.py --rollouts $OUT --fallen exclude --replications 1 \
  --out $D/data/humanml3d_phys/eval_${TAG}_test_exclude.json 2>&1 \
  | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|duration_completion\|Traceback"
echo "=== $TAG done $(date)"
