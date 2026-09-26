#!/bin/bash
# Evaluate ONE checkpoint on the test split under the full protocol.
# usage: eval_one.sh <ckpt_path> <tag> [num_steps] [cfg]
set -o pipefail
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
cd $D; source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null
export CUDA_VISIBLE_DEVICES=0
CKPT=$1; TAG=$2; STEPS=${3:-32}; CFG=${4:-3.5}
cd $D/UniPhys && python main_hml_rollout.py phc/env=env_im_vae phc.env.num_envs=256 phc.headless=True phc.env.episode_length=320 phc.env.stateInit=Start \
  diffusion_forcing/algorithm=df_humanoid diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt diffusion_forcing.algorithm.diffusion.use_ema=False \
  diffusion_forcing.task=interact +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=eval_$TAG \
  +hml.policy=mc +hml.ckpt=$CKPT +hml.items=$D/data/humanml3d_phys/rollout_items_test_random.json \
  +hml.out=$D/data/humanml3d_phys/rollouts_${TAG}_test.pkl +hml.num_steps=$STEPS +hml.cfg=$CFG 2>&1 | grep -a "MC policy\|done:\|Traceback\|Error" | tr '\r' '\n'
cd $D
python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --fallen exclude --replications 1 --out data/humanml3d_phys/eval_${TAG}_test_exclude.json 2>&1 | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|^mm_dist \|^diversity \|duration_completion\|Traceback"
python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --fallen truncate --replications 1 --out data/humanml3d_phys/eval_${TAG}_test_truncate.json 2>&1 | grep -a "^top1 \|^fid \|Traceback"
python scripts/hml_phys/08_paper_metrics.py --rollouts data/humanml3d_phys/rollouts_${TAG}_test.pkl --out data/humanml3d_phys/paper_metrics_${TAG}.json 2>&1 | grep -a "Float\|Traceback"
echo "==== $TAG DONE"
