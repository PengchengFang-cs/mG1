#!/bin/bash
# TEST-split evaluation of every mc_v2 checkpoint, full protocol (Euler 32, cfg 3.5, neutral start, random caption)
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null
export CUDA_VISIBLE_DEVICES=0
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
run_roll () {
  cd $D/UniPhys && python main_hml_rollout.py phc/env=env_im_vae phc.env.num_envs=256 phc.headless=True phc.env.episode_length=320 phc.env.stateInit=Start \
    diffusion_forcing/algorithm=df_humanoid diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt diffusion_forcing.algorithm.diffusion.use_ema=False \
    diffusion_forcing.task=interact +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=mc_v2_eval_$1 \
    +hml.policy=mc +hml.ckpt=$2 +hml.items=$3 +hml.out=$4 +hml.num_steps=$5 +hml.cfg=$6 2>&1 | grep -a "MC policy\|done:\|Traceback\|Error" | tr '\r' '\n'
  cd $D
}
for tag in step_50000 best_val step_10000 step_20000 step_30000 step_40000; do
  [ -f $D/outputs/mc_v2/$tag.pt ] || { echo "missing $tag"; continue; }
  run_roll test_$tag $D/outputs/mc_v2/$tag.pt $D/data/humanml3d_phys/rollout_items_test_random.json $D/data/humanml3d_phys/rollouts_mc_v2_${tag}_test.pkl 32 3.5
  python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_mc_v2_${tag}_test.pkl --fallen exclude --replications 20 --out data/humanml3d_phys/eval_mc_v2_${tag}_test_exclude.json 2>&1 | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|^mm_dist \|^diversity \|duration_completion\|physics_gen_raw\|Traceback"
  python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_mc_v2_${tag}_test.pkl --fallen truncate --replications 5 --out data/humanml3d_phys/eval_mc_v2_${tag}_test_truncate.json 2>&1 | grep -a "^top1 \|^fid \|Traceback"
  echo "==== $tag DONE"
done
echo "==== V2 CHAIN DONE"
