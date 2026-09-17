#!/bin/bash
# Checkpoint screening on the VAL split (Euler 10) then the two agreed TEST evaluations (Euler 32, cfg 3.5).
set -o pipefail
cd /iridisfs/scratch/pf2m24/projects/motion_rebot
source scripts/activate_uniphys.sh >/dev/null 2>&1; module load cuda/12.4.0 gcc/11.5.0 2>/dev/null
export CUDA_VISIBLE_DEVICES=0
D=/iridisfs/scratch/pf2m24/projects/motion_rebot
run_roll () {  # tag ckpt items out steps cfg
  cd $D/UniPhys && python main_hml_rollout.py phc/env=env_im_vae phc.env.num_envs=256 phc.headless=True phc.env.episode_length=320 phc.env.stateInit=Start \
    diffusion_forcing/algorithm=df_humanoid diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt diffusion_forcing.algorithm.diffusion.use_ema=False \
    diffusion_forcing.task=interact +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=mc_eval_$1 \
    +hml.policy=mc +hml.ckpt=$2 +hml.items=$3 +hml.out=$4 +hml.num_steps=$5 +hml.cfg=$6 2>&1 | grep -a "MC policy\|done:\|Traceback\|Error" | tr '\r' '\n'
  cd $D
}
# ---- val screening
for tag in best_val step_50000 step_100000 step_250000; do
  run_roll val_$tag $D/outputs/mc_v1/$tag.pt $D/data/humanml3d_phys/rollout_items_val_random.json $D/data/humanml3d_phys/rollouts_mc_v1_${tag}_val_e10.pkl 10 3.5
  python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_mc_v1_${tag}_val_e10.pkl --split val --fallen exclude --replications 5 --out data/humanml3d_phys/eval_mc_v1_${tag}_val_e10.json 2>&1 | grep -a "episodes \|^top1 \|^fid \|^mm_dist \|duration\|physics_gen_raw\|Traceback"
done
echo "==== VAL SCREENING DONE"
# ---- test: final (250k) and best_val, full protocol
for tag in step_250000 best_val; do
  run_roll test_$tag $D/outputs/mc_v1/$tag.pt $D/data/humanml3d_phys/rollout_items_test_random.json $D/data/humanml3d_phys/rollouts_mc_v1_${tag}_test.pkl 32 3.5
  python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_mc_v1_${tag}_test.pkl --fallen exclude --replications 20 --out data/humanml3d_phys/eval_mc_v1_${tag}_test_exclude.json 2>&1 | grep -a "episodes \|^top1 \|^top2 \|^top3 \|^fid \|^mm_dist \|^diversity \|duration\|physics_gen_raw\|Traceback"
  python scripts/hml_phys/07_eval_rollouts.py --rollouts data/humanml3d_phys/rollouts_mc_v1_${tag}_test.pkl --fallen truncate --replications 5 --out data/humanml3d_phys/eval_mc_v1_${tag}_test_truncate.json 2>&1 | grep -a "^top1 \|^fid \|Traceback"
done
echo "==== ALL DONE"
