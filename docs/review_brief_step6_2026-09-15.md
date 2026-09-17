# Code review brief — step 6: MotionCraft-style physics policy (implementation + training)

## Contract (what the user agreed; docs/07_motioncraft_adaptation.md is the authoritative spec, esp. §1, §1.5, §2–§5, §10–§12)
- Token per frame (30 fps): root 15 = root_trans 3 | root_rot_6d 6 | root_trans_vel 3 | root_rot_vel 3;
  body 420 = local_positions 72 | local_vel 72 | dof_pose_6d 138 | dof_vel 69 | action 69. Computed exactly like UniPhys
  `cano_seq_smpl_or_smplx` + `get_repr(return_last=True)` (UniPhys/uniphys/utils/motion_repr_utils.py), window-canonical
  frame (frame 0 = oldest history frame). action[t] = action executed from frame t to t+1 (HF field `action`).
- Window T = 16 history + 32 future, sliding windows stride 1 over training clips; history frames observed (mask 1,
  hard-imputed, no loss), future generated.
- Model: MotionCraft blocks (vendor_motioncraft/models/codeflow/dit_blocks.py FrameMotionTextDiT) — root stream then
  body stream; body conditioned on the PREDICTED root (detached in training); hidden 768, 8 heads, root 2 double + 4
  single, body 3 double + 6 single; AdaLN cond = timestep + pooled CLIP + scalar conditions (progress, total_len);
  text = CLIP ViT-L/14 tokens (≤50) + pooled, joint attention as in MotionCraft.
- Flow: rectified flow, MotionCraft convention (t=1 clean), logit-normal t (p_mean -0.8, p_std 0.8), network predicts x0,
  loss in velocity space with (1-t) clamped at 0.05, loss only on future frames; second loss = root position/velocity
  consistency (weight 0.01); NO other losses. Text dropout 0.1 (empty caption CLIP features), CFG 3.5 at sampling,
  Euler 32 steps, x0-space CFG, observed frames re-imposed every step.
- Progress condition: (frames executed / clip frames, clip duration seconds/10). BABEL text: NOT in v1.
- Augmentations: start-rest p=0.1 (history = 16 copies of the clip's first frame, zero velocities, hold action =
  (dof_pos - pd_offset)/pd_scale; future = clip frames [0:32)); neutral-pose p=0.05 only for clips that start near
  rest (history = evaluation neutral standing state); history noise OFF by default (switch kept).
- Training: AdamW lr 1e-4 wd 0.01, grad clip 1.0, bf16 autocast, EMA 0.995 every 10 steps, batch 256, 300k steps,
  checkpoint every 50k + best val loss (val loss with EMA weights on a fixed val subset); no scheduler (MotionCraft).
- Closed loop: every K=4 executed frames re-plan from the last 16 raw physics frames; execute the action channels of
  the first 4 future frames as raw 69-d actions (env applies pd_target = offset + scale*a); start from the fixed
  standing pose (phc.env.stateInit=Start), history filled with the start frame + hold action, 2 warm-up hold steps;
  fall bookkeeping via env `_terminate_buf` (timeouts are not falls); same recording format as the UniPhys rollouts.
- Constraints: CPU work only via srun on compute nodes; no silent simplifications; every deviation must be documented.

## Files to review (under /iridisfs/scratch/pf2m24/projects/motion_rebot)
- hml_phys/tokens.py (reference: UniPhys/uniphys/utils/motion_repr_utils.py, UniPhys/uniphys/utils/quaternion.py;
  scripts/hml_phys/check_tokens.py reports max diff 6e-5 vs UniPhys on 30 random windows; note the deliberate
  reproduction of UniPhys's l_hip/r_hip swap in get_repr)
- hml_phys/dataset.py, scripts/hml_phys/compute_token_stats.py, scripts/hml_phys/build_text_cache.py, hml_phys/text_clip.py
  (reference: vendor_motioncraft/models/raw_motion/raw_flow_dit.py FrozenCLIPTextEncoder; MotionCraft text dropout =
  empty string through CLIP, train_hy273_raw_flow.py:1289-1297)
- hml_phys/mc_model.py (reference: vendor_motioncraft/models/raw_motion/unified_kimodo_flow_dit.py, dit_blocks.py)
- hml_phys/flow.py (reference: vendor_motioncraft/models/raw_motion/flow_schedule.py, train_hy273_raw_flow.py
  representation_loss_pair 775-802, sample_hy273_raw.py sample_ode Euler step 491-512 and CFG)
- scripts/hml_phys/train_mc.py
- hml_phys/mc_rollout.py and the dispatch in UniPhys/main_hml_rollout.py (`+hml.policy=mc`); reference for env
  stepping: hml_phys/uniphys_rollout.py, UniPhys/phc/env/tasks/humanoid.py (pre_physics_step / _action_to_pd_targets),
  UniPhys/phc/learning/common_player.py env_step
- Evidence: logs/hml_phys/token_stats.log, text_cache.log, env_constants.log, outputs/mc_smoke/train_log.txt (60-step
  smoke), data/humanml3d_phys/rollouts_mc_smoke.pkl (8-episode closed-loop smoke, all fell as expected for an untrained
  model), logs/hml_phys/train_mc_v1.log (full run, started 2026-09-15 evening), NOTES.md section 线 C.
