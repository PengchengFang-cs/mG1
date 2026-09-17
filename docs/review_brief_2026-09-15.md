# Code review brief — HumanML3D physics benchmark pipeline (steps 1–5)

## What was agreed with the user (the contract)
Goal: reproduce the sim-character text-to-motion physics benchmark (UniPhys / CLoSD / MIND / SCRIPT protocol)
so that a new policy (MotionCraft-based, later) can be compared to published numbers. Steps approved:
1. Download assets: UniPhys HF PULSE-tracked AMASS state-action pickles (only the files HumanML3D needs) and
   HumanML3D index.csv. Produce a manifest with sizes / counts / success ratio.
2. Build the HumanML3D physics dataset: cut AMASS physics sequences into HumanML3D clips using index.csv
   with correct 20 fps ↔ 30 fps (and 33.33 fps for 100-fps sources) frame alignment, keep only successful
   tracking, drop mirrored (M) clips in v1, attach HumanML3D texts and the train/val/test split, report
   coverage (how many of the 14,616 clips, total hours) to compare with MIND/SCRIPT data scale.
3. Phys-GT check: tracked test clips → 263-d HumanML3D features → text_mot_match evaluator (R-Precision,
   FID) as the physics upper bound; compare against the kinematic GT numbers.
4. Evaluation pipeline: closed-loop rollouts in Isaac Gym from a standing pose for every test caption,
   simulated joint positions → 263-d → evaluator (R-Precision top-1/2/3, FID, MM-Dist, Diversity,
   MModality optional, batch 32, repeated runs), plus physics metrics (Floating, Jerk, Skating/Penetration,
   Duration). Validate with the official UniPhys checkpoint (published R@1 0.09–0.14, FID 0.49–0.60).
5. Adaptation spec document for MotionCraft (no code, no training).
Constraints: no training; no GPU jobs beyond these steps; CPU compute only on compute nodes via srun;
all numbers must follow the published protocol, no self-invented shortcuts; nothing may be silently
skipped or reduced to save effort.

## Files to review (all under /iridisfs/scratch/pf2m24/projects/motion_rebot)
- hml_phys/sim2hml.py, hml_phys/phys_metrics.py, hml_phys/evaluator.py, hml_phys/uniphys_rollout.py
- hml_phys/t2m/*.py (vendored from KV-Control; check the import fixes and that nothing else changed
  semantically — originals at /iridisfs/scratch/pf2m24/projects/Umdd/KV-Control/kvctrl/{common,utils,models})
- scripts/hml_phys/01_match_index.py, 02_download_hf.py, 03_build_dataset.py, 04_physgt_eval.py,
  05_eval_gt_kinematic.py, 06a_prep_rollout_items.py, 07_eval_rollouts.py
- UniPhys/main_hml_rollout.py (diff against UniPhys/main.py) and how it calls hml_phys/uniphys_rollout.py
  (compare with UniPhys/uniphys/algorithms/diffusion_forcing/df_humanoid.py evaluate_t2m_babel, lines 690–877)
- docs/06_hml_phys_protocol.md (protocol description must match the code), docs/07_motioncraft_adaptation.md
  (spec must be consistent with the contract and with the MotionCraft code in vendor_motioncraft/)
- Reference implementations for the protocol: vendor_closd/closd/utils/rep_util.py,
  vendor_closd/closd/diffusion_planner/data_loaders/humanml/utils/metrics.py,
  vendor_closd/closd/diffusion_planner/data_loaders/humanml/motion_loaders/comp_v6_model_dataset.py,
  vendor_closd/closd/env/tasks/closd.py (save_hml_episodes); official HumanML3D processing semantics are
  documented in scripts/hml_phys/03_build_dataset.py docstring.
- Outputs / logs for evidence: data/humanml3d_phys/{match_stats.json, build_stats.json,
  eval_gt_kinematic_smoke.json, eval_gt_via_converter_recover.json}, logs/hml_phys/*.log, NOTES.md
  (section "线 C").

## Known results (to check against the code)
- GT-vs-GT evaluator: R@1 0.515 / R@2 0.706 / R@3 0.797 / MM-Dist 2.972 / Div 9.468 (paper 0.511/0.703/0.797/2.974/9.503).
- Converter validation (official joints → converter → evaluator): R@1 0.515, FID 0.003.
- Dataset: 10,903 clips, 22.0 h (train 8,718 / val 530 / test 1,655).
