# mG1 — text-driven physics-based humanoid control

Working repository for a text-conditioned physics policy on the **sim-character HumanML3D benchmark**
(SMPL humanoid in Isaac Gym, PULSE low-level control), plus the earlier G1 / ADAPT line.

Everything here is code and documentation. Data, checkpoints, logs and third-party clones stay on the cluster
(see `.gitignore`).

## What is in here

| Path | Contents |
|---|---|
| `hml_phys/` | the benchmark and the policy: token representation, dataset, DiT model, rectified flow, closed-loop rollouts, evaluator, physics metrics |
| `hml_phys/t2m/` | HumanML3D text-motion evaluator (Guo et al. 2022), vendored from KV-Control with import fixes only |
| `scripts/hml_phys/` | dataset construction, evaluations, training, checkpoint-evaluation chains |
| `docs/` | protocol (`06`), model adaptation spec (`07`), method/landscape reading notes (`00`–`05`), code-review briefs |
| `adapt/` | earlier G1 line: token diffusion policy (DDPM / rectified flow, cross-attention and AdaLN denoisers) |
| `NOTES.md` | running log of every step, decision and number |

## The benchmark (docs/06)

HumanML3D test split, closed-loop rollouts from a fixed neutral standing pose; simulated joint positions are
converted to the official 263-d features and scored with the Guo et al. 2022 evaluator (R-Precision, FID,
MM-Dist, Diversity), together with physics metrics (Floating, Penetration, Foot-sliding, Skating, Jerk) and
Duration. The pipeline is validated in two ways: the evaluator reproduces the published ground-truth numbers
(R@1 0.515 vs 0.511), and the official UniPhys checkpoint run through it lands on the value MIND reports for
UniPhys (R@1 0.093 vs 0.087).

## Data

Physics trajectories come from the PULSE-tracked AMASS state-action pairs released with UniPhys, sliced into
HumanML3D clips via `index.csv` with per-file frame-rate resolution (some AMASS subsets are 100 fps and were
resampled to 33.3 fps, not 30). Result: 10,902 clips / 22 h with texts and the official split.
See `scripts/hml_phys/01_match_index.py` … `03_build_dataset.py`.

## Model

MotionCraft-style two-stream DiT (root stream then body stream, the body conditioned on the predicted root),
rectified flow with x0 prediction and a velocity-space loss, CLIP ViT-L/14 text, AdaLN conditioning.
The token is a physics frame: root 15-d and body 420-d (joint positions, velocities, 6-D rotations, joint
velocities and the 69-d PD action). History frames are observed (hard-imputed, no loss), future frames are
generated; every K frames the first K actions are executed in simulation and the window slides.
See `docs/07_motioncraft_adaptation.md`.

## Status

Two trained versions; all selection and reporting on the test split under the full protocol.

| | R@1 | R@3 | FID | Floating (mm) | Duration |
|---|---|---|---|---|---|
| Kinematic GT | 0.515 | 0.799 | 0.002 | — | — |
| Physics GT (PULSE-tracked) | 0.458 | 0.760 | 2.56 | 15.5 | — |
| UniPhys (official checkpoint, our run) | 0.093 | 0.225 | 14.71 | 17.5 | 0.711 |
| v1, 25k steps (182M, 32-frame future) | 0.254 | 0.494 | 8.59 | 17.0 | 0.612 |
| v1, 100k steps | 0.221 | 0.428 | 8.98 | 15.4 | 0.708 |
| v2, 50k steps (82M, whole-sequence future) | 0.230 | 0.462 | 9.11 | 15.9 | 0.246 |

Published numbers from MIND and SCRIPT use their own evaluators, so only R-Precision is roughly comparable
across them; FID, MM-Dist and Diversity are not. The full table, with those rows and the evaluator caveats,
is in `docs/06_hml_phys_protocol.md` §2.1b.

Open issues: text alignment is still far from MIND (0.468 with its intent mechanism); v1 overfits after ~25k
steps; v2 fixes the overfitting but falls far more often and costs much more to roll out, because generating
the whole remaining sequence every K frames is mostly wasted computation.

## Third-party code

UniPhys (environment, PULSE, representation), CLoSD (evaluation reference), KiMoDo, MotionCraft (model),
KV-Control (HumanML3D evaluator). Cloned locally, not redistributed here.
