#!/bin/bash
# Fetch everything FRoM-W1 needs that is not already on disk.
# Compute nodes are offline, so this runs on the login node (I/O only, per the cluster rule).
set -uo pipefail
R=/iridisfs/scratch/pf2m24/projects/motion_rebot
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh >/dev/null 2>&1
export HF_HOME=/iridisfs/scratch/pf2m24/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
W=$R/external/fromw1_weights
mkdir -p "$W"
CLI=/scratch/pf2m24/miniconda3/bin/huggingface-cli

echo "[dl] FRoM-W1 weights: H-GPT (Motion-X VQ-VAE + CoT LoRA) and the G1 tracking policy"
$CLI download OpenMOSS-Team/FRoM-W1 --local-dir "$W" \
  --include "hgpt/motionx/*" "hact/g1/*" "eval/*" "hgpt/README.md" "hact/README.md" \
  --exclude "*events.out.tfevents*" || echo "[dl] FRoM-W1 FAILED"

echo "[dl] Llama-3.1-8B base (gated; the stored token has access)"
$CLI download meta-llama/Llama-3.1-8B --local-dir "$W/Meta-Llama-3.1-8B" \
  --include "*.json" "*.safetensors" "*.model" "tokenizer*" \
  --exclude "original/*" || echo "[dl] Llama FAILED"

echo "[dl] done  $(du -sh "$W" | cut -f1)"
