#!/bin/bash
# Install textop_tracker + modified rsl_rl into the Isaac Lab container (writable). Needs tunnel.
set -o pipefail
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh >/dev/null 2>&1
export APPTAINERENV_ALL_PROXY=$ALL_PROXY APPTAINERENV_HTTPS_PROXY=$ALL_PROXY APPTAINERENV_HTTP_PROXY=$ALL_PROXY APPTAINERENV_https_proxy=$ALL_PROXY APPTAINERENV_http_proxy=$ALL_PROXY
export APPTAINERENV_PIP_CACHE_DIR=/scratch/pf2m24/.cache/pip
ROOT=/iridisfs/scratch/pf2m24/projects/motion_rebot
bash $ROOT/scripts/isaaclab_exec.sh --writable bash -lc '
set -e
PY=/workspace/isaaclab/_isaac_sim/python.sh
$PY -m pip install toml psutil onnxscript "wandb>=0.19" 2>&1 | tail -1
cd '$ROOT'/TextOp/TextOpTracker && $PY -m pip install -e source/textop_tracker 2>&1 | tail -1
$PY -m pip uninstall rsl_rl_lib -y 2>&1 | tail -1
cd '$ROOT'/TextOp && $PY -m pip install -e deps/rsl_rl-modular-normed/ 2>&1 | tail -1
$PY -c "import rsl_rl, textop_tracker; print(\"rsl_rl:\", rsl_rl.__file__); print(\"textop_tracker ok\")"
'
rc=$?
tunnel_down
echo "install exit $rc"
exit $rc
