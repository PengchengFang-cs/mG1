#!/bin/bash
# Run a command inside the Isaac Lab 2.1.0 sandbox on a compute node.
# Usage: bash scripts/isaaclab_exec.sh [--writable] [--offline] <command...>
# - HOME is redirected to scratch (Kit caches, downloaded extensions live there); /iridisfs and /scratch are bound.
# - An SSH SOCKS tunnel to the login node is opened and exported into the container (Kit fetches
#   Nucleus assets and registry extensions over it). Pass --offline to skip the tunnel.
module load apptainer/1.4.0 >/dev/null 2>&1
SB=/iridisfs/scratch/pf2m24/containers/isaaclab_2.1.0.sandbox
CHOME=/iridisfs/scratch/pf2m24/isaaclab_home
mkdir -p $CHOME/kit_data $CHOME/kit_logs $CHOME/kit_cache
MODE=""; OFFLINE=0
while [[ "$1" == --* ]]; do
  case "$1" in
    --writable) MODE="--writable"; shift;;
    --offline) OFFLINE=1; shift;;
    *) break;;
  esac
done
if [[ $OFFLINE -eq 0 ]]; then
  source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh >/dev/null 2>&1
  export APPTAINERENV_ALL_PROXY=$ALL_PROXY APPTAINERENV_HTTPS_PROXY=$ALL_PROXY APPTAINERENV_HTTP_PROXY=$ALL_PROXY
  export APPTAINERENV_https_proxy=$ALL_PROXY APPTAINERENV_http_proxy=$ALL_PROXY APPTAINERENV_NO_PROXY="$NO_PROXY"
  export APPTAINERENV_PIP_CACHE_DIR=/scratch/pf2m24/.cache/pip
fi
export APPTAINERENV_ISAACLAB_PATH=/workspace/isaaclab
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PYTHONUNBUFFERED=1
apptainer exec --nv $MODE --home $CHOME:/root \
  --bind /iridisfs/scratch:/iridisfs/scratch --bind /scratch:/scratch \
  --bind $CHOME/kit_data:/isaac-sim/kit/data --bind $CHOME/kit_logs:/isaac-sim/kit/logs --bind $CHOME/kit_cache:/isaac-sim/kit/cache \
  $SB "$@"
rc=$?
# NOTE: the tunnel is shared by concurrent container runs on this node; leave it up (do not tunnel_down here).
exit $rc
