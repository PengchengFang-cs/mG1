#!/bin/bash
# Pull nvcr.io/nvidia/isaac-lab:2.1.0 into an apptainer sandbox on a compute node (needs the SSH tunnel).
set -o pipefail
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh >/dev/null 2>&1
# apptainer's Go HTTP client wants socks5:// (not socks5h://)
export ALL_PROXY=socks5://127.0.0.1:18080 HTTPS_PROXY=$ALL_PROXY HTTP_PROXY=$ALL_PROXY https_proxy=$ALL_PROXY http_proxy=$ALL_PROXY
module load apptainer/1.4.0
export APPTAINER_CACHEDIR=/scratch/pf2m24/.cache/apptainer APPTAINER_TMPDIR=/scratch/pf2m24/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
TARGET=/iridisfs/scratch/pf2m24/containers/isaaclab_2.1.0.sandbox
echo "[$(date)] start pull -> $TARGET on $(hostname)"
apptainer build --sandbox $TARGET docker://nvcr.io/nvidia/isaac-lab:2.1.0
rc=$?
echo "[$(date)] build exit $rc"
du -sh $TARGET 2>/dev/null
tunnel_down
exit $rc
