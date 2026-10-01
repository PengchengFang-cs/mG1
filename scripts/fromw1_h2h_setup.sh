#!/usr/bin/env bash
# Build the `h2h` conda env for human2humanoid (FRoM-W1's H-ACT tracking side).
#
# Cloned from `uniphys` rather than built from scratch: that env already carries the hard parts --
# python 3.8, Isaac Gym (egg-link), torch, smplx, smpl_sim, mujoco -- and a fresh env's pip cannot use
# the SOCKS tunnel (PySocks vs bundled urllib3: "PoolKey.__new__() got an unexpected keyword argument").
# Cloning also keeps UniPhys's own env untouched, since the editable installs below would otherwise
# land in it.
#
# Run on a compute node:
#   srun --jobid=<id> --overlap --ntasks=1 bash scripts/fromw1_h2h_setup.sh
set -euo pipefail

H2H=/iridisfs/scratch/pf2m24/projects/motion_rebot/external/FRoM-W1/H-ACT/human2humanoid
CONDA=/iridisfs/scratch/pf2m24/miniconda3

module load gcc/11.5.0
source "$CONDA/etc/profile.d/conda.sh"
export TMPDIR=/scratch/pf2m24/tmp; mkdir -p "$TMPDIR"

if [ ! -d "$CONDA/envs/h2h" ]; then
  echo "=== cloning uniphys -> h2h (no network needed) ==="
  conda create --clone uniphys -n h2h -y
else
  echo "=== h2h already exists, skipping clone ==="
fi

conda activate h2h
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export CC=$(which gcc) CXX=$(which g++)

echo "=== baseline ==="
python -c "import sys, torch, numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('numpy', numpy.__version__)"
python -c "import isaacgym; print('isaacgym OK')" 2>&1 | tail -2

# Compute nodes have no outbound network; everything past here needs the tunnel.
source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh

echo "=== editable installs (--no-deps: do not let pip re-resolve the cloned env) ==="
cd "$H2H"
for p in rsl_rl legged_gym phc; do
  echo "--- $p ---"
  pip install -q --no-deps -e "$p"
done

echo "=== missing third-party deps ==="
# Only what the clone lacks. smplx / smpl_sim / mujoco / torch already come from uniphys, and the two
# git+ requirements in requirements.txt are exactly those two packages.
pip install -q --no-deps \
  torchgeometry numpy-stl easydict termcolor ipdb chardet \
  pyvirtualdisplay imageio-ffmpeg gdown pynput lxml joblib \
  scikit-image scikit-learn opencv-python==4.6.0.66 || true

# pip pulls numpy forward and that removes np.bool, which chumpy still uses. Pin it back every time.
pip install -q "numpy==1.23.5"

echo "=== verify ==="
python - <<'PY'
import importlib, traceback
mods = ["isaacgym", "torch", "numpy", "legged_gym", "rsl_rl", "phc",
        "smplx", "smpl_sim", "mujoco", "torchgeometry", "easydict", "joblib", "chumpy"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok      {m}")
    except Exception as e:
        print(f"  FAIL    {m}: {type(e).__name__}: {e}")
import numpy; print("numpy", numpy.__version__)
PY
echo "=== done ==="
