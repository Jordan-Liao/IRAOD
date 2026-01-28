#!/usr/bin/env bash
set -euo pipefail

# IRAOD environment setup script (conda + pip).
#
# Usage:
#   bash scripts/setup_env_iraod.sh
#
# Options (env vars):
#   ENV_NAME=iraod            # conda env name
#   PYTHON_VERSION=3.10       # conda python version
#   CUDA_VARIANT=cu118        # cu118 | cpu
#   TORCH_VERSION=2.0.1
#   TORCHVISION_VERSION=0.15.2
#   MMCV_VERSION=1.7.2
#   FORCE_RECREATE=0          # 1 to delete and recreate env
#
# Notes:
# - MMCV wheels are sensitive to torch/cuda versions.
# - This script targets the versions used by the repo's experiment scripts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-iraod}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VARIANT="${CUDA_VARIANT:-cu118}" # cu118 | cpu

TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
MMCV_VERSION="${MMCV_VERSION:-1.7.2}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
OPENCV_HEADLESS_VERSION="${OPENCV_HEADLESS_VERSION:-4.11.0.86}"

FORCE_RECREATE="${FORCE_RECREATE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup_env_iraod] ERROR: conda not found in PATH." >&2
  exit 1
fi

# Make `conda activate` work in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ "${FORCE_RECREATE}" == "1" ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup_env_iraod] FORCE_RECREATE=1: removing env ${ENV_NAME} ..."
    conda env remove -n "${ENV_NAME}" -y
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup_env_iraod] creating env ${ENV_NAME} (python=${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "[setup_env_iraod] env exists: ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

# Keep numpy on 1.x: opencv/mmcv wheels are not yet compatible with numpy>=2.
pip install "numpy==${NUMPY_VERSION}"

if [[ "${CUDA_VARIANT}" == "cpu" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  MMCV_INDEX_URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_VARIANT}"
  MMCV_INDEX_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_VARIANT}/torch2.0/index.html"
fi

echo "[setup_env_iraod] installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} (${CUDA_VARIANT}) ..."
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX_URL}"

echo "[setup_env_iraod] installing openmim + mmcv-full==${MMCV_VERSION} ..."
pip install -U openmim
mim install "mmcv-full==${MMCV_VERSION}" -f "${MMCV_INDEX_URL}"

echo "[setup_env_iraod] installing repo requirements ..."
pip install -r "${REPO_ROOT}/requirements.txt"

# Some deps (e.g. albumentations) may pull an opencv-python-headless that
# requires numpy>=2; pin back to a numpy1-compatible wheel.
pip install "numpy==${NUMPY_VERSION}" "opencv-python-headless==${OPENCV_HEADLESS_VERSION}"
pip uninstall -y opencv-python || true

echo "[setup_env_iraod] verifying core versions ..."
python -c "import sys, torch, mmcv, mmdet, mmrotate; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmrotate', mmrotate.__version__)"

echo "[setup_env_iraod] DONE. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  conda run -n ${ENV_NAME} python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR"
