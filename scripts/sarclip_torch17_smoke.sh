#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-sarclip_torch171}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"

# Prefer CPU wheels to avoid CUDA/cuDNN compatibility issues in the smoke.
TORCH_SPEC="${TORCH_SPEC:-torch==1.7.1+cpu}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.8.2+cpu}"
TORCH_WHL_INDEX="${TORCH_WHL_INDEX:-https://download.pytorch.org/whl/torch_stable.html}"

MODEL="${MODEL:-RN50}"
IMAGE="${IMAGE:-dataset/RSAR/train/images/0000002.png}"
PROMPTS="${PROMPTS:-an SAR image of ship}"
OUT="${OUT:-work_dirs/sanity/sarclip_smoke_torch171.log}"

echo "[sarclip_torch17_smoke] env=${ENV_NAME}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[sarclip_torch17_smoke] create conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

  echo "[sarclip_torch17_smoke] install torch/torchvision (CPU wheels)"
  conda run -n "${ENV_NAME}" python -m pip install "${TORCH_SPEC}" "${TORCHVISION_SPEC}" -f "${TORCH_WHL_INDEX}"
fi

echo "[sarclip_torch17_smoke] install SARCLIP runtime deps (without gdal/transformers)"
# NOTE: newer timm requires torch.fx (torch>=1.8). Pin a compatible timm for torch==1.7.1.
conda run -n "${ENV_NAME}" python -m pip install --upgrade \
  ftfy "timm==0.4.12" regex pandas matplotlib safetensors tqdm huggingface-hub fvcore fsspec

mkdir -p "$(dirname "${OUT}")"

tmp_out="${OUT}.tmp"
rm -f "${tmp_out}"

echo "[sarclip_torch17_smoke] run tools/sarclip_smoke.py (cpu) ..."
conda run -n "${ENV_NAME}" python tools/sarclip_smoke.py \
  --image "${IMAGE}" \
  --prompts "${PROMPTS}" \
  --model "${MODEL}" \
  --device cpu \
  --out "${tmp_out}"

{
  conda run -n "${ENV_NAME}" python -c "import inspect, torch; print('torch=' + torch.__version__); print('has_batch_first=' + str('batch_first' in inspect.signature(torch.nn.MultiheadAttention.__init__).parameters))"
  cat "${tmp_out}"
  echo ""
  echo "OK"
} > "${OUT}"
rm -f "${tmp_out}"

echo "[sarclip_torch17_smoke] wrote: ${OUT}"
