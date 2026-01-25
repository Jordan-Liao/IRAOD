#!/usr/bin/env bash
set -euo pipefail

# Generates a real RSAR interference split under dataset/RSAR/*/images-interf_jamB/,
# then verifies layout + that corrupt images are not identical to clean.

DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
ENV_NAME="${ENV_NAME:-iraod}"
CORRUPT="${CORRUPT:-interf_jamB}"
ITYPE="${ITYPE:-noise_am_jamming}"
PARAMS_JSON="${PARAMS_JSON:-{\"lineFrequency\":0.05,\"baseIntensity\":150,\"noiseSigma\":200.0,\"lineWidth\":10,\"direction\":\"vertical\",\"blendFactor\":0.3}}"
SPLITS="${SPLITS:-test,val,train}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"

PYTHONUNBUFFERED=1 conda run -n "${ENV_NAME}" python -u tools/prepare_rsar_interference.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}" \
  --type "${ITYPE}" \
  --params-json "${PARAMS_JSON}" \
  --splits "${SPLITS}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --force-replace-symlink \
  --diff-samples 128

conda run -n "${ENV_NAME}" python tools/verify_rsar_corrupt_switch.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}"

conda run -n "${ENV_NAME}" python tools/verify_rsar_interference_diff.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}" \
  --split test \
  --samples 256 \
  --seed "${SEED}"

echo "[prepare_rsar_interf_jamB] OK"
