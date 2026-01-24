#!/usr/bin/env bash
set -euo pipefail

# Generates a real RSAR interference split under dataset/RSAR/*/images-interf_jamA/,
# then verifies layout + that corrupt images are not identical to clean.

DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
CORRUPT="${CORRUPT:-interf_jamA}"
ITYPE="${ITYPE:-noise_jamming}"
PARAMS_JSON="${PARAMS_JSON:-{\"jsRatio\":10,\"stripeFreq\":0.01,\"stripeAmplitude\":50}}"
SPLITS="${SPLITS:-test,val,train}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"

PYTHONUNBUFFERED=1 conda run -n dino_sar python -u tools/prepare_rsar_interference.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}" \
  --type "${ITYPE}" \
  --params-json "${PARAMS_JSON}" \
  --splits "${SPLITS}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --force-replace-symlink \
  --diff-samples 128

conda run -n dino_sar python tools/verify_rsar_corrupt_switch.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}"

conda run -n dino_sar python tools/verify_rsar_interference_diff.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "${CORRUPT}" \
  --split test \
  --samples 256 \
  --seed "${SEED}"

echo "[prepare_rsar_interf_jamA] OK"

