#!/usr/bin/env bash
set -euo pipefail

# Generate a representative severity level for jamB (interf_jamB_s3) on train/val.
# This is intended for robustness training without exploding disk usage.
#
# Outputs:
#   dataset/RSAR/train/images-interf_jamB_s3/
#   dataset/RSAR/val/images-interf_jamB_s3/

DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
SPLITS="${SPLITS:-train,val}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --splits "${SPLITS}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --force-replace-symlink
  --diff-samples 256
)

if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

# s3 params match `scripts/prepare_rsar_interf_severity_test.sh`:
# lineFrequency=0.05, baseIntensity=150, direction=vertical
# noiseSigma=120, lineWidth=10, blendFactor=0.35
PARAMS_JSON='{"lineFrequency":0.05,"baseIntensity":150,"noiseSigma":120,"lineWidth":10,"direction":"vertical","blendFactor":0.35}'

echo "[prepare_rsar_interf_jamB_s3_trainval] data_root=${DATA_ROOT} splits=${SPLITS} workers=${WORKERS} seed=${SEED}"

PYTHONUNBUFFERED=1 conda run -n dino_sar python -u tools/prepare_rsar_interference.py \
  --corrupt "interf_jamB_s3" \
  --type "noise_am_jamming" \
  --params-json "${PARAMS_JSON}" \
  "${COMMON_ARGS[@]}"

conda run -n dino_sar python tools/verify_rsar_corrupt_switch.py \
  --data-root "${DATA_ROOT}" \
  --corrupt "interf_jamB_s3" \
  --splits "${SPLITS}"

echo "[prepare_rsar_interf_jamB_s3_trainval] OK"

