#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <source_ckpt|none> [train_config] [work_root]" >&2
  echo "Examples:" >&2
  echo "  RSAR_PROTOCOL=phase3 $0 none work_dirs/rsar_phase3_7corr" >&2
  echo "  RSAR_PROTOCOL=phase3 $0 work_dirs/my_teacher/latest.pth" >&2
  echo "  RSAR_PROTOCOL=sfodrs  $0 work_dirs/rsar_sfodrs_source/latest.pth" >&2
  exit 2
fi

SOURCE_CKPT="$1"
TRAIN_CONFIG="${2:-}"
WORK_ROOT="${3:-work_dirs/rsar_sfodrs}"
PYTHON_BIN="${PYTHON:-python3}"
export CGA_MODE="${CGA_MODE:-sfodrs}"

RSAR_PROTOCOL="${RSAR_PROTOCOL:-sfodrs}"
if [[ -z "${TRAIN_CONFIG}" ]]; then
  if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
    TRAIN_CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py"
  else
    TRAIN_CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py"
  fi
fi

# Eval config must match EMA checkpoints for Phase 3 training (teacher is a plain detector).
RSAR_EVAL_CONFIG="${RSAR_EVAL_CONFIG:-configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py}"
export RSAR_EVAL_CONFIG

CORRS=(
  chaff
  gaussian_white_noise
  point_target
  noise_suppression
  am_noise_horizontal
  smart_suppression
  am_noise_vertical
)

echo "[run_rsar_sfodrs_7corr] source_ckpt=${SOURCE_CKPT}"
echo "[run_rsar_sfodrs_7corr] protocol=${RSAR_PROTOCOL}"
echo "[run_rsar_sfodrs_7corr] train_config=${TRAIN_CONFIG}"
echo "[run_rsar_sfodrs_7corr] eval_config=${RSAR_EVAL_CONFIG}"
echo "[run_rsar_sfodrs_7corr] work_root=${WORK_ROOT}"
echo "[run_rsar_sfodrs_7corr] cga_mode=${CGA_MODE}"

echo "[run_rsar_sfodrs_7corr] step=source_clean_test"
if [[ "${SOURCE_CKPT}" == "none" || "${SOURCE_CKPT}" == "null" || "${SOURCE_CKPT}" == "-" ]]; then
  echo "[run_rsar_sfodrs_7corr] source_clean_test skipped (SOURCE_CKPT=${SOURCE_CKPT})"
else
  if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
    RSAR_STAGE=source_clean_test \
    RSAR_TARGET_DOMAIN=clean \
    RSAR_USE_CGA=0 \
    "${PYTHON_BIN}" -u test.py "${TRAIN_CONFIG}" "${SOURCE_CKPT}" \
      --work-dir "${WORK_ROOT}/clean/source_clean_test" \
      --eval mAP
  else
    "${PYTHON_BIN}" -u test.py "${RSAR_EVAL_CONFIG}" "${SOURCE_CKPT}" \
      --work-dir "${WORK_ROOT}/clean/source_clean_test" \
      --cga-scorer none \
      --eval mAP
  fi
fi

for CORR in "${CORRS[@]}"; do
  bash scripts/exp_rsar_sfodrs_adapt.sh "${CORR}" "${SOURCE_CKPT}" "${TRAIN_CONFIG}" "${WORK_ROOT}"
done

echo "[run_rsar_sfodrs_7corr] step=collect_results"
"${PYTHON_BIN}" -u tools/collect_rsar_sfodrs_results.py \
  --work-root "${WORK_ROOT}" \
  --out-csv "${WORK_ROOT}/rsar_sfodrs_results.csv" \
  --out-md "${WORK_ROOT}/rsar_sfodrs_results.md"

echo "[run_rsar_sfodrs_7corr] done: ${WORK_ROOT}"
