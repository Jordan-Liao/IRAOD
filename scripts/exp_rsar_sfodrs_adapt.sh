#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <corruption> <source_ckpt> [config] [work_root]" >&2
  echo "Example: $0 chaff work_dirs/rsar_sfodrs_source/latest.pth" >&2
  exit 2
fi

CORR="$1"
SOURCE_CKPT="$2"
CONFIG="${3:-}"
WORK_ROOT="${4:-work_dirs/rsar_sfodrs}"
PYTHON_BIN="${PYTHON:-python3}"

WD_BASE="${WORK_ROOT}/${CORR}"
WD_DIRECT="${WD_BASE}/direct_test"
WD_ADAPT="${WD_BASE}/self_training"
WD_ADAPT_CGA="${WD_BASE}/self_training_plus_cga"

export RSAR_TARGET_DOMAIN="${CORR}"
export CGA_MODE="${CGA_MODE:-sfodrs}"
RSAR_PROTOCOL="${RSAR_PROTOCOL:-sfodrs}"
RSAR_EVAL_CONFIG="${RSAR_EVAL_CONFIG:-configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py}"

if [[ -z "${CONFIG}" ]]; then
  if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
    CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py"
  else
    CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py"
  fi
fi

echo "[exp_rsar_sfodrs_adapt] corr=${CORR}"
echo "[exp_rsar_sfodrs_adapt] source_ckpt=${SOURCE_CKPT}"
echo "[exp_rsar_sfodrs_adapt] config=${CONFIG}"
echo "[exp_rsar_sfodrs_adapt] work_root=${WORK_ROOT}"
echo "[exp_rsar_sfodrs_adapt] cga_mode=${CGA_MODE}"
echo "[exp_rsar_sfodrs_adapt] protocol=${RSAR_PROTOCOL}"
echo "[exp_rsar_sfodrs_adapt] eval_config=${RSAR_EVAL_CONFIG}"

echo "[exp_rsar_sfodrs_adapt] step=direct_test"
if [[ "${SOURCE_CKPT}" == "none" || "${SOURCE_CKPT}" == "null" || "${SOURCE_CKPT}" == "-" ]]; then
  echo "[exp_rsar_sfodrs_adapt] direct_test skipped (SOURCE_CKPT=${SOURCE_CKPT})"
else
  if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
    RSAR_STAGE=direct_test \
    RSAR_USE_CGA=0 \
    "${PYTHON_BIN}" -u test.py "${CONFIG}" "${SOURCE_CKPT}" \
      --work-dir "${WD_DIRECT}" \
      --eval mAP
  else
    "${PYTHON_BIN}" -u test.py "${RSAR_EVAL_CONFIG}" "${SOURCE_CKPT}" \
      --cfg-options corrupt="${CORR}" \
      --work-dir "${WD_DIRECT}" \
      --cga-scorer none \
      --eval mAP
  fi
fi

if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
  echo "[exp_rsar_sfodrs_adapt] step=adapt (no CGA) split=corruptions/${CORR}/val/images -> eval=corruptions/${CORR}/test/images"
  RSAR_STAGE=target_adapt \
  RSAR_USE_CGA=0 \
  "${PYTHON_BIN}" -u train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT}" \
    --teacher-ckpt "${SOURCE_CKPT}" \
    --no-validate
else
  echo "[exp_rsar_sfodrs_adapt] step=adapt (stable Phase 3, no CGA) corrupt=${CORR}"
  export PSEUDO_EARLYSTOP="${PSEUDO_EARLYSTOP:-1}"
  export PSEUDO_EARLYSTOP_WARMUP_EPOCHS="${PSEUDO_EARLYSTOP_WARMUP_EPOCHS:-2}"
  export PSEUDO_EARLYSTOP_PATIENCE="${PSEUDO_EARLYSTOP_PATIENCE:-2}"
  export PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC="${PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC:-0.995}"
  export PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO="${PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO:-5000}"

  TRAIN_ARGS=()
  if [[ "${SOURCE_CKPT}" != "none" && "${SOURCE_CKPT}" != "null" && "${SOURCE_CKPT}" != "-" ]]; then
    TRAIN_ARGS+=(--teacher-ckpt "${SOURCE_CKPT}")
  fi

  "${PYTHON_BIN}" -u train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT}" \
    --cfg-options corrupt="${CORR}" \
    --cga-scorer none \
    "${TRAIN_ARGS[@]}" \
    --no-validate
fi

echo "[exp_rsar_sfodrs_adapt] step=eval self_training"
if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
  RSAR_STAGE=target_eval \
  RSAR_USE_CGA=0 \
  "${PYTHON_BIN}" -u test.py "${CONFIG}" "${WD_ADAPT}/latest_ema.pth" \
    --work-dir "${WD_ADAPT}/eval_target" \
    --eval mAP
else
  "${PYTHON_BIN}" -u test.py "${RSAR_EVAL_CONFIG}" "${WD_ADAPT}/latest_ema.pth" \
    --cfg-options corrupt="${CORR}" \
    --work-dir "${WD_ADAPT}/eval_target" \
    --cga-scorer none \
    --eval mAP
fi

echo "[exp_rsar_sfodrs_adapt] step=adapt (+CGA via SARCLIP) prompt='A SAR image of a {}'"
if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
  RSAR_STAGE=target_adapt \
  RSAR_USE_CGA=1 \
  "${PYTHON_BIN}" -u train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT_CGA}" \
    --teacher-ckpt "${SOURCE_CKPT}" \
    --cga-scorer sarclip \
    --cga-templates "A SAR image of a {}" \
    --cga-tau 100 \
    --cga-expand-ratio 0.4 \
    --no-validate
else
  echo "[exp_rsar_sfodrs_adapt] step=adapt (stable Phase 3, +SARCLIP CGA) corrupt=${CORR}"
  export PSEUDO_EARLYSTOP="${PSEUDO_EARLYSTOP:-1}"
  export PSEUDO_EARLYSTOP_WARMUP_EPOCHS="${PSEUDO_EARLYSTOP_WARMUP_EPOCHS:-2}"
  export PSEUDO_EARLYSTOP_PATIENCE="${PSEUDO_EARLYSTOP_PATIENCE:-2}"
  export PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC="${PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC:-0.995}"
  export PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO="${PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO:-2000}"

  TRAIN_ARGS=()
  if [[ "${SOURCE_CKPT}" != "none" && "${SOURCE_CKPT}" != "null" && "${SOURCE_CKPT}" != "-" ]]; then
    TRAIN_ARGS+=(--teacher-ckpt "${SOURCE_CKPT}")
  fi

  "${PYTHON_BIN}" -u train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT_CGA}" \
    --cfg-options corrupt="${CORR}" \
    --cga-scorer sarclip \
    --cga-templates "A SAR image of a {}" \
    --cga-tau 100 \
    --cga-expand-ratio 0.4 \
    "${TRAIN_ARGS[@]}" \
    --no-validate
fi

echo "[exp_rsar_sfodrs_adapt] step=eval self_training_plus_cga"
if [[ "${RSAR_PROTOCOL}" == "sfodrs" ]]; then
  RSAR_STAGE=target_eval \
  RSAR_USE_CGA=0 \
  "${PYTHON_BIN}" -u test.py "${CONFIG}" "${WD_ADAPT_CGA}/latest_ema.pth" \
    --work-dir "${WD_ADAPT_CGA}/eval_target" \
    --eval mAP
else
  "${PYTHON_BIN}" -u test.py "${RSAR_EVAL_CONFIG}" "${WD_ADAPT_CGA}/latest_ema.pth" \
    --cfg-options corrupt="${CORR}" \
    --work-dir "${WD_ADAPT_CGA}/eval_target" \
    --cga-scorer none \
    --eval mAP
fi

echo "[exp_rsar_sfodrs_adapt] done: ${WD_BASE}"
