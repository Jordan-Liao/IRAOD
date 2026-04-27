#!/usr/bin/env bash
# Run one RSAR SFOD-RS corruption domain: direct, adapt, eval, or full.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/iraod.sh"
iraod_cd_root

CORR="${CORR:-${1:-}}"
SOURCE_CKPT="${SOURCE_CKPT:-${2:-}}"
CONFIG="${CONFIG:-${TRAIN_CONFIG:-${3:-configs/current/rsar_sfodrs.py}}}"
WORK_ROOT="${WORK_ROOT:-${4:-work_dirs/rsar_sfodrs}}"
RSAR_DOMAIN_MODE="${RSAR_DOMAIN_MODE:-full}"
MASTER_PORT="${MASTER_PORT:-29501}"
NGPUS="${NGPUS:-1}"

if [[ -z "${CORR}" || -z "${SOURCE_CKPT}" ]]; then
  echo "Usage: CORR=<corruption> SOURCE_CKPT=<ckpt> WORK_ROOT=<dir> bash scripts/run/rsar_sfodrs_domain.sh" >&2
  echo "   or: bash scripts/run/rsar_sfodrs_domain.sh <corruption> <source_ckpt> [config] [work_root]" >&2
  exit 2
fi

WD_BASE="${WORK_ROOT}/${CORR}"
WD_DIRECT="${WD_BASE}/direct_test"
WD_ADAPT="${WD_BASE}/self_training"
WD_ADAPT_CGA="${WD_BASE}/self_training_plus_cga"
LOG="${WORK_ROOT}/launch.log"

CGA_SCORER_ARG="${CGA_SCORER:-sarclip}"
CGA_TEMPLATES_ARG="${CGA_TEMPLATES:-A SAR image of a {}}"
CGA_TAU_ARG="${CGA_TAU:-100}"
CGA_EXPAND_RATIO_ARG="${CGA_EXPAND_RATIO:-0.4}"

mkdir -p "${WORK_ROOT}"
iraod_log_file "${LOG}" "rsar-domain" "corr=${CORR} mode=${RSAR_DOMAIN_MODE} source=${SOURCE_CKPT}"
iraod_log_file "${LOG}" "rsar-domain" "config=${CONFIG} work_root=${WORK_ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT}"

run_direct() {
  if iraod_source_ckpt_is_null "${SOURCE_CKPT}"; then
    iraod_log_file "${LOG}" "rsar-domain" "skip direct_test corr=${CORR} source=${SOURCE_CKPT}"
    return 0
  fi
  iraod_log_file "${LOG}" "rsar-domain" "step=direct_test corr=${CORR}"
  iraod_ddp_env_run \
    "RSAR_STAGE=direct_test" \
    "RSAR_TARGET_DOMAIN=${CORR}" \
    "RSAR_USE_CGA=0" \
    -- test.py "${CONFIG}" "${SOURCE_CKPT}" \
    --work-dir "${WD_DIRECT}" \
    --eval mAP
}

run_adapt() {
  iraod_log_file "${LOG}" "rsar-domain" "step=adapt_nocga corr=${CORR}"
  iraod_ddp_env_run \
    "RSAR_STAGE=target_adapt" \
    "RSAR_TARGET_DOMAIN=${CORR}" \
    "RSAR_USE_CGA=0" \
    -- train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT}" \
    --teacher-ckpt "${SOURCE_CKPT}" \
    --no-validate
}

run_eval() {
  iraod_log_file "${LOG}" "rsar-domain" "step=eval_nocga corr=${CORR}"
  iraod_ddp_env_run \
    "RSAR_STAGE=target_eval" \
    "RSAR_TARGET_DOMAIN=${CORR}" \
    "RSAR_USE_CGA=0" \
    -- test.py "${CONFIG}" "${WD_ADAPT}/latest_ema.pth" \
    --work-dir "${WD_ADAPT}/eval_target" \
    --eval mAP
}

run_adapt_cga() {
  local cga_args=(
    --cga-scorer "${CGA_SCORER_ARG}"
    --cga-templates "${CGA_TEMPLATES_ARG}"
    --cga-tau "${CGA_TAU_ARG}"
    --cga-expand-ratio "${CGA_EXPAND_RATIO_ARG}"
  )
  iraod_append_if_set cga_args --sarclip-model "${SARCLIP_MODEL:-}"
  iraod_append_if_set cga_args --sarclip-pretrained "${SARCLIP_PRETRAINED:-}"
  iraod_append_if_set cga_args --clip-model "${CLIP_MODEL:-}"

  iraod_log_file "${LOG}" "rsar-domain" "step=adapt_cga corr=${CORR} scorer=${CGA_SCORER_ARG}"
  iraod_ddp_env_run \
    "RSAR_STAGE=target_adapt" \
    "RSAR_TARGET_DOMAIN=${CORR}" \
    "RSAR_USE_CGA=1" \
    -- train.py "${CONFIG}" \
    --work-dir "${WD_ADAPT_CGA}" \
    --teacher-ckpt "${SOURCE_CKPT}" \
    "${cga_args[@]}" \
    --no-validate
}

run_eval_cga() {
  iraod_log_file "${LOG}" "rsar-domain" "step=eval_cga corr=${CORR}"
  iraod_ddp_env_run \
    "RSAR_STAGE=target_eval" \
    "RSAR_TARGET_DOMAIN=${CORR}" \
    "RSAR_USE_CGA=0" \
    -- test.py "${CONFIG}" "${WD_ADAPT_CGA}/latest_ema.pth" \
    --work-dir "${WD_ADAPT_CGA}/eval_target" \
    --eval mAP
}

case "${RSAR_DOMAIN_MODE}" in
  full)
    run_direct
    run_adapt
    run_eval
    run_adapt_cga
    run_eval_cga
    ;;
  direct)
    run_direct
    ;;
  adapt)
    if [[ "${RSAR_USE_CGA:-0}" == "1" ]]; then
      run_adapt_cga
    else
      run_adapt
    fi
    ;;
  eval)
    if [[ "${RSAR_USE_CGA:-0}" == "1" ]]; then
      run_eval_cga
    else
      run_eval
    fi
    ;;
  adapt_cga)
    run_adapt_cga
    ;;
  eval_cga)
    run_eval_cga
    ;;
  *)
    echo "Unknown RSAR_DOMAIN_MODE=${RSAR_DOMAIN_MODE}. Use full|direct|adapt|eval|adapt_cga|eval_cga." >&2
    exit 2
    ;;
esac

iraod_log_file "${LOG}" "rsar-domain" "done corr=${CORR}"
