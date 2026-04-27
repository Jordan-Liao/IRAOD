#!/usr/bin/env bash
# Recommended RSAR SFOD-RS full entrypoint.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/iraod.sh"
iraod_cd_root

SOURCE_CKPT="${SOURCE_CKPT:-${1:-auto}}"
WORK_ROOT="${WORK_ROOT:-${2:-work_dirs/rsar_sfodrs_$(date +%Y%m%d_%H%M%S)}}"
CONFIG="${CONFIG:-${TRAIN_CONFIG:-configs/current/rsar_sfodrs.py}}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG="${WORK_ROOT}/launch.log"

mkdir -p "${WORK_ROOT}"
iraod_log_file "${LOG}" "rsar-full" "source=${SOURCE_CKPT} work_root=${WORK_ROOT}"
iraod_log_file "${LOG}" "rsar-full" "config=${CONFIG} cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT}"

run_source_train_if_needed() {
  local source_dir="${WORK_ROOT}/source_train"
  local source_latest="${source_dir}/latest.pth"
  local train_args=(--work-dir "${source_dir}")

  iraod_append_if_set train_args --max-epochs "${RSAR_SOURCE_EPOCHS:-}"

  if [[ "${SOURCE_CKPT}" != "auto" ]]; then
    return 0
  fi

  if [[ -f "${source_latest}" && ! iraod_is_dry_run ]]; then
    iraod_log_file "${LOG}" "rsar-full" "reuse source_train checkpoint ${source_latest}"
  else
    iraod_log_file "${LOG}" "rsar-full" "step=source_train"
    iraod_ddp_env_run \
      "RSAR_STAGE=source_train" \
      "RSAR_TARGET_DOMAIN=clean" \
      "RSAR_USE_CGA=0" \
      -- train.py "${CONFIG}" "${train_args[@]}"
  fi

  SOURCE_CKPT="${source_latest}"
  export SOURCE_CKPT
  iraod_log_file "${LOG}" "rsar-full" "auto source resolved to ${SOURCE_CKPT}"
}

run_clean_test() {
  if iraod_source_ckpt_is_null "${SOURCE_CKPT}"; then
    iraod_log_file "${LOG}" "rsar-full" "skip source_clean_test source=${SOURCE_CKPT}"
    return 0
  fi

  iraod_log_file "${LOG}" "rsar-full" "step=source_clean_test"
  iraod_ddp_env_run \
    "RSAR_STAGE=source_clean_test" \
    "RSAR_TARGET_DOMAIN=clean" \
    "RSAR_USE_CGA=0" \
    -- test.py "${CONFIG}" "${SOURCE_CKPT}" \
    --work-dir "${WORK_ROOT}/clean/source_clean_test" \
    --eval mAP
}

run_source_train_if_needed
run_clean_test

for CORR in "${IRAOD_CORRUPTIONS[@]}"; do
  CORR="${CORR}" SOURCE_CKPT="${SOURCE_CKPT}" CONFIG="${CONFIG}" WORK_ROOT="${WORK_ROOT}" \
    NGPUS="${NGPUS}" MASTER_PORT="${MASTER_PORT}" \
    bash scripts/run/rsar_sfodrs_domain.sh
done

iraod_log_file "${LOG}" "rsar-full" "step=collect"
WORK_ROOT="${WORK_ROOT}" bash scripts/run/collect_results.sh

iraod_log_file "${LOG}" "rsar-full" "done work_root=${WORK_ROOT}"
