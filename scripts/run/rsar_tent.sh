#!/usr/bin/env bash
# Unified RSAR TENT entrypoint: adapt, eval, ensemble, or full.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/iraod.sh"
iraod_cd_root

TENT_MODE="${TENT_MODE:-${1:-full}}"
SOURCE_CKPT="${SOURCE_CKPT:-${2:-}}"
WORK_ROOT="${WORK_ROOT:-${3:-work_dirs/rsar_sfodrs_tent_$(date +%Y%m%d_%H%M%S)}}"
CONFIG="${CONFIG:-configs/current/rsar_sfodrs.py}"
RSAR_DATA_ROOT="${RSAR_DATA_ROOT:-dataset/RSAR}"
REF_ANN="${REF_ANN:-${RSAR_DATA_ROOT}/val/annfiles}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29510}"

TENT_EPOCHS="${TENT_EPOCHS:-2}"
TENT_LR="${TENT_LR:-0.0001}"
TENT_CONF="${TENT_CONF:-0.5}"
TENT_MAX_BATCHES="${TENT_MAX_BATCHES:-500}"
NMS_IOU="${TENT_ENS_NMS_IOU:-0.1}"

LOG="${WORK_ROOT}/launch.log"
mkdir -p "${WORK_ROOT}"
iraod_log_file "${LOG}" "tent" "mode=${TENT_MODE} source=${SOURCE_CKPT:-unset} work_root=${WORK_ROOT}"
iraod_log_file "${LOG}" "tent" "cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT} epochs=${TENT_EPOCHS} lr=${TENT_LR} conf=${TENT_CONF}"

require_source_ckpt() {
  if [[ -z "${SOURCE_CKPT}" ]]; then
    echo "SOURCE_CKPT is required for TENT_MODE=${TENT_MODE}" >&2
    exit 2
  fi
}

run_adapt() {
  require_source_ckpt
  local corr tent_dir tent_ckpt target_img first_cuda
  for corr in "${IRAOD_CORRUPTIONS[@]}"; do
    tent_dir="${WORK_ROOT}/${corr}/tent"
    tent_ckpt="${tent_dir}/latest.pth"
    target_img="${RSAR_DATA_ROOT}/corruptions/${corr}/val/images"
    mkdir -p "${tent_dir}"
    if [[ -f "${tent_ckpt}" && ! iraod_is_dry_run ]]; then
      iraod_log_file "${LOG}" "tent" "skip adapt corr=${corr} checkpoint exists"
      continue
    fi

    iraod_log_file "${LOG}" "tent" "step=adapt corr=${corr}"
    first_cuda="${CUDA_VISIBLE_DEVICES:-}"
    first_cuda="${first_cuda%%,*}"
    local env_args=()
    if [[ -n "${first_cuda}" ]]; then
      env_args+=("CUDA_VISIBLE_DEVICES=${first_cuda}")
    fi
    iraod_env_run_logged "${tent_dir}/tent.log" \
      "${env_args[@]}" \
      -- "${PYTHON_BIN}" -u tools/tent_adapt_per_corr.py \
      --source-cfg "${CONFIG}" \
      --source-ckpt "${SOURCE_CKPT}" \
      --corruption "${corr}" \
      --target-img "${target_img}" \
      --ref-ann "${REF_ANN}" \
      --out "${tent_ckpt}" \
      --epochs "${TENT_EPOCHS}" \
      --lr "${TENT_LR}" \
      --conf-thr "${TENT_CONF}" \
      --samples-per-gpu 2 \
      --workers-per-gpu 2 \
      --max-batches "${TENT_MAX_BATCHES}"
  done
}

run_eval() {
  local corr tent_ckpt eval_dir
  for corr in "${IRAOD_CORRUPTIONS[@]}"; do
    tent_ckpt="${WORK_ROOT}/${corr}/tent/latest.pth"
    eval_dir="${WORK_ROOT}/${corr}/tent_eval"
    if [[ ! -f "${tent_ckpt}" && ! iraod_is_dry_run ]]; then
      iraod_log_file "${LOG}" "tent" "skip eval corr=${corr} missing checkpoint"
      continue
    fi
    if compgen -G "${eval_dir}/eval_*.json" >/dev/null && ! iraod_is_dry_run; then
      iraod_log_file "${LOG}" "tent" "skip eval corr=${corr} already evaluated"
      continue
    fi
    mkdir -p "${eval_dir}"
    iraod_log_file "${LOG}" "tent" "step=eval corr=${corr}"
    iraod_ddp_env_run_logged "${eval_dir}/eval.log" \
      "RSAR_STAGE=target_eval" \
      "RSAR_TARGET_DOMAIN=${corr}" \
      "RSAR_USE_CGA=0" \
      "RSAR_USE_TTA=0" \
      -- test.py "${CONFIG}" "${tent_ckpt}" \
      --work-dir "${eval_dir}" \
      --eval mAP
  done
}

dump_tent_predictions() {
  local corr="$1"
  local out_pkl="$2"
  local ckpt="${WORK_ROOT}/${corr}/tent/latest.pth"
  mkdir -p "$(dirname "${out_pkl}")"
  iraod_log_file "${LOG}" "tent" "step=dump_tent corr=${corr}"
  iraod_ddp_env_run \
    "RSAR_STAGE=target_eval" \
    "RSAR_TARGET_DOMAIN=${corr}" \
    "RSAR_USE_CGA=0" \
    "RSAR_USE_TTA=0" \
    -- test.py "${CONFIG}" "${ckpt}" \
    --work-dir "$(dirname "${out_pkl}")" \
    --out "${out_pkl}" \
    --eval mAP
}

merge_direct_tent() {
  local corr="$1"
  local direct_pkl="$2"
  local tent_pkl="$3"
  local out_dir="$4"
  mkdir -p "${out_dir}"
  iraod_log_file "${LOG}" "tent" "step=ensemble corr=${corr}"
  iraod_env_run_logged "${out_dir}/merge.log" \
    "RSAR_STAGE=target_eval" \
    "RSAR_TARGET_DOMAIN=${corr}" \
    "RSAR_USE_CGA=0" \
    "RSAR_USE_TTA=0" \
    -- "${PYTHON_BIN}" -u tools/ensemble_merge_eval.py \
    --source-pkl "${direct_pkl}" \
    --adapted-pkl "${tent_pkl}" \
    --cfg "${CONFIG}" \
    --work-dir "${out_dir}" \
    --nms-iou "${NMS_IOU}"
}

run_ensemble() {
  local direct_work_root="${DIRECT_WORK_ROOT:-${ENS_WORK_ROOT:-}}"
  if [[ -z "${direct_work_root}" ]]; then
    echo "DIRECT_WORK_ROOT or ENS_WORK_ROOT is required for TENT_MODE=ensemble" >&2
    exit 2
  fi

  local out_root="${OUT_ROOT:-${direct_work_root}/tent_ensemble}"
  local corr tent_pkl direct_pkl out_dir
  mkdir -p "${out_root}"
  for corr in "${IRAOD_CORRUPTIONS[@]}"; do
    tent_pkl="${out_root}/${corr}/tent_preds/preds.pkl"
    direct_pkl="${direct_work_root}/${corr}/direct/preds.pkl"
    out_dir="${out_root}/${corr}/ensemble"
    if [[ ! -f "${tent_pkl}" || iraod_is_dry_run ]]; then
      dump_tent_predictions "${corr}" "${tent_pkl}"
    fi
    if [[ ! -f "${direct_pkl}" && ! iraod_is_dry_run ]]; then
      iraod_log_file "${LOG}" "tent" "skip ensemble corr=${corr} missing direct predictions"
      continue
    fi
    merge_direct_tent "${corr}" "${direct_pkl}" "${tent_pkl}" "${out_dir}"
  done
}

case "${TENT_MODE}" in
  full)
    run_adapt
    run_eval
    ;;
  adapt)
    run_adapt
    ;;
  eval)
    run_eval
    ;;
  ensemble|fusion)
    run_ensemble
    ;;
  *)
    echo "Unknown TENT_MODE=${TENT_MODE}. Use full|adapt|eval|ensemble." >&2
    exit 2
    ;;
esac

iraod_log_file "${LOG}" "tent" "done mode=${TENT_MODE}"
