#!/usr/bin/env bash
# TENT + direct max-confidence fusion.
# Reuses direct preds.pkl from P0 ensemble run, dumps TENT preds.pkl, merges via union+rotated NMS.
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> <ensemble_wr> <tent_wr> [out_root]}"
ENS_WR="${2:?}"
TENT_WR="${3:?}"
OUT_ROOT="${4:-${ENS_WR}/tent_ensemble}"

NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29512}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)
NMS_IOU="${TENT_ENS_NMS_IOU:-0.1}"

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/driver.log"
echo "[tent_ens] $(date -Is) ens_wr=${ENS_WR} tent_wr=${TENT_WR} out=${OUT_ROOT} cuda=${CUDA_VISIBLE_DEVICES:-unset} nms_iou=${NMS_IOU}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

dump_tent() {
  local corr="$1" out_pkl="$2"
  local ckpt="${TENT_WR}/${corr}/tent/latest.pth"
  if [[ -f "${out_pkl}" ]]; then
    echo "[tent_ens] $(date -Is) SKIP dump corr=${corr} (exists)" | tee -a "${LOG}"
    return 0
  fi
  mkdir -p "$(dirname "${out_pkl}")"
  echo "[tent_ens] $(date -Is) DUMP TENT corr=${corr} ckpt=${ckpt}" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    ddp_run test.py "${CFG}" "${ckpt}" \
      --work-dir "$(dirname "${out_pkl}")" \
      --out "${out_pkl}" \
      --eval mAP
}

merge_direct_tent() {
  local corr="$1" direct_pkl="$2" tent_pkl="$3" out_dir="$4"
  if [[ -f "${out_dir}/eval_ensemble.json" ]]; then
    echo "[tent_ens] $(date -Is) SKIP merge corr=${corr} (exists)" | tee -a "${LOG}"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "[tent_ens] $(date -Is) MERGE direct+TENT corr=${corr}" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    "${PYTHON_BIN}" -u tools/ensemble_merge_eval.py \
      --source-pkl "${direct_pkl}" \
      --adapted-pkl "${tent_pkl}" \
      --cfg "${CFG}" \
      --work-dir "${out_dir}" \
      --nms-iou "${NMS_IOU}" \
      > "${out_dir}/merge.log" 2>&1
}

# Phase A: dump TENT predictions (reuse direct from ENS_WR)
for CORR in "${CORRS[@]}"; do
  tent_pkl="${OUT_ROOT}/${CORR}/tent_preds/preds.pkl"
  dump_tent "${CORR}" "${tent_pkl}"
done

# Phase B: merge
for CORR in "${CORRS[@]}"; do
  direct_pkl="${ENS_WR}/${CORR}/direct/preds.pkl"
  tent_pkl="${OUT_ROOT}/${CORR}/tent_preds/preds.pkl"
  out_dir="${OUT_ROOT}/${CORR}/ensemble"
  if [[ ! -f "${direct_pkl}" ]]; then
    echo "[tent_ens] $(date -Is) MISSING direct pkl for ${CORR}" | tee -a "${LOG}"
    continue
  fi
  if [[ ! -f "${tent_pkl}" ]]; then
    echo "[tent_ens] $(date -Is) MISSING tent pkl for ${CORR}" | tee -a "${LOG}"
    continue
  fi
  merge_direct_tent "${CORR}" "${direct_pkl}" "${tent_pkl}" "${out_dir}"
done

echo "[tent_ens] $(date -Is) DONE" | tee -a "${LOG}"
