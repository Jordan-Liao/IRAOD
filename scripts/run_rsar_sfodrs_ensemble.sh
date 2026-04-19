#!/usr/bin/env bash
# P0 ensemble: run source_ckpt + adapted_ckpt on each corruption test set,
# merge predictions by union + rotated NMS, re-evaluate mAP.
# Produces eval_ensemble.json alongside each corruption's adapted ckpt.
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> <fixed_work_root> [heavyfix_work_root] [out_root]}"
FIXED_WR="${2:?}"
HEAVYFIX_WR="${3:-}"
OUT_ROOT="${4:-${FIXED_WR}/ensemble}"

NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29509}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NMS_IOU="${ENSEMBLE_NMS_IOU:-0.1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)
HEAVY=(noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/driver.log"
echo "[ensemble] $(date -Is) source=${SOURCE_CKPT}" | tee -a "${LOG}"
echo "[ensemble] $(date -Is) fixed_wr=${FIXED_WR}" | tee -a "${LOG}"
echo "[ensemble] $(date -Is) heavyfix_wr=${HEAVYFIX_WR:-none}" | tee -a "${LOG}"
echo "[ensemble] $(date -Is) out_root=${OUT_ROOT}" | tee -a "${LOG}"
echo "[ensemble] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT} nms_iou=${NMS_IOU}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

dump_predictions() {
  local ckpt="$1" corr="$2" out_pkl="$3"
  if [[ -f "${out_pkl}" ]]; then
    echo "[ensemble] $(date -Is) SKIP dump corr=${corr} out=${out_pkl} (exists)" | tee -a "${LOG}"
    return 0
  fi
  echo "[ensemble] $(date -Is) DUMP corr=${corr} ckpt=${ckpt} -> ${out_pkl}" | tee -a "${LOG}"
  mkdir -p "$(dirname "${out_pkl}")"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    ddp_run test.py "${CFG}" "${ckpt}" \
      --work-dir "$(dirname "${out_pkl}")" \
      --out "${out_pkl}" \
      --eval mAP
}

merge_and_eval() {
  local corr="$1" src_pkl="$2" adp_pkl="$3" out_dir="$4"
  if [[ -f "${out_dir}/eval_ensemble.json" ]]; then
    echo "[ensemble] $(date -Is) SKIP merge corr=${corr} (exists)" | tee -a "${LOG}"
    return 0
  fi
  echo "[ensemble] $(date -Is) MERGE corr=${corr} -> ${out_dir}" | tee -a "${LOG}"
  mkdir -p "${out_dir}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    "${PYTHON_BIN}" -u tools/ensemble_merge_eval.py \
      --source-pkl "${src_pkl}" \
      --adapted-pkl "${adp_pkl}" \
      --cfg "${CFG}" \
      --work-dir "${out_dir}" \
      --nms-iou "${NMS_IOU}" \
      > "${out_dir}/merge.log" 2>&1
}

# Phase A: dump source_ckpt predictions once per corruption (this = direct_test.pkl)
for CORR in "${CORRS[@]}"; do
  src_pkl="${OUT_ROOT}/${CORR}/direct/preds.pkl"
  dump_predictions "${SOURCE_CKPT}" "${CORR}" "${src_pkl}"
done

# Phase B: dump adapted ckpt predictions per corruption (prefer heavyfix LoRA > fixed CGA > fixed no-CGA)
for CORR in "${CORRS[@]}"; do
  # Pick the strongest adapted ckpt available for this corruption
  LORA_CKPT="${HEAVYFIX_WR}/${CORR}/self_training_plus_cga_lora/latest_ema.pth"
  CGA_CKPT="${FIXED_WR}/${CORR}/self_training_plus_cga/latest_ema.pth"
  SELF_CKPT="${FIXED_WR}/${CORR}/self_training/latest_ema.pth"
  if [[ -n "${HEAVYFIX_WR}" && -f "${LORA_CKPT}" ]]; then
    ADP_LABEL="cga_lora"
    ADP_CKPT="${LORA_CKPT}"
  elif [[ -f "${CGA_CKPT}" ]]; then
    ADP_LABEL="cga"
    ADP_CKPT="${CGA_CKPT}"
  else
    ADP_LABEL="self"
    ADP_CKPT="${SELF_CKPT}"
  fi
  adp_pkl="${OUT_ROOT}/${CORR}/adapted_${ADP_LABEL}/preds.pkl"
  dump_predictions "${ADP_CKPT}" "${CORR}" "${adp_pkl}"
  echo "[ensemble] $(date -Is) corr=${CORR} adapted source=${ADP_LABEL}" | tee -a "${LOG}"
done

# Phase C: merge + eval
for CORR in "${CORRS[@]}"; do
  src_pkl="${OUT_ROOT}/${CORR}/direct/preds.pkl"
  # Find the adapted pkl we just dumped
  adp_pkl=""
  for lbl in cga_lora cga self; do
    cand="${OUT_ROOT}/${CORR}/adapted_${lbl}/preds.pkl"
    if [[ -f "${cand}" ]]; then
      adp_pkl="${cand}"
      break
    fi
  done
  if [[ -z "${adp_pkl}" ]]; then
    echo "[ensemble] $(date -Is) SKIP merge corr=${CORR} (no adapted pkl)" | tee -a "${LOG}"
    continue
  fi
  out_dir="${OUT_ROOT}/${CORR}/ensemble"
  merge_and_eval "${CORR}" "${src_pkl}" "${adp_pkl}" "${out_dir}"
done

echo "[ensemble] $(date -Is) DONE" | tee -a "${LOG}"
