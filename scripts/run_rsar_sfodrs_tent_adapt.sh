#!/usr/bin/env bash
# P1 TENT adapt-only pass on single GPU (no eval here; eval runs later via DDP).
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_tent_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
REF_ANN=dataset/RSAR/val/annfiles
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

TENT_EPOCHS="${TENT_EPOCHS:-2}"
TENT_LR="${TENT_LR:-0.0001}"
TENT_CONF="${TENT_CONF:-0.5}"
TENT_MAX_BATCHES="${TENT_MAX_BATCHES:-500}"

mkdir -p "${WORK_ROOT}"
LOG="${WORK_ROOT}/launch.log"
echo "[tent-adapt] $(date -Is) source=${SOURCE_CKPT} work_root=${WORK_ROOT}" | tee -a "${LOG}"
echo "[tent-adapt] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} epochs=${TENT_EPOCHS} lr=${TENT_LR} conf=${TENT_CONF} max_batches=${TENT_MAX_BATCHES}" | tee -a "${LOG}"

for CORR in "${CORRS[@]}"; do
  TENT_DIR="${WORK_ROOT}/${CORR}/tent"
  TENT_CKPT="${TENT_DIR}/latest.pth"
  mkdir -p "${TENT_DIR}"
  if [[ -f "${TENT_CKPT}" ]]; then
    echo "[tent-adapt] $(date -Is) SKIP corr=${CORR} (ckpt exists)" | tee -a "${LOG}"
    continue
  fi
  echo "[tent-adapt] $(date -Is) step=adapt corr=${CORR}" | tee -a "${LOG}"
  "${PYTHON_BIN}" -u tools/tent_adapt_per_corr.py \
    --source-cfg "${CFG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --corruption "${CORR}" \
    --target-img "dataset/RSAR/corruptions/${CORR}/val/images" \
    --ref-ann "${REF_ANN}" \
    --out "${TENT_CKPT}" \
    --epochs "${TENT_EPOCHS}" --lr "${TENT_LR}" --conf-thr "${TENT_CONF}" \
    --samples-per-gpu 2 --workers-per-gpu 2 \
    --max-batches "${TENT_MAX_BATCHES}" > "${TENT_DIR}/tent.log" 2>&1
  echo "[tent-adapt] $(date -Is) done corr=${CORR}" | tee -a "${LOG}"
done

echo "[tent-adapt] $(date -Is) ALL_ADAPT_DONE work_root=${WORK_ROOT}" | tee -a "${LOG}"
