#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <corr> <cuda_visible_devices> <master_port> <work_root> <source_ckpt> [config]" >&2
  exit 2
fi

CORR="$1"
CUDA_DEVICES="$2"
MASTER_PORT="$3"
WORK_ROOT="$4"
SOURCE_CKPT="$5"
CFG="${6:-configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

ROOT_DIR="/mnt/SSD1_8TB/zechuan/IRAOD"
cd "${ROOT_DIR}"

LOG_FILE="${WORK_ROOT}/${CORR}/single_corr_launch.log"
mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  echo "[single_corr] $(date -Iseconds) corr=${CORR} $*" | tee -a "${LOG_FILE}"
}

ddp_run() {
  local script="$1"
  shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port="${MASTER_PORT}" \
    --use_env \
    "${script}" "$@" --launcher pytorch
}

log start "cuda=${CUDA_VISIBLE_DEVICES}" "port=${MASTER_PORT}" "work_root=${WORK_ROOT}"

log step=direct_test
RSAR_STAGE=direct_test RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
  ddp_run test.py "${CFG}" "${SOURCE_CKPT}" \
    --work-dir "${WORK_ROOT}/${CORR}/direct_test" --eval mAP

log step=adapt_nocga
RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
  ddp_run train.py "${CFG}" \
    --work-dir "${WORK_ROOT}/${CORR}/self_training" \
    --teacher-ckpt "${SOURCE_CKPT}" --no-validate

log step=eval_nocga
RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
  ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training/latest_ema.pth" \
    --work-dir "${WORK_ROOT}/${CORR}/self_training/eval_target" --eval mAP

log step=adapt_cga
RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=1 \
  ddp_run train.py "${CFG}" \
    --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga" \
    --teacher-ckpt "${SOURCE_CKPT}" \
    --cga-scorer sarclip --cga-templates "A SAR image of a {}" \
    --cga-tau 100 --cga-expand-ratio 0.4 --no-validate

log step=eval_cga
RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
  ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training_plus_cga/latest_ema.pth" \
    --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga/eval_target" --eval mAP

log done
