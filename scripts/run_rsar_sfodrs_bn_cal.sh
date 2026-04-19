#!/usr/bin/env bash
# Per-corruption BN calibration (TENT-style) + DDP eval on 7 RSAR corruptions.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29507 \
#     bash scripts/run_rsar_sfodrs_bn_cal.sh <source_ckpt> [work_root]
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_bn_cal_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29507}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
REF_ANN=dataset/RSAR/val/annfiles
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${WORK_ROOT}"
LOG="${WORK_ROOT}/launch.log"
echo "[bn_cal] $(date -Is) source=${SOURCE_CKPT} work_root=${WORK_ROOT}" | tee -a "${LOG}"
echo "[bn_cal] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  CAL_DIR="${WORK_ROOT}/${CORR}/bn_cal"
  CAL_CKPT="${CAL_DIR}/latest.pth"
  EVAL_DIR="${WORK_ROOT}/${CORR}/bn_eval"
  mkdir -p "${CAL_DIR}" "${EVAL_DIR}"

  if [[ -f "${CAL_CKPT}" ]]; then
    echo "[bn_cal] $(date -Is) SKIP cal corr=${CORR} (already done)" | tee -a "${LOG}"
  else
    echo "[bn_cal] $(date -Is) step=calibrate corr=${CORR}" | tee -a "${LOG}"
    # Single-GPU BN forward-only calibration on corruption val images
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "${PYTHON_BIN}" -u tools/bn_calibrate_per_corr.py \
      --source-cfg "${CFG}" \
      --source-ckpt "${SOURCE_CKPT}" \
      --corruption "${CORR}" \
      --target-img "dataset/RSAR/corruptions/${CORR}/val/images" \
      --ref-ann "${REF_ANN}" \
      --out "${CAL_CKPT}" \
      --samples-per-gpu 8 --workers-per-gpu 4 > "${CAL_DIR}/cal.log" 2>&1
  fi

  echo "[bn_cal] $(date -Is) step=eval corr=${CORR}" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    ddp_run test.py "${CFG}" "${CAL_CKPT}" \
      --work-dir "${EVAL_DIR}" --eval mAP > "${EVAL_DIR}/eval.log" 2>&1
done

echo "[bn_cal] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${LOG}"
