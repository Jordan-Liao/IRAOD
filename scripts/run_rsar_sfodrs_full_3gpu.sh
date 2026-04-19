#!/usr/bin/env bash
# SFOD-RS faithful 7-corruption source-free adaptation on RSAR, 3-GPU DDP.
# Usage: NGPUS=3 CUDA_VISIBLE_DEVICES=6,7,8 MASTER_PORT=29501 \
#          bash scripts/run_rsar_sfodrs_full_3gpu.sh <source_ckpt> [work_root]
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29501}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CGA_MODE="${CGA_MODE:-sfodrs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${WORK_ROOT}"
echo "[full_3gpu] $(date -Is) source=${SOURCE_CKPT}"       | tee -a "${WORK_ROOT}/launch.log"
echo "[full_3gpu] $(date -Is) work_root=${WORK_ROOT}"        | tee -a "${WORK_ROOT}/launch.log"
echo "[full_3gpu] $(date -Is) ngpus=${NGPUS} port=${MASTER_PORT}" | tee -a "${WORK_ROOT}/launch.log"
echo "[full_3gpu] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} cga_mode=${CGA_MODE}" | tee -a "${WORK_ROOT}/launch.log"

ddp_run() {
  local script="$1"; shift
  if [[ "${NGPUS}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.launch \
      --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
      "${script}" "$@" --launcher pytorch
  else
    "${PYTHON_BIN}" -u "${script}" "$@"
  fi
}

# 0) source_clean_test
echo "[full_3gpu] $(date -Is) step=source_clean_test" | tee -a "${WORK_ROOT}/launch.log"
RSAR_STAGE=source_clean_test RSAR_TARGET_DOMAIN=clean RSAR_USE_CGA=0 \
  ddp_run test.py "${CFG}" "${SOURCE_CKPT}" \
    --work-dir "${WORK_ROOT}/clean/source_clean_test" --eval mAP

for CORR in "${CORRS[@]}"; do
  echo "[full_3gpu] $(date -Is) === corruption=${CORR} ===" | tee -a "${WORK_ROOT}/launch.log"

  echo "[full_3gpu] $(date -Is) step=direct_test corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=direct_test RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${SOURCE_CKPT}" \
      --work-dir "${WORK_ROOT}/${CORR}/direct_test" --eval mAP

  echo "[full_3gpu] $(date -Is) step=adapt_nocga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run train.py "${CFG}" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training" \
      --teacher-ckpt "${SOURCE_CKPT}" --no-validate

  echo "[full_3gpu] $(date -Is) step=eval_nocga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training/latest_ema.pth" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training/eval_target" --eval mAP

  echo "[full_3gpu] $(date -Is) step=adapt_cga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=1 \
    ddp_run train.py "${CFG}" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga" \
      --teacher-ckpt "${SOURCE_CKPT}" \
      --cga-scorer sarclip --cga-templates 'A SAR image of a {}' --cga-tau 100 --cga-expand-ratio 0.4 \
      --no-validate

  echo "[full_3gpu] $(date -Is) step=eval_cga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training_plus_cga/latest_ema.pth" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga/eval_target" --eval mAP
done

echo "[full_3gpu] $(date -Is) step=collect" | tee -a "${WORK_ROOT}/launch.log"
"${PYTHON_BIN}" -u tools/collect_rsar_sfodrs_results.py \
  --work-root "${WORK_ROOT}" \
  --out-csv "${WORK_ROOT}/rsar_sfodrs_results.csv" \
  --out-md "${WORK_ROOT}/rsar_sfodrs_results.md"

echo "[full_3gpu] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${WORK_ROOT}/launch.log"
