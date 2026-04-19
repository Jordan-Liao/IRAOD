#!/usr/bin/env bash
# P1 TENT: per-corruption source-free BN-affine + entropy-loss adaptation.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29510 \
#     bash scripts/run_rsar_sfodrs_tent.sh <source_ckpt> [work_root]
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_tent_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29510}"
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

mkdir -p "${WORK_ROOT}"
LOG="${WORK_ROOT}/launch.log"
echo "[tent] $(date -Is) source=${SOURCE_CKPT} work_root=${WORK_ROOT}" | tee -a "${LOG}"
echo "[tent] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT} epochs=${TENT_EPOCHS} lr=${TENT_LR} conf_thr=${TENT_CONF}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  TENT_DIR="${WORK_ROOT}/${CORR}/tent"
  TENT_CKPT="${TENT_DIR}/latest.pth"
  EVAL_DIR="${WORK_ROOT}/${CORR}/tent_eval"
  mkdir -p "${TENT_DIR}" "${EVAL_DIR}"

  if [[ -f "${TENT_CKPT}" ]]; then
    echo "[tent] $(date -Is) SKIP cal corr=${CORR} (exists)" | tee -a "${LOG}"
  else
    echo "[tent] $(date -Is) step=adapt corr=${CORR}" | tee -a "${LOG}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "${PYTHON_BIN}" -u tools/tent_adapt_per_corr.py \
      --source-cfg "${CFG}" \
      --source-ckpt "${SOURCE_CKPT}" \
      --corruption "${CORR}" \
      --target-img "dataset/RSAR/corruptions/${CORR}/val/images" \
      --ref-ann "${REF_ANN}" \
      --out "${TENT_CKPT}" \
      --epochs "${TENT_EPOCHS}" --lr "${TENT_LR}" --conf-thr "${TENT_CONF}" \
      --samples-per-gpu 2 --workers-per-gpu 2 > "${TENT_DIR}/tent.log" 2>&1
  fi

  if [[ -f "${EVAL_DIR}"/eval_*.json ]]; then
    echo "[tent] $(date -Is) SKIP eval corr=${CORR} (exists)" | tee -a "${LOG}"
  else
    echo "[tent] $(date -Is) step=eval corr=${CORR}" | tee -a "${LOG}"
    RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
      ddp_run test.py "${CFG}" "${TENT_CKPT}" \
        --work-dir "${EVAL_DIR}" --eval mAP > "${EVAL_DIR}/eval.log" 2>&1
  fi
done

echo "[tent] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${LOG}"
