#!/usr/bin/env bash
# P1 TENT eval: DDP eval of all 7 TENT ckpts on their respective corruption test sets.
set -Eeuo pipefail

TENT_WR="${1:?usage: $0 <tent_work_root>}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29511}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

LOG="${TENT_WR}/eval_driver.log"
echo "[tent-eval] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  TENT_CKPT="${TENT_WR}/${CORR}/tent/latest.pth"
  EVAL_DIR="${TENT_WR}/${CORR}/tent_eval"
  if [[ ! -f "${TENT_CKPT}" ]]; then
    echo "[tent-eval] $(date -Is) SKIP corr=${CORR} (no ckpt)" | tee -a "${LOG}"
    continue
  fi
  if ls "${EVAL_DIR}"/eval_*.json >/dev/null 2>&1; then
    echo "[tent-eval] $(date -Is) SKIP corr=${CORR} (already evaluated)" | tee -a "${LOG}"
    continue
  fi
  mkdir -p "${EVAL_DIR}"
  echo "[tent-eval] $(date -Is) step=eval corr=${CORR}" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 RSAR_USE_TTA=0 \
    ddp_run test.py "${CFG}" "${TENT_CKPT}" \
      --work-dir "${EVAL_DIR}" --eval mAP > "${EVAL_DIR}/eval.log" 2>&1
done

echo "[tent-eval] $(date -Is) DONE" | tee -a "${LOG}"
