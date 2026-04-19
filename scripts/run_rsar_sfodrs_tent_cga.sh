#!/usr/bin/env bash
# Plan 3: TENT ckpt as teacher + CGA (SARCLIP LoRA) adapt + eval.
# Uses per-corruption TENT ckpt from Plan 1 as starting point (replaces source_ckpt).
# Applies E0113 heavyfix-style hyperparams: score_thr per-class, burn-in 1, adapt_epochs 3 (shorter
# than heavyfix 6 to keep total time manageable), early_stop max_majority=0.85.
set -Eeuo pipefail

TENT_WR="${1:?usage: $0 <tent_work_root> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_tent_cga_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29513}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CGA_MODE="${CGA_MODE:-sfodrs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${WORK_ROOT}"
LOG="${WORK_ROOT}/launch.log"
echo "[tent_cga] $(date -Is) tent_wr=${TENT_WR} work_root=${WORK_ROOT}" | tee -a "${LOG}"
echo "[tent_cga] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT} lora=${SARCLIP_LORA:-unset}" | tee -a "${LOG}"
echo "[tent_cga] $(date -Is) score_thr=${RSAR_PSEUDO_SCORE_THR:-default} burn=${RSAR_BURN_IN_EPOCHS:-0} weight_u=${RSAR_WEIGHT_U:-1.0} lr=${RSAR_ADAPT_LR:-default} epochs=${RSAR_ADAPT_EPOCHS:-12}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  TENT_CKPT="${TENT_WR}/${CORR}/tent/latest.pth"
  if [[ ! -f "${TENT_CKPT}" ]]; then
    echo "[tent_cga] $(date -Is) SKIP corr=${CORR} (no TENT ckpt)" | tee -a "${LOG}"
    continue
  fi

  echo "[tent_cga] $(date -Is) === corr=${CORR} adapt_cga_from_tent ===" | tee -a "${LOG}"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=1 \
    ddp_run train.py "${CFG}" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_tent" \
      --teacher-ckpt "${TENT_CKPT}" \
      --cga-scorer sarclip --cga-templates 'A SAR image of a {}' --cga-tau 100 --cga-expand-ratio 0.4 \
      --no-validate

  echo "[tent_cga] $(date -Is) === corr=${CORR} eval ===" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training_plus_cga_tent/latest_ema.pth" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_tent/eval_target" --eval mAP
done

echo "[tent_cga] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${LOG}"
