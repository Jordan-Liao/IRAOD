#!/usr/bin/env bash
# Per-class pseudo-label cap fix: rerun adapt_cga with RSAR_PSEUDO_CAP + LoRA on 4 heavy corrs.
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_capfix_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29508}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CGA_MODE="${CGA_MODE:-sfodrs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${WORK_ROOT}"
LOG="${WORK_ROOT}/launch.log"
echo "[capfix] $(date -Is) source=${SOURCE_CKPT} work_root=${WORK_ROOT}" | tee -a "${LOG}"
echo "[capfix] $(date -Is) cuda=${CUDA_VISIBLE_DEVICES:-unset} ngpus=${NGPUS} port=${MASTER_PORT}" | tee -a "${LOG}"
echo "[capfix] $(date -Is) lora=${SARCLIP_LORA:-unset} cap=${RSAR_PSEUDO_CAP:-unset}" | tee -a "${LOG}"
echo "[capfix] $(date -Is) score_thr=${RSAR_PSEUDO_SCORE_THR:-default} burn=${RSAR_BURN_IN_EPOCHS:-0} weight_u=${RSAR_WEIGHT_U:-1.0} lr=${RSAR_ADAPT_LR:-default} epochs=${RSAR_ADAPT_EPOCHS:-12}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  echo "[capfix] $(date -Is) === corr=${CORR} adapt_cga_cap ===" | tee -a "${LOG}"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=1 \
    ddp_run train.py "${CFG}" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_cap" \
      --teacher-ckpt "${SOURCE_CKPT}" \
      --cga-scorer sarclip --cga-templates 'A SAR image of a {}' --cga-tau 100 --cga-expand-ratio 0.4 \
      --no-validate

  echo "[capfix] $(date -Is) === corr=${CORR} eval ===" | tee -a "${LOG}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training_plus_cga_cap/latest_ema.pth" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_cap/eval_target" --eval mAP
done

echo "[capfix] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${LOG}"
