#!/usr/bin/env bash
# Heavy-corruption fix pass: SARCLIP LoRA + tuned per-class thr + shorter adapt + TTA-aware eval
# Only targets the 4 heavy corruptions where CGA previously helped little / hurt.
set -Eeuo pipefail

SOURCE_CKPT="${1:?usage: $0 <source_ckpt> [work_root]}"
WORK_ROOT="${2:-work_dirs/rsar_sfodrs_heavyfix_$(date +%Y%m%d_%H%M%S)}"
NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29505}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CGA_MODE="${CGA_MODE:-sfodrs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

mkdir -p "${WORK_ROOT}"
echo "[heavyfix] $(date -Is) source=${SOURCE_CKPT}"   | tee -a "${WORK_ROOT}/launch.log"
echo "[heavyfix] $(date -Is) work_root=${WORK_ROOT}"    | tee -a "${WORK_ROOT}/launch.log"
echo "[heavyfix] $(date -Is) ngpus=${NGPUS} port=${MASTER_PORT} cuda=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "${WORK_ROOT}/launch.log"
echo "[heavyfix] $(date -Is) lora=${SARCLIP_LORA:-unset}"  | tee -a "${WORK_ROOT}/launch.log"
echo "[heavyfix] $(date -Is) score_thr=${RSAR_PSEUDO_SCORE_THR:-default}" | tee -a "${WORK_ROOT}/launch.log"
echo "[heavyfix] $(date -Is) burn_in=${RSAR_BURN_IN_EPOCHS:-0} weight_u=${RSAR_WEIGHT_U:-1.0} lr=${RSAR_ADAPT_LR:-default} epochs=${RSAR_ADAPT_EPOCHS:-12}" | tee -a "${WORK_ROOT}/launch.log"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

for CORR in "${CORRS[@]}"; do
  echo "[heavyfix] $(date -Is) === corruption=${CORR} ===" | tee -a "${WORK_ROOT}/launch.log"

  # re-baseline direct_test (fast, sanity — must match 原 run)
  echo "[heavyfix] $(date -Is) step=direct_test corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=direct_test RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${SOURCE_CKPT}" \
      --work-dir "${WORK_ROOT}/${CORR}/direct_test" --eval mAP

  # adapt_cga with SARCLIP LoRA-enabled CGA
  echo "[heavyfix] $(date -Is) step=adapt_cga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=1 \
    ddp_run train.py "${CFG}" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_lora" \
      --teacher-ckpt "${SOURCE_CKPT}" \
      --cga-scorer sarclip --cga-templates 'A SAR image of a {}' --cga-tau 100 --cga-expand-ratio 0.4 \
      --no-validate

  echo "[heavyfix] $(date -Is) step=eval_cga corr=${CORR}" | tee -a "${WORK_ROOT}/launch.log"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${CORR}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${WORK_ROOT}/${CORR}/self_training_plus_cga_lora/latest_ema.pth" \
      --work-dir "${WORK_ROOT}/${CORR}/self_training_plus_cga_lora/eval_target" --eval mAP
done

echo "[heavyfix] $(date -Is) DONE work_root=${WORK_ROOT}" | tee -a "${WORK_ROOT}/launch.log"
