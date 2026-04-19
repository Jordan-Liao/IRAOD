#!/usr/bin/env bash
# Re-eval all existing RSAR SFOD-RS ckpts with TTA (multi-scale flip).
# Saves to <FIXED_WR>/tta_eval/<corr>/<stage>/ to avoid overwriting original evals.
set -Eeuo pipefail

FIXED_WR="${1:?usage: $0 <fixed_work_root> [heavyfix_work_root]}"
HEAVYFIX_WR="${2:-}"
SOURCE_CKPT="${SOURCE_CKPT:-work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth}"

NGPUS="${NGPUS:-3}"
MASTER_PORT="${MASTER_PORT:-29506}"
PYTHON_BIN="${PYTHON:-/home/zechuan/anaconda3/envs/iraod/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export RSAR_USE_TTA=1

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py
CORRS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)
HEAVY=(noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)
OUT="${FIXED_WR}/tta_eval"
mkdir -p "${OUT}"
LOG="${OUT}/driver.log"
echo "[tta_eval] $(date -Is) fixed_wr=${FIXED_WR} heavyfix_wr=${HEAVYFIX_WR:-none} out=${OUT}" | tee -a "${LOG}"

ddp_run() {
  local script="$1"; shift
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}" --use_env \
    "${script}" "$@" --launcher pytorch
}

run_eval() {
  local corr="$1" stage="$2" ckpt="$3" tag="$4"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[tta_eval] $(date -Is) SKIP corr=${corr} tag=${tag} (ckpt missing: ${ckpt})" | tee -a "${LOG}"; return 0
  fi
  local wd="${OUT}/${corr}/${tag}"
  if ls "${wd}"/eval_*.json >/dev/null 2>&1; then
    echo "[tta_eval] $(date -Is) SKIP corr=${corr} tag=${tag} (already evaluated)" | tee -a "${LOG}"; return 0
  fi
  echo "[tta_eval] $(date -Is) RUN corr=${corr} tag=${tag} ckpt=${ckpt}" | tee -a "${LOG}"
  RSAR_STAGE="${stage}" RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 \
    ddp_run test.py "${CFG}" "${ckpt}" --work-dir "${wd}" --eval mAP
}

# clean source_clean_test (special)
run_eval clean source_clean_test "${SOURCE_CKPT}" source_clean_test

for CORR in "${CORRS[@]}"; do
  run_eval "${CORR}" direct_test "${SOURCE_CKPT}" direct_test
  run_eval "${CORR}" target_eval "${FIXED_WR}/${CORR}/self_training/latest_ema.pth" self_training
  run_eval "${CORR}" target_eval "${FIXED_WR}/${CORR}/self_training_plus_cga/latest_ema.pth" self_training_plus_cga
done

if [[ -n "${HEAVYFIX_WR}" && -d "${HEAVYFIX_WR}" ]]; then
  for CORR in "${HEAVY[@]}"; do
    run_eval "${CORR}" target_eval "${HEAVYFIX_WR}/${CORR}/self_training_plus_cga_lora/latest_ema.pth" self_training_plus_cga_lora
  done
fi

echo "[tta_eval] $(date -Is) DONE" | tee -a "${LOG}"
