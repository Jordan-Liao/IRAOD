#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/SSD1_8TB/zechuan/IRAOD

MAIN_ROOT="${1:-work_dirs/rsar_sfodrs_full_fix_20260424_172627}"
SHARD_ROOT="${2:-work_dirs/rsar_sfodrs_shard_late3_ddp_20260426_001855}"

PYTHON_BIN="${PYTHON_BIN:-/home/zechuan/anaconda3/envs/iraod/bin/python}"
CFG="${CFG:-configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py}"
SOURCE_CKPT="${SOURCE_CKPT:-${MAIN_ROOT}/source_train/latest.pth}"
MASTER_PORT="${MASTER_PORT:-29643}"
LOG="${MAIN_ROOT}/cutover_resume.log"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

log() {
  echo "[cutover-resume] $(date -Iseconds) $*" | tee -a "${LOG}"
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

log start
log main_root="${MAIN_ROOT}"
log shard_root="${SHARD_ROOT}"

for corr in am_noise_horizontal smart_suppression am_noise_vertical; do
  mkdir -p "${MAIN_ROOT}/${corr}"

  if [[ -d "${SHARD_ROOT}/${corr}/direct_test" ]]; then
    mkdir -p "${MAIN_ROOT}/${corr}"
    cp -a "${SHARD_ROOT}/${corr}/direct_test" "${MAIN_ROOT}/${corr}/"
  fi

  if [[ -d "${SHARD_ROOT}/${corr}/self_training/eval_target" ]]; then
    mkdir -p "${MAIN_ROOT}/${corr}/self_training"
    cp -a "${SHARD_ROOT}/${corr}/self_training/eval_target" "${MAIN_ROOT}/${corr}/self_training/"
  fi

  if [[ -d "${SHARD_ROOT}/${corr}/self_training_plus_cga/eval_target" ]]; then
    mkdir -p "${MAIN_ROOT}/${corr}/self_training_plus_cga"
    cp -a "${SHARD_ROOT}/${corr}/self_training_plus_cga/eval_target" "${MAIN_ROOT}/${corr}/self_training_plus_cga/"
  fi

  log synced_eval_artifacts corr="${corr}"
done

ts="$(date +%Y%m%d_%H%M%S)"
for corr in point_target noise_suppression; do
  if [[ -d "${MAIN_ROOT}/${corr}" ]]; then
    mv "${MAIN_ROOT}/${corr}" "${MAIN_ROOT}/${corr}_aborted_${ts}"
    log moved_partial corr="${corr}"
  fi
done

for corr in point_target noise_suppression; do
  log start_corr corr="${corr}"

  log step=direct_test corr="${corr}"
  RSAR_STAGE=direct_test RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 CUDA_VISIBLE_DEVICES=1,2,3 \
    ddp_run test.py "${CFG}" "${SOURCE_CKPT}" \
      --work-dir "${MAIN_ROOT}/${corr}/direct_test" --eval mAP

  log step=adapt_nocga corr="${corr}"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 CUDA_VISIBLE_DEVICES=1,2,3 \
    ddp_run train.py "${CFG}" \
      --work-dir "${MAIN_ROOT}/${corr}/self_training" \
      --teacher-ckpt "${SOURCE_CKPT}" --no-validate

  log step=eval_nocga corr="${corr}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 CUDA_VISIBLE_DEVICES=1,2,3 \
    ddp_run test.py "${CFG}" "${MAIN_ROOT}/${corr}/self_training/latest_ema.pth" \
      --work-dir "${MAIN_ROOT}/${corr}/self_training/eval_target" --eval mAP

  log step=adapt_cga corr="${corr}"
  RSAR_STAGE=target_adapt RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=1 CUDA_VISIBLE_DEVICES=1,2,3 \
    ddp_run train.py "${CFG}" \
      --work-dir "${MAIN_ROOT}/${corr}/self_training_plus_cga" \
      --teacher-ckpt "${SOURCE_CKPT}" \
      --cga-scorer sarclip --cga-templates "A SAR image of a {}" \
      --cga-tau 100 --cga-expand-ratio 0.4 --no-validate

  log step=eval_cga corr="${corr}"
  RSAR_STAGE=target_eval RSAR_TARGET_DOMAIN="${corr}" RSAR_USE_CGA=0 CUDA_VISIBLE_DEVICES=1,2,3 \
    ddp_run test.py "${CFG}" "${MAIN_ROOT}/${corr}/self_training_plus_cga/latest_ema.pth" \
      --work-dir "${MAIN_ROOT}/${corr}/self_training_plus_cga/eval_target" --eval mAP
done

log step=collect_results
"${PYTHON_BIN}" -u tools/collect_rsar_sfodrs_results.py \
  --work-root "${MAIN_ROOT}" \
  --out-csv "${MAIN_ROOT}/rsar_sfodrs_results.csv" \
  --out-md "${MAIN_ROOT}/rsar_sfodrs_results.md"

log done
