#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dino_sar}"
CONFIG="${CONFIG:-configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining.py}"
DATA_ROOT="${DATA_ROOT:-dataset/DIOR}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_dior_ut}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/dior_ut}"

SMOKE="${SMOKE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
N="${N:-200}"

SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"

CORRUPTS="${CORRUPTS:-clean,cloudy,brightness,contrast}"

mkdir -p "${WORK_DIR}" "${SPLIT_DIR}"

TRAIN_LIST="${DATA_ROOT}/ImageSets/train.txt"
VAL_LIST="${DATA_ROOT}/ImageSets/val.txt"
TEST_LIST="${DATA_ROOT}/ImageSets/test.txt"
if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_LIST="${SPLIT_DIR}/train_smoke.txt"
  VAL_LIST="${SPLIT_DIR}/val_smoke.txt"
  TEST_LIST="${SPLIT_DIR}/test_smoke.txt"
  head -n "${N}" "${DATA_ROOT}/ImageSets/train.txt" > "${TRAIN_LIST}"
  head -n "${N}" "${DATA_ROOT}/ImageSets/val.txt" > "${VAL_LIST}"
  head -n "${N}" "${DATA_ROOT}/ImageSets/test.txt" > "${TEST_LIST}"
fi

ANN_SUBDIR="${DATA_ROOT}/Annotations/OrientedBoundingBoxes"
IMG_DIR="${DATA_ROOT}/JPEGImages"

echo "[exp_dior_ut] train corrupt=clean (ENV=${ENV_NAME}) ..."
conda run -n "${ENV_NAME}" python train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    corrupt=clean \
    runner.max_epochs="${MAX_EPOCHS}" \
    lr_config.step="[$((${MAX_EPOCHS}))]" \
    data.samples_per_gpu="${SAMPLES_PER_GPU}" \
    data.workers_per_gpu="${WORKERS_PER_GPU}" \
    data.train.ann_file="${TRAIN_LIST}" \
    data.train.ann_file_u="${VAL_LIST}" \
    data.val.ann_file="${TEST_LIST}" \
    data.test.ann_file="${TEST_LIST}" \
    data.train.img_prefix="${IMG_DIR}" \
    data.train.img_prefix_u="${DATA_ROOT}/Corruption/JPEGImages-clean" \
    data.val.img_prefix="${DATA_ROOT}/Corruption/JPEGImages-clean" \
    data.test.img_prefix="${DATA_ROOT}/Corruption/JPEGImages-clean" \
    data.train.ann_subdir="${ANN_SUBDIR}" \
    data.val.ann_subdir="${ANN_SUBDIR}" \
    data.test.ann_subdir="${ANN_SUBDIR}"

IFS=',' read -r -a CORRUPT_LIST <<< "${CORRUPTS}"
for c in "${CORRUPT_LIST[@]}"; do
  c="$(echo "${c}" | xargs)"
  [[ -z "${c}" ]] && continue
  EVAL_DIR="${WORK_DIR}/eval_${c}"
  VIS_DIR="${WORK_DIR}/vis_${c}"
  mkdir -p "${EVAL_DIR}" "${VIS_DIR}"

  echo "[exp_dior_ut] test corrupt=${c} (ENV=${ENV_NAME}) ..."
  conda run -n "${ENV_NAME}" python test.py "${CONFIG}" "${WORK_DIR}/latest.pth" \
    --eval mAP \
    --work-dir "${EVAL_DIR}" \
    --show-dir "${VIS_DIR}" \
    --cfg-options \
      corrupt="${c}" \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
      data.test.ann_file="${TEST_LIST}" \
      data.test.img_prefix="${DATA_ROOT}/Corruption/JPEGImages-${c}" \
      data.test.ann_subdir="${ANN_SUBDIR}"
done

echo "[exp_dior_ut] done: ${WORK_DIR}"
