#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dino_sar}"
CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py"
DATA_ROOT="${DATA_ROOT:-dataset/DIOR}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_smoke_dior}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/dior}"
N="${N:-50}"

mkdir -p "${SPLIT_DIR}"
mkdir -p "${WORK_DIR}"

head -n "${N}" "${DATA_ROOT}/ImageSets/train.txt" > "${SPLIT_DIR}/train_smoke.txt"
head -n "${N}" "${DATA_ROOT}/ImageSets/val.txt" > "${SPLIT_DIR}/val_smoke.txt"
head -n "${N}" "${DATA_ROOT}/ImageSets/test.txt" > "${SPLIT_DIR}/test_smoke.txt"

ANN_SUBDIR="${DATA_ROOT}/Annotations/OrientedBoundingBoxes"
IMG_DIR="${DATA_ROOT}/JPEGImages"

echo "[smoke_dior] train (ENV=${ENV_NAME}) ..."
conda run -n "${ENV_NAME}" python train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    corrupt=clean \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=0 \
    data.train.ann_file="${SPLIT_DIR}/train_smoke.txt" \
    data.train.ann_file_u="${SPLIT_DIR}/val_smoke.txt" \
    data.val.ann_file="${SPLIT_DIR}/test_smoke.txt" \
    data.test.ann_file="${SPLIT_DIR}/test_smoke.txt" \
    data.train.img_prefix="${IMG_DIR}" \
    data.train.img_prefix_u="${IMG_DIR}" \
    data.val.img_prefix="${IMG_DIR}" \
    data.test.img_prefix="${IMG_DIR}" \
    data.train.ann_subdir="${ANN_SUBDIR}" \
    data.val.ann_subdir="${ANN_SUBDIR}" \
    data.test.ann_subdir="${ANN_SUBDIR}"

echo "[smoke_dior] test (ENV=${ENV_NAME}) ..."
conda run -n "${ENV_NAME}" python test.py \
  "${CONFIG}" \
  "${WORK_DIR}/latest.pth" \
  --eval mAP \
  --cfg-options \
    corrupt=clean \
    data.test.ann_file="${SPLIT_DIR}/test_smoke.txt" \
    data.test.img_prefix="${IMG_DIR}" \
    data.test.ann_subdir="${ANN_SUBDIR}"

echo "[smoke_dior] done: ${WORK_DIR}"
