#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-iraod}"
CONFIG="${CONFIG:-configs/experiments/dior/baseline_oriented_rcnn_dior.py}"
CKPT="${CKPT:-baseline/baseline.pth}"
DATA_ROOT="${DATA_ROOT:-dataset/DIOR}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_dior_baseline_eval}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/dior_baseline_eval}"

SMOKE="${SMOKE:-1}"
N_TEST="${N_TEST:-200}"

SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"

CORRUPTS="${CORRUPTS:-clean,cloudy,brightness,contrast}"

mkdir -p "${WORK_DIR}" "${SPLIT_DIR}"

TEST_LIST="${DATA_ROOT}/ImageSets/test.txt"
if [[ "${SMOKE}" == "1" ]]; then
  TEST_LIST="${SPLIT_DIR}/test_smoke.txt"
  head -n "${N_TEST}" "${DATA_ROOT}/ImageSets/test.txt" > "${TEST_LIST}"
fi

ANN_SUBDIR="${DATA_ROOT}/Annotations/OrientedBoundingBoxes"

IFS=',' read -r -a CORRUPT_LIST <<< "${CORRUPTS}"
for c in "${CORRUPT_LIST[@]}"; do
  c="$(echo "${c}" | xargs)"
  [[ -z "${c}" ]] && continue
  EVAL_DIR="${WORK_DIR}/eval_${c}"
  VIS_DIR="${WORK_DIR}/vis_${c}"
  mkdir -p "${EVAL_DIR}" "${VIS_DIR}"

  echo "[exp_dior_baseline_eval] test corrupt=${c} (ENV=${ENV_NAME}) ..."
  conda run -n "${ENV_NAME}" python test.py "${CONFIG}" "${CKPT}" \
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

echo "[exp_dior_baseline_eval] done: ${WORK_DIR}"
