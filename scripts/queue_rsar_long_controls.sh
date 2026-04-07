#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source ~/anaconda3/etc/profile.d/conda.sh

ENV_NAME="${ENV_NAME:-iraod}"
FINAL_ROOT="${FINAL_ROOT:-work_dirs/controls/rsar_clip_guided_sfod}"
PARTIAL_ROOT="${PARTIAL_ROOT:-work_dirs/controls_smoke/direct_bn}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"

SOURCE_CONFIG="${SOURCE_CONFIG:-configs/experiments/rsar/frontier_026_ocafpn_24ep_oriented_rcnn_rsar.py}"
SOURCE_CKPT="${SOURCE_CKPT:-work_dirs/frontier_026_ocafpn_24ep/latest.pth}"

SELFTRAIN_GPUS="${SELFTRAIN_GPUS:-2,3,4,5,6}"
CGA_GPUS="${CGA_GPUS:-2,3,4,5,6}"

wait_until_gone() {
  local pattern="$1"
  while pgrep -af "$pattern" >/dev/null; do
    sleep "${WAIT_INTERVAL}"
  done
}

copy_partial_method() {
  local method="$1"
  if [[ -d "${PARTIAL_ROOT}/${method}" ]]; then
    mkdir -p "${FINAL_ROOT}/${method}"
    rsync -az "${PARTIAL_ROOT}/${method}/" "${FINAL_ROOT}/${method}/"
  fi
}

mkdir -p "${FINAL_ROOT}"

echo "[queue_rsar_long_controls] FINAL_ROOT=${FINAL_ROOT}"
echo "[queue_rsar_long_controls] PARTIAL_ROOT=${PARTIAL_ROOT}"
echo "[queue_rsar_long_controls] SOURCE_CONFIG=${SOURCE_CONFIG}"
echo "[queue_rsar_long_controls] SOURCE_CKPT=${SOURCE_CKPT}"

wait_until_gone "tools/run_direct_test.py|tools/run_bn_calibration.py"
copy_partial_method clean
copy_partial_method direct
copy_partial_method bn
conda run -n "${ENV_NAME}" python tools/collect_controls_results.py --result-root "${FINAL_ROOT}"

wait_until_gone "tools/run_tent_adapt.py|tools/run_shot_adapt.py"

RUN_DIRECT=0 \
RUN_BN=0 \
RUN_TENT=0 \
RUN_SHOT=0 \
RUN_SELFTRAIN=1 \
RUN_CGA=1 \
RESULT_ROOT="${FINAL_ROOT}" \
SOURCE_CONFIG="${SOURCE_CONFIG}" \
SOURCE_CKPT="${SOURCE_CKPT}" \
SELFTRAIN_GPUS="${SELFTRAIN_GPUS}" \
CGA_GPUS="${CGA_GPUS}" \
scripts/exp_rsar_controls.sh
