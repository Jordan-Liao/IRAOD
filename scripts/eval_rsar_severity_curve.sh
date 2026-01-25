#!/usr/bin/env bash
set -euo pipefail

# Evaluate a checkpoint on RSAR interference "severity" suites and write a CSV summary.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 \
#   CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth \
#   CORRUPT_BASE=interf_jamA \
#   N_TEST=1000 \
#   bash scripts/eval_rsar_severity_curve.sh
#
# It expects test-only severity dirs like:
#   dataset/RSAR/test/images-interf_jamA_s1 ... s5
#
# Outputs:
#   work_dirs/exp_rsar_severity/<tag>/<corrupt_base>/severity_summary.csv
#   work_dirs/exp_rsar_severity/<tag>/<corrupt_base>/<corrupt>/eval_*.json

ENV_NAME="${ENV_NAME:-dino_sar}"
CONFIG="${CONFIG:-configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py}"
DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "[eval_rsar_severity_curve] ERROR: CKPT is required" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_rsar_severity_curve] ERROR: ckpt not found: ${CKPT}" >&2
  exit 2
fi

CORRUPT_BASE="${CORRUPT_BASE:-}"
if [[ -z "${CORRUPT_BASE}" ]]; then
  echo "[eval_rsar_severity_curve] ERROR: CORRUPT_BASE is required (e.g. interf_jamA|interf_jamB)" >&2
  exit 2
fi

SEVERITIES="${SEVERITIES:-1,2,3,4,5}"
INCLUDE_CLEAN="${INCLUDE_CLEAN:-1}"

N_TEST="${N_TEST:-1000}"
N_TRAIN="${N_TRAIN:-50}"
N_VAL="${N_VAL:-50}"
SMOKE="${SMOKE:-1}"

SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

# CGA is not used in mmrotate test-time by default; keep it disabled unless you explicitly enable it.
export CGA_SCORER="${CGA_SCORER:-none}"

TAG="${TAG:-$(basename "$(dirname "${CKPT}")")}"
OUT_ROOT="${OUT_ROOT:-work_dirs/exp_rsar_severity/${TAG}/${CORRUPT_BASE}}"

mkdir -p "${OUT_ROOT}"
SUMMARY_CSV="${OUT_ROOT}/severity_summary.csv"

echo "corrupt,mAP,eval_json" > "${SUMMARY_CSV}"

split_dir_base="${OUT_ROOT}/_splits"

run_eval() {
  local corrupt="$1"

  # Ensure the test corrupt directory exists, otherwise exp_rsar_ut.sh may silently fall back to clean.
  if [[ "${corrupt}" != "clean" ]]; then
    local expect_dir="${DATA_ROOT}/test/images-${corrupt}"
    if [[ ! -d "${expect_dir}" ]]; then
      echo "[eval_rsar_severity_curve] ERROR: missing test corrupt dir: ${expect_dir}" >&2
      exit 3
    fi
  fi

  local work_dir="${OUT_ROOT}/${corrupt}"
  local vis_dir="${OUT_ROOT}/${corrupt}/vis"
  local split_dir="${split_dir_base}/${corrupt}_n${N_TEST}"
  mkdir -p "${work_dir}" "${vis_dir}" "${split_dir}"

  echo "[eval_rsar_severity_curve] eval corrupt=${corrupt} n_test=${N_TEST} ckpt=${CKPT}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  CGA_SCORER="${CGA_SCORER}" \
  CORRUPT="${corrupt}" \
  DO_TRAIN=0 DO_TEST=1 \
  SMOKE="${SMOKE}" \
  N_TRAIN="${N_TRAIN}" N_VAL="${N_VAL}" N_TEST="${N_TEST}" \
  SAMPLES_PER_GPU="${SAMPLES_PER_GPU}" WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
  CONFIG="${CONFIG}" DATA_ROOT="${DATA_ROOT}" \
  WORK_DIR="${work_dir}" VIS_DIR="${vis_dir}" SPLIT_DIR="${split_dir}" \
  CKPT="${CKPT}" \
  bash scripts/exp_rsar_ut.sh

  WORK_DIR="${work_dir}" CORRUPT="${corrupt}" SUMMARY_CSV="${SUMMARY_CSV}" python - <<'PY'
import json
import os
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"]).resolve()
corrupt = os.environ["CORRUPT"]
summary_csv = Path(os.environ["SUMMARY_CSV"]).resolve()

eval_files = sorted(work_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
if not eval_files:
    raise SystemExit(f"no eval_*.json under {work_dir}")
latest = eval_files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
mp = float(data.get("metric", {}).get("mAP", float("nan")))
with summary_csv.open("a", encoding="utf-8") as f:
    f.write(f"{corrupt},{mp},{latest}\n")
print(f"[eval_rsar_severity_curve] {corrupt} mAP={mp:.6f} eval={latest}")
PY
}

if [[ "${INCLUDE_CLEAN}" == "1" ]]; then
  run_eval "clean"
fi

IFS=',' read -r -a sevs <<< "${SEVERITIES}"
for s in "${sevs[@]}"; do
  s="$(echo "${s}" | xargs)"
  if [[ -z "${s}" ]]; then
    continue
  fi
  run_eval "${CORRUPT_BASE}_s${s}"
done

echo "[eval_rsar_severity_curve] DONE summary=${SUMMARY_CSV}"
