#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source_ckpt> <direct_predictions_work_root> <tent_work_root> [out_root]" >&2
  exit 2
fi

echo "[deprecated] scripts/run_rsar_sfodrs_tent_ensemble.sh -> scripts/run/rsar_tent.sh" >&2

SOURCE_CKPT="${SOURCE_CKPT:-$1}"
DIRECT_WORK_ROOT="${DIRECT_WORK_ROOT:-$2}"
WORK_ROOT="${WORK_ROOT:-$3}"
OUT_ROOT="${OUT_ROOT:-${4:-}}"

if [[ -n "${OUT_ROOT}" ]]; then
  TENT_MODE=ensemble SOURCE_CKPT="${SOURCE_CKPT}" DIRECT_WORK_ROOT="${DIRECT_WORK_ROOT}" WORK_ROOT="${WORK_ROOT}" OUT_ROOT="${OUT_ROOT}" \
    bash scripts/run/rsar_tent.sh
else
  TENT_MODE=ensemble SOURCE_CKPT="${SOURCE_CKPT}" DIRECT_WORK_ROOT="${DIRECT_WORK_ROOT}" WORK_ROOT="${WORK_ROOT}" \
    bash scripts/run/rsar_tent.sh
fi
