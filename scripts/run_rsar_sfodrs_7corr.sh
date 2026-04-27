#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <source_ckpt|auto|none> [train_config|work_root] [work_root]" >&2
  exit 2
fi

echo "[deprecated] scripts/run_rsar_sfodrs_7corr.sh -> scripts/run/rsar_sfodrs_full.sh" >&2

SOURCE_CKPT="${SOURCE_CKPT:-$1}"
CONFIG="${CONFIG:-${TRAIN_CONFIG:-}}"
WORK_ROOT="${WORK_ROOT:-}"

if [[ $# -ge 2 ]]; then
  if [[ "$2" == *.py ]]; then
    CONFIG="$2"
    WORK_ROOT="${3:-${WORK_ROOT:-work_dirs/rsar_sfodrs}}"
  else
    WORK_ROOT="$2"
  fi
fi

CONFIG="${CONFIG:-configs/current/rsar_sfodrs.py}"
WORK_ROOT="${WORK_ROOT:-work_dirs/rsar_sfodrs}"
NGPUS="${NGPUS:-1}"

SOURCE_CKPT="${SOURCE_CKPT}" CONFIG="${CONFIG}" WORK_ROOT="${WORK_ROOT}" NGPUS="${NGPUS}" \
  bash scripts/run/rsar_sfodrs_full.sh
