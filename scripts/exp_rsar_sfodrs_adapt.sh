#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <corruption> <source_ckpt> [config] [work_root]" >&2
  exit 2
fi

echo "[deprecated] scripts/exp_rsar_sfodrs_adapt.sh -> scripts/run/rsar_sfodrs_domain.sh" >&2

CORR="${CORR:-$1}"
SOURCE_CKPT="${SOURCE_CKPT:-$2}"
CONFIG="${CONFIG:-${TRAIN_CONFIG:-${3:-configs/current/rsar_sfodrs.py}}}"
WORK_ROOT="${WORK_ROOT:-${4:-work_dirs/rsar_sfodrs}}"

CORR="${CORR}" SOURCE_CKPT="${SOURCE_CKPT}" CONFIG="${CONFIG}" WORK_ROOT="${WORK_ROOT}" \
  bash scripts/run/rsar_sfodrs_domain.sh
