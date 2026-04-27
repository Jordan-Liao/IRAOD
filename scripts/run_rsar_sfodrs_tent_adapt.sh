#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

if [[ $# -lt 1 && -z "${SOURCE_CKPT:-}" ]]; then
  echo "Usage: $0 <source_ckpt> [work_root]" >&2
  exit 2
fi

echo "[deprecated] scripts/run_rsar_sfodrs_tent_adapt.sh -> scripts/run/rsar_tent.sh" >&2

SOURCE_CKPT="${SOURCE_CKPT:-${1:-}}"
WORK_ROOT="${WORK_ROOT:-${2:-}}"

if [[ -n "${WORK_ROOT}" ]]; then
  TENT_MODE=adapt SOURCE_CKPT="${SOURCE_CKPT}" WORK_ROOT="${WORK_ROOT}" bash scripts/run/rsar_tent.sh
else
  TENT_MODE=adapt SOURCE_CKPT="${SOURCE_CKPT}" bash scripts/run/rsar_tent.sh
fi
