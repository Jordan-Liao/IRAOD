#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

echo "[deprecated] scripts/run_rsar_sfodrs_full_3gpu.sh -> scripts/run/rsar_sfodrs_full.sh" >&2

SOURCE_CKPT="${SOURCE_CKPT:-${1:-auto}}"
WORK_ROOT="${WORK_ROOT:-${2:-}}"

if [[ -n "${WORK_ROOT}" ]]; then
  SOURCE_CKPT="${SOURCE_CKPT}" WORK_ROOT="${WORK_ROOT}" bash scripts/run/rsar_sfodrs_full.sh
else
  SOURCE_CKPT="${SOURCE_CKPT}" bash scripts/run/rsar_sfodrs_full.sh
fi
