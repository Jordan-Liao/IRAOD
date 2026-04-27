#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

TENT_WR="${1:-${WORK_ROOT:-}}"
if [[ -z "${TENT_WR}" ]]; then
  echo "Usage: $0 <tent_work_root>" >&2
  exit 2
fi

echo "[deprecated] scripts/run_rsar_sfodrs_tent_eval.sh -> scripts/run/rsar_tent.sh" >&2

TENT_MODE=eval WORK_ROOT="${TENT_WR}" bash scripts/run/rsar_tent.sh
