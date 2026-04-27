#!/usr/bin/env bash
# Collect RSAR SFOD-RS eval JSON files into CSV/Markdown summaries.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/iraod.sh"
iraod_cd_root

WORK_ROOT="${WORK_ROOT:-${1:-work_dirs/rsar_sfodrs}}"
OUT_CSV="${OUT_CSV:-${WORK_ROOT}/rsar_sfodrs_results.csv}"
OUT_MD="${OUT_MD:-${WORK_ROOT}/rsar_sfodrs_results.md}"

iraod_log "collect" "work_root=${WORK_ROOT} out_csv=${OUT_CSV} out_md=${OUT_MD}"
iraod_run "${PYTHON_BIN}" -u tools/collect_rsar_sfodrs_results.py \
  --work-root "${WORK_ROOT}" \
  --out-csv "${OUT_CSV}" \
  --out-md "${OUT_MD}"
