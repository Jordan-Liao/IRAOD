#!/usr/bin/env bash
set -euo pipefail

# Refresh lightweight result summaries from `work_dirs/` artifacts:
# - work_dirs/results/metrics.csv
# - work_dirs/results/ablation_table.md
# - experiments.csv
#
# This script is intentionally lightweight and does NOT run training/eval.

ENV_NAME="${ENV_NAME:-dino_sar}"

METRICS_CSV="${METRICS_CSV:-work_dirs/results/metrics.csv}"
ABLATION_MD="${ABLATION_MD:-work_dirs/results/ablation_table.md}"
EXPERIMENTS_CSV="${EXPERIMENTS_CSV:-experiments.csv}"

# Curated work-dir globs:
# - include main experiments
# - exclude `work_dirs/exp_rsar_severity` (it's an evaluation suite, not a training experiment)
WORK_DIR_GLOBS=(
  work_dirs/exp_smoke_*
  work_dirs/exp_dior_*
  work_dirs/exp_rsar_baseline*
  work_dirs/exp_rsar_ut*
)

echo "[refresh_results] env=${ENV_NAME}"
echo "[refresh_results] metrics=${METRICS_CSV}"
echo "[refresh_results] ablation=${ABLATION_MD}"
echo "[refresh_results] experiments=${EXPERIMENTS_CSV}"

conda run -n "${ENV_NAME}" python tools/export_metrics.py \
  --work-dirs "${WORK_DIR_GLOBS[@]}" \
  --out-csv "${METRICS_CSV}"

conda run -n "${ENV_NAME}" python tools/ablation_table.py \
  --csv "${METRICS_CSV}" \
  --out-md "${ABLATION_MD}"

conda run -n "${ENV_NAME}" python tools/export_experiments.py \
  --metrics-csv "${METRICS_CSV}" \
  --out-csv "${EXPERIMENTS_CSV}"

if [[ "${PLOTS:-0}" == "1" ]]; then
  echo "[refresh_results] PLOTS=1, generating plots under work_dirs/results/plots ..."
  conda run -n "${ENV_NAME}" python tools/plot_all.py \
    --metrics-csv "${METRICS_CSV}" \
    --log-json-glob "work_dirs/exp_*/*.log.json" \
    --out-dir "work_dirs/results/plots"
fi

echo "[refresh_results] DONE"
