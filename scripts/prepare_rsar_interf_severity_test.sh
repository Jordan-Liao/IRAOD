#!/usr/bin/env bash
set -euo pipefail

# Generate RSAR interference "severity" suites on a chosen split (default: test only)
# to support robustness curves without duplicating full train/val data.
#
# Outputs (default):
#   dataset/RSAR/test/images-interf_jamA_s{1..5}/
#   dataset/RSAR/test/images-interf_jamB_s{1..5}/

DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
ENV_NAME="${ENV_NAME:-iraod}"
SPLITS="${SPLITS:-test}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --splits "${SPLITS}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --force-replace-symlink
  --diff-samples 128
)

if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

run_one() {
  local corrupt="$1"
  local itype="$2"
  local params_json="$3"
  echo "[prepare_rsar_interf_severity_test] === ${corrupt} type=${itype} splits=${SPLITS} ==="

  PYTHONUNBUFFERED=1 conda run -n "${ENV_NAME}" python -u tools/prepare_rsar_interference.py \
    --corrupt "${corrupt}" \
    --type "${itype}" \
    --params-json "${params_json}" \
    "${COMMON_ARGS[@]}"

  conda run -n "${ENV_NAME}" python tools/verify_rsar_corrupt_switch.py \
    --data-root "${DATA_ROOT}" \
    --corrupt "${corrupt}" \
    --splits "${SPLITS}"

  # Only run diff check on test split if requested; otherwise run on the first split.
  local diff_split="test"
  if [[ "${SPLITS}" != *"test"* ]]; then
    diff_split="${SPLITS%%,*}"
  fi
  conda run -n "${ENV_NAME}" python tools/verify_rsar_interference_diff.py \
    --data-root "${DATA_ROOT}" \
    --corrupt "${corrupt}" \
    --split "${diff_split}" \
    --samples 256 \
    --seed "${SEED}"
}

# ---- jamA severity: noise_jamming ----
# Keep stripeFreq fixed; vary jsRatio (dB) for a mild->strong curve.
JAMA_STRIPE_FREQ="0.01"
JAMA_STRIPE_AMP="20"
JAMA_JSR_LIST=(0 2 4 6 8)

for i in "${!JAMA_JSR_LIST[@]}"; do
  s="$((i + 1))"
  jsr="${JAMA_JSR_LIST[$i]}"
  run_one "interf_jamA_s${s}" "noise_jamming" "{\"jsRatio\":${jsr},\"stripeFreq\":${JAMA_STRIPE_FREQ},\"stripeAmplitude\":${JAMA_STRIPE_AMP}}"
done

# ---- jamB severity: noise_am_jamming ----
# Keep lineFrequency/baseIntensity/direction fixed; vary noiseSigma/lineWidth/blendFactor.
JAMB_LINE_FREQ="0.05"
JAMB_BASE_INT="150"
JAMB_DIR="vertical"
JAMB_NOISE_SIGMA=(50 80 120 160 200)
JAMB_LINE_WIDTH=(6 8 10 12 14)
JAMB_BLEND=(0.15 0.25 0.35 0.45 0.55)

for i in "${!JAMB_NOISE_SIGMA[@]}"; do
  s="$((i + 1))"
  ns="${JAMB_NOISE_SIGMA[$i]}"
  lw="${JAMB_LINE_WIDTH[$i]}"
  bf="${JAMB_BLEND[$i]}"
  run_one "interf_jamB_s${s}" "noise_am_jamming" "{\"lineFrequency\":${JAMB_LINE_FREQ},\"baseIntensity\":${JAMB_BASE_INT},\"noiseSigma\":${ns},\"lineWidth\":${lw},\"direction\":\"${JAMB_DIR}\",\"blendFactor\":${bf}}"
done

echo "[prepare_rsar_interf_severity_test] OK"
