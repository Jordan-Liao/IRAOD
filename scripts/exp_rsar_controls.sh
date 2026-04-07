#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-iraod}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${RSAR_DATA_ROOT:-${REPO_ROOT}/dataset/RSAR}}"
RESULT_ROOT="${RESULT_ROOT:-work_dirs/controls/rsar_clip_guided_sfod}"

SOURCE_CONFIG="${SOURCE_CONFIG:-configs/experiments/rsar/frontier_026_ocafpn_24ep_oriented_rcnn_rsar.py}"
SOURCE_CKPT="${SOURCE_CKPT:-work_dirs/frontier_026_ocafpn_24ep/latest.pth}"

SEED="${SEED:-3407}"

RUN_DIRECT="${RUN_DIRECT:-1}"
RUN_BN="${RUN_BN:-1}"
RUN_TENT="${RUN_TENT:-1}"
RUN_SHOT="${RUN_SHOT:-1}"
RUN_SELFTRAIN="${RUN_SELFTRAIN:-1}"
RUN_CGA="${RUN_CGA:-1}"

DIRECT_GPU="${DIRECT_GPU:-1}"
BN_GPU="${BN_GPU:-1}"
TENT_GPU="${TENT_GPU:-1}"
SHOT_GPU="${SHOT_GPU:-1}"
SELFTRAIN_GPUS="${SELFTRAIN_GPUS:-1,2,3,4,5}"
CGA_GPUS="${CGA_GPUS:-1,2,3,4,5}"

BN_SAMPLES_PER_GPU="${BN_SAMPLES_PER_GPU:-16}"
BN_WORKERS_PER_GPU="${BN_WORKERS_PER_GPU:-4}"

TENT_SAMPLES_PER_GPU="${TENT_SAMPLES_PER_GPU:-4}"
TENT_WORKERS_PER_GPU="${TENT_WORKERS_PER_GPU:-2}"
TENT_EPOCHS="${TENT_EPOCHS:-1}"
TENT_LR="${TENT_LR:-1e-4}"
TENT_TOPK="${TENT_TOPK:-256}"
TENT_MIN_FG_CONF="${TENT_MIN_FG_CONF:-0.05}"

SHOT_SAMPLES_PER_GPU="${SHOT_SAMPLES_PER_GPU:-4}"
SHOT_WORKERS_PER_GPU="${SHOT_WORKERS_PER_GPU:-2}"
SHOT_EPOCHS="${SHOT_EPOCHS:-3}"
SHOT_LR="${SHOT_LR:-1e-4}"
SHOT_TOPK="${SHOT_TOPK:-256}"
SHOT_MIN_FG_CONF="${SHOT_MIN_FG_CONF:-0.05}"

SELFTRAIN_SAMPLES_PER_GPU="${SELFTRAIN_SAMPLES_PER_GPU:-8}"
SELFTRAIN_WORKERS_PER_GPU="${SELFTRAIN_WORKERS_PER_GPU:-4}"
SELFTRAIN_MAX_EPOCHS="${SELFTRAIN_MAX_EPOCHS:-24}"
SELFTRAIN_LR="${SELFTRAIN_LR:-0.02}"
SELFTRAIN_WEIGHT_U="${SELFTRAIN_WEIGHT_U:-0.5}"
SELFTRAIN_TAU="${SELFTRAIN_TAU:-0.5}"
SELFTRAIN_EMA_MOMENTUM="${SELFTRAIN_EMA_MOMENTUM:-0.998}"
SELFTRAIN_MASTER_PORT="${SELFTRAIN_MASTER_PORT:-29621}"

CGA_SAMPLES_PER_GPU="${CGA_SAMPLES_PER_GPU:-8}"
CGA_WORKERS_PER_GPU="${CGA_WORKERS_PER_GPU:-4}"
CGA_MAX_EPOCHS="${CGA_MAX_EPOCHS:-24}"
CGA_LR="${CGA_LR:-0.02}"
CGA_WEIGHT_U="${CGA_WEIGHT_U:-0.5}"
CGA_TAU="${CGA_TAU:-0.5}"
CGA_EMA_MOMENTUM="${CGA_EMA_MOMENTUM:-0.998}"
CGA_MASTER_PORT="${CGA_MASTER_PORT:-29622}"
CGA_LAMBDA="${CGA_LAMBDA:-0.2}"
STRICT_PAPER_PROMPT="${STRICT_PAPER_PROMPT:-0}"
SARCLIP_MODEL="${SARCLIP_MODEL:-ViT-L-14}"
SARCLIP_PRETRAINED="${SARCLIP_PRETRAINED:-weights/sarclip/ViT-L-14/vit_l_14_model.safetensors}"
SARCLIP_LORA="${SARCLIP_LORA:-lora_finetune/SARCLIP_LoRA_Interference.pt}"
CGA_TEMPLATES="${CGA_TEMPLATES:-}"

mkdir -p "${RESULT_ROOT}"

echo "[exp_rsar_controls] DATA_ROOT=${DATA_ROOT}"
echo "[exp_rsar_controls] RESULT_ROOT=${RESULT_ROOT}"
echo "[exp_rsar_controls] SOURCE_CONFIG=${SOURCE_CONFIG}"
echo "[exp_rsar_controls] SOURCE_CKPT=${SOURCE_CKPT}"

if [[ "${RUN_DIRECT}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${DIRECT_GPU}" conda run -n "${ENV_NAME}" python tools/run_direct_test.py \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --seed "${SEED}"
fi

if [[ "${RUN_BN}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${BN_GPU}" conda run -n "${ENV_NAME}" python tools/run_bn_calibration.py \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --samples-per-gpu "${BN_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${BN_WORKERS_PER_GPU}" \
    --seed "${SEED}"
fi

if [[ "${RUN_TENT}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${TENT_GPU}" conda run -n "${ENV_NAME}" python tools/run_tent_adapt.py \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --samples-per-gpu "${TENT_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${TENT_WORKERS_PER_GPU}" \
    --epochs "${TENT_EPOCHS}" \
    --lr "${TENT_LR}" \
    --topk "${TENT_TOPK}" \
    --min-fg-conf "${TENT_MIN_FG_CONF}" \
    --seed "${SEED}"
fi

if [[ "${RUN_SHOT}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${SHOT_GPU}" conda run -n "${ENV_NAME}" python tools/run_shot_adapt.py \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --samples-per-gpu "${SHOT_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${SHOT_WORKERS_PER_GPU}" \
    --epochs "${SHOT_EPOCHS}" \
    --lr "${SHOT_LR}" \
    --topk "${SHOT_TOPK}" \
    --min-fg-conf "${SHOT_MIN_FG_CONF}" \
    --seed "${SEED}"
fi

if [[ "${RUN_SELFTRAIN}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python tools/run_selftrain_adapt.py \
    --method selftrain \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --samples-per-gpu "${SELFTRAIN_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${SELFTRAIN_WORKERS_PER_GPU}" \
    --max-epochs "${SELFTRAIN_MAX_EPOCHS}" \
    --lr "${SELFTRAIN_LR}" \
    --weight-u "${SELFTRAIN_WEIGHT_U}" \
    --tau "${SELFTRAIN_TAU}" \
    --ema-momentum "${SELFTRAIN_EMA_MOMENTUM}" \
    --cuda-visible-devices "${SELFTRAIN_GPUS}" \
    --master-port "${SELFTRAIN_MASTER_PORT}" \
    --seed "${SEED}"
fi

if [[ "${RUN_CGA}" == "1" ]]; then
  CGA_ARGS=()
  if [[ -n "${CGA_TEMPLATES}" ]]; then
    CGA_ARGS+=(--cga-templates "${CGA_TEMPLATES}")
  fi
  if [[ "${STRICT_PAPER_PROMPT}" == "1" ]]; then
    CGA_ARGS+=(--strict-paper-prompt)
  fi

  conda run -n "${ENV_NAME}" python tools/run_selftrain_adapt.py \
    --method cga \
    --source-config "${SOURCE_CONFIG}" \
    --source-ckpt "${SOURCE_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --samples-per-gpu "${CGA_SAMPLES_PER_GPU}" \
    --workers-per-gpu "${CGA_WORKERS_PER_GPU}" \
    --max-epochs "${CGA_MAX_EPOCHS}" \
    --lr "${CGA_LR}" \
    --weight-u "${CGA_WEIGHT_U}" \
    --tau "${CGA_TAU}" \
    --ema-momentum "${CGA_EMA_MOMENTUM}" \
    --cuda-visible-devices "${CGA_GPUS}" \
    --master-port "${CGA_MASTER_PORT}" \
    --sarclip-model "${SARCLIP_MODEL}" \
    --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
    --sarclip-lora "${SARCLIP_LORA}" \
    --cga-lambda "${CGA_LAMBDA}" \
    --seed "${SEED}" \
    "${CGA_ARGS[@]}"
fi

conda run -n "${ENV_NAME}" python tools/collect_controls_results.py --result-root "${RESULT_ROOT}"
echo "[exp_rsar_controls] done"
