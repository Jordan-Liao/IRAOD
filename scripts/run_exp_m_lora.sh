#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Exp M + SARCLIP LoRA: 用 LoRA 增强的 SARCLIP 做 CGA 伪标签重评分
# 与原始 Exp M 对比，看 LoRA 对训练时 CGA 的影响
# ============================================================================

cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:/home/zechuan/miniconda3/bin:$PATH

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

NGPU=5
MASTER_PORT=29504
WORK_DIR="work_dirs/exp_m_lora_cga"
UT_WORKDIR="${UT_WORKDIR:-work_dirs/ut_rsar_corrected}"

if [ -n "${EXP_M_INIT_CKPT:-}" ]; then
    INIT_CKPT="$EXP_M_INIT_CKPT"
else
    INIT_CKPT="$(ls -t "${UT_WORKDIR}"/iter_*.pth 2>/dev/null | grep -Ev '_ema\.pth$' | head -1 || true)"
    if [ -z "$INIT_CKPT" ] && [ -f "${UT_WORKDIR}/latest.pth" ]; then
        INIT_CKPT="${UT_WORKDIR}/latest.pth"
    fi
fi
EMA_CKPT="${EXP_M_EMA_CKPT:-$INIT_CKPT}"

# 设置 SARCLIP LoRA 环境变量（使用最充分训练的 P0032v2, r=16, 30ep）
export SARCLIP_LORA="${PWD}/work_dirs/p0032v2_sarclip_lora_r16/lora_final.pth"
export CGA_SCORER="sarclip"
export SARCLIP_MODEL="RN50"

if [ ! -f "$SARCLIP_LORA" ]; then
    log "ERROR: LoRA checkpoint not found: $SARCLIP_LORA"
    exit 1
fi

if [ ! -f "$INIT_CKPT" ]; then
    log "ERROR: init checkpoint not found: $INIT_CKPT"
    exit 1
fi

if [ ! -f "$EMA_CKPT" ]; then
    log "ERROR: EMA checkpoint not found: $EMA_CKPT"
    exit 1
fi

log "=== Exp M + SARCLIP LoRA CGA ==="
log "SARCLIP_LORA=$SARCLIP_LORA"
log "Work dir: $WORK_DIR"
log "Init checkpoint: $INIT_CKPT"
log "EMA checkpoint: $EMA_CKPT"

mkdir -p "$WORK_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch \
    --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
    train.py configs/unbiased_teacher/sfod/exp_m_wu_schedule.py \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    --cfg-options \
        "load_from=$INIT_CKPT" \
        "model.ema_ckpt=$EMA_CKPT" \
        data.samples_per_gpu=2 \
        data.workers_per_gpu=4 \
    2>&1 | tee "$WORK_DIR/train.log"

log "=== Exp M + LoRA DONE ==="
