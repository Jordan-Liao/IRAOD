#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# IRAOD experiment chain: Baseline → UT Corrected → Exp M
# Multi-GPU (5x A6000) training
# ============================================================================

cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:/home/zechuan/miniconda3/bin:$PATH

NGPU=5
PORT=29501

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- Step 1: Wait for baseline to finish ----
log "Waiting for baseline training to finish..."
while pgrep -f "baseline_oriented_rcnn_rsar" > /dev/null 2>&1; do
    sleep 60
done
log "Baseline training finished."

# Find the best baseline checkpoint
BASELINE_DIR="work_dirs/exp_rsar_baseline"
BASELINE_CKPT=""
for e in 12 11 10 9 8; do
    if [ -f "${BASELINE_DIR}/epoch_${e}.pth" ]; then
        BASELINE_CKPT="${BASELINE_DIR}/epoch_${e}.pth"
        break
    fi
done
# Also check for latest.pth
if [ -z "$BASELINE_CKPT" ] && [ -f "${BASELINE_DIR}/latest.pth" ]; then
    BASELINE_CKPT="${BASELINE_DIR}/latest.pth"
fi
if [ -z "$BASELINE_CKPT" ]; then
    log "ERROR: No baseline checkpoint found in ${BASELINE_DIR}"
    exit 1
fi
log "Using baseline checkpoint: ${BASELINE_CKPT}"

# ---- Step 2: Run UT Corrected Baseline ----
UT_CONFIG="configs/unbiased_teacher/sfod/ut_oriented_rcnn_r50_rsar_corrected.py"
UT_WORKDIR="work_dirs/ut_rsar_corrected"
mkdir -p "${UT_WORKDIR}"

log "Starting UT corrected baseline (5-GPU)..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch \
    --nproc_per_node=${NGPU} --master_port=${PORT} \
    train.py "${UT_CONFIG}" \
    --work-dir "${UT_WORKDIR}" \
    --launcher pytorch \
    --cfg-options \
        "load_from=${BASELINE_CKPT}" \
        "model.ema_ckpt=${BASELINE_CKPT}" \
        data.samples_per_gpu=2 \
        data.workers_per_gpu=4 \
    2>&1 | tee "${UT_WORKDIR}/train.log"

log "UT corrected training finished."

# Find UT checkpoint (last epoch or iteration)
UT_CKPT=""
for e in 12 11 10 9 8; do
    if [ -f "${UT_WORKDIR}/epoch_${e}.pth" ]; then
        UT_CKPT="${UT_WORKDIR}/epoch_${e}.pth"
        break
    fi
done
# Check iteration checkpoints (SemiEpochBasedRunner uses iter_XXXX.pth)
if [ -z "$UT_CKPT" ]; then
    UT_CKPT=$(ls -t ${UT_WORKDIR}/iter_*.pth 2>/dev/null | grep -Ev '_ema\.pth$' | head -1 || true)
fi
if [ -z "$UT_CKPT" ] && [ -f "${UT_WORKDIR}/latest.pth" ]; then
    UT_CKPT="${UT_WORKDIR}/latest.pth"
fi
if [ -z "$UT_CKPT" ]; then
    log "ERROR: No UT checkpoint found in ${UT_WORKDIR}"
    exit 1
fi
log "Using UT checkpoint: ${UT_CKPT}"

# ---- Step 3: Run Exp M (best experiment, mAP 0.6701) ----
EXP_M_CONFIG="configs/unbiased_teacher/sfod/exp_m_wu_schedule.py"
EXP_M_WORKDIR="work_dirs/exp_m_wu_schedule"
mkdir -p "${EXP_M_WORKDIR}"

log "Starting Exp M (weight_u schedule, 5-GPU)..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch \
    --nproc_per_node=${NGPU} --master_port=${PORT} \
    train.py "${EXP_M_CONFIG}" \
    --work-dir "${EXP_M_WORKDIR}" \
    --launcher pytorch \
    --cfg-options \
        "load_from=${UT_CKPT}" \
        "model.ema_ckpt=${UT_CKPT}" \
        data.samples_per_gpu=2 \
        data.workers_per_gpu=4 \
    2>&1 | tee "${EXP_M_WORKDIR}/train.log"

log "Exp M training finished."
log "All experiments complete!"
