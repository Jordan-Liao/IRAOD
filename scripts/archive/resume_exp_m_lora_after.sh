#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:/home/zechuan/miniconda3/bin:$PATH

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log 'Waiting for Exp M (original) to finish...'
while pgrep -f 'configs/unbiased_teacher/sfod/exp_m_wu_schedule.py|work_dirs/exp_m_wu_schedule' > /dev/null 2>&1; do
    sleep 60
done
log 'Exp M done. Starting Exp M + LoRA resume...'

# Set SARCLIP LoRA env vars
export SARCLIP_LORA="${PWD}/work_dirs/p0032v2_sarclip_lora_r16/lora_final.pth"
export CGA_SCORER="sarclip"
export SARCLIP_MODEL="RN50"
RESUME_CKPT="${PWD}/work_dirs/exp_m_lora_cga/latest.pth"
RESUME_EMA_CKPT="${PWD}/work_dirs/exp_m_lora_cga/latest_ema.pth"

if [ ! -f "$SARCLIP_LORA" ]; then
    log "ERROR: LoRA checkpoint not found: $SARCLIP_LORA"
    exit 1
fi

if [ ! -f "$RESUME_CKPT" ]; then
    log "ERROR: resume checkpoint not found: $RESUME_CKPT"
    exit 1
fi

if [ ! -f "$RESUME_EMA_CKPT" ]; then
    log "ERROR: resume EMA checkpoint not found: $RESUME_EMA_CKPT"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch     --nproc_per_node=5 --master_port=29504     train.py configs/unbiased_teacher/sfod/exp_m_lora_resume.py     --work-dir work_dirs/exp_m_lora_cga     --launcher pytorch     --cfg-options         data.samples_per_gpu=2         data.workers_per_gpu=4     2>&1 | tee work_dirs/exp_m_lora_cga/resume.log

log 'Exp M + LoRA DONE'
