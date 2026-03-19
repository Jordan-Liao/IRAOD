#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# P0032 + P0033: SARCLIP LoRA 5-GPU distributed training on RSAR
# ============================================================================

cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:/home/zechuan/miniconda3/bin:$PATH

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

NGPU=5
MASTER_PORT=29503
TRAIN_SCRIPT="lora_finetune/lora_sarclip_train.py"

log "Starting LoRA experiments (5-GPU distributed)."

# ---- P0032: Full SARCLIP LoRA (no L_ent, baseline) ----
P0032_DIR="work_dirs/p0032_sarclip_lora"
mkdir -p "$P0032_DIR"

log "=== P0032: SARCLIP LoRA fine-tuning (baseline, no L_ent) | ${NGPU} GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
    --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --sarclip-dir third_party/SARCLIP \
    --sarclip-model RN50 \
    --data-root dataset/RSAR/train/images \
    --ann-file dataset/RSAR/train/annfiles \
    --classes ship aircraft car tank bridge harbor \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --lora-r 8 --lora-alpha 16 \
    --ent-weight 0.0 \
    --output-dir "$P0032_DIR" \
    2>&1 | tee "$P0032_DIR/train.log"

log "P0032 finished. Checkpoint: $P0032_DIR/lora_final.pth"

# ---- P0033: SARCLIP LoRA + L_ent entropy minimization ----
P0033_DIR="work_dirs/p0033_sarclip_lora_ent"
mkdir -p "$P0033_DIR"

log "=== P0033: SARCLIP LoRA + L_ent (ent_weight=0.1, score_thr=0.5) | ${NGPU} GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
    --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --sarclip-dir third_party/SARCLIP \
    --sarclip-model RN50 \
    --data-root dataset/RSAR/train/images \
    --ann-file dataset/RSAR/train/annfiles \
    --classes ship aircraft car tank bridge harbor \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --lora-r 8 --lora-alpha 16 \
    --ent-weight 0.1 --ent-score-thr 0.5 \
    --output-dir "$P0033_DIR" \
    2>&1 | tee "$P0033_DIR/train.log"

log "P0033 finished. Checkpoint: $P0033_DIR/lora_final.pth"

# ---- Verify both checkpoints loadable ----
log "=== Verification: loading both checkpoints ==="
python -c "
import torch, sys
sys.path.insert(0, 'third_party/SARCLIP')
import sar_clip
from tools.lora_utils import LoraConfig, inject_lora, load_lora_state_dict

for name, path in [('P0032', '$P0032_DIR/lora_final.pth'), ('P0033', '$P0033_DIR/lora_final.pth')]:
    ckpt = torch.load(path, map_location='cpu')
    meta = ckpt.get('meta', {})
    m = sar_clip.create_model_with_args('RN50', pretrained=None, precision='fp32', device='cpu', cache_dir=None, output_dict=True)
    def vf(n, _): return n.startswith('visual.')
    inject_lora(m, LoraConfig(r=int(meta.get('r',8)), alpha=float(meta.get('alpha',16))), module_filter=vf)
    load_lora_state_dict(m, ckpt['state_dict'])
    x = torch.randn(1,3,224,224)
    out = m(image=x)
    print(f'  [{name}] OK | ent_weight={meta.get(\"ent_weight\",0)} | shape={out[\"image_features\"].shape}')
"

log "=== ALL DONE ==="
log "P0032 checkpoint: $P0032_DIR/lora_final.pth"
log "P0033 checkpoint: $P0033_DIR/lora_final.pth"
