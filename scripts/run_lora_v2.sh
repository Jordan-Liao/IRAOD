#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# P0032v2 + P0033v2: SARCLIP LoRA (r=16, epochs=30) 5-GPU distributed
# Waits for current LoRA experiments to finish, then runs larger-scale
# ============================================================================

cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:/home/zechuan/miniconda3/bin:$PATH

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

NGPU=5
MASTER_PORT=29503
TRAIN_SCRIPT="lora_finetune/lora_sarclip_train.py"

# ---- Wait for current LoRA to finish ----
log "Waiting for current LoRA experiments to finish..."
while pgrep -f "lora_sarclip_train" > /dev/null 2>&1; do
    sleep 30
done
log "Previous LoRA done. Starting v2 (r=16, epochs=30)."

# ---- P0032v2: LoRA r=16 baseline ----
P0032_DIR="work_dirs/p0032v2_sarclip_lora_r16"
mkdir -p "$P0032_DIR"

log "=== P0032v2: SARCLIP LoRA r=16 (no L_ent) | ${NGPU} GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
    --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --sarclip-dir third_party/SARCLIP \
    --sarclip-model RN50 \
    --data-root dataset/RSAR/train/images \
    --ann-file dataset/RSAR/train/annfiles \
    --classes ship aircraft car tank bridge harbor \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --lora-r 16 --lora-alpha 32 \
    --ent-weight 0.0 \
    --output-dir "$P0032_DIR" \
    2>&1 | tee "$P0032_DIR/train.log"

log "P0032v2 finished. Checkpoint: $P0032_DIR/lora_final.pth"

# ---- P0033v2: LoRA r=16 + L_ent ----
P0033_DIR="work_dirs/p0033v2_sarclip_lora_ent_r16"
mkdir -p "$P0033_DIR"

log "=== P0033v2: SARCLIP LoRA r=16 + L_ent (ent_weight=0.1) | ${NGPU} GPUs ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun \
    --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --sarclip-dir third_party/SARCLIP \
    --sarclip-model RN50 \
    --data-root dataset/RSAR/train/images \
    --ann-file dataset/RSAR/train/annfiles \
    --classes ship aircraft car tank bridge harbor \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --lora-r 16 --lora-alpha 32 \
    --ent-weight 0.1 --ent-score-thr 0.5 \
    --output-dir "$P0033_DIR" \
    2>&1 | tee "$P0033_DIR/train.log"

log "P0033v2 finished. Checkpoint: $P0033_DIR/lora_final.pth"

# ---- Verify ----
log "=== Verification ==="
python -c "
import torch, sys
sys.path.insert(0, 'third_party/SARCLIP')
import sar_clip
from tools.lora_utils import LoraConfig, inject_lora, load_lora_state_dict

for name, path in [('P0032v2', '$P0032_DIR/lora_final.pth'), ('P0033v2', '$P0033_DIR/lora_final.pth')]:
    ckpt = torch.load(path, map_location='cpu')
    meta = ckpt.get('meta', {})
    m = sar_clip.create_model_with_args('RN50', pretrained=None, precision='fp32', device='cpu', cache_dir=None, output_dict=True)
    def vf(n, _): return n.startswith('visual.')
    inject_lora(m, LoraConfig(r=int(meta.get('r',16)), alpha=float(meta.get('alpha',32))), module_filter=vf)
    load_lora_state_dict(m, ckpt['state_dict'])
    x = torch.randn(1,3,224,224)
    out = m(image=x)
    print(f'  [{name}] OK | r={meta.get(\"r\")} ent_weight={meta.get(\"ent_weight\",0)} epochs={meta.get(\"epoch\")} | shape={out[\"image_features\"].shape}')
"

log "=== ALL DONE (v2) ==="
log "P0032v2: $P0032_DIR/lora_final.pth"
log "P0033v2: $P0033_DIR/lora_final.pth"
