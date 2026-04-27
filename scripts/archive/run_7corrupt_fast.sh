#!/bin/bash
# 7 种干扰 × OrthoNet + SARCLIP LoRA + 12ep SFOD
# 5 GPU, samp=8, test_interval=3 evaluation.interval=1, workers=8, precomputed features
set -e
export NCCL_BLOCKING_WAIT=1

cd /mnt/SSD1_8TB/zechuan/IRAOD
source ~/anaconda3/etc/profile.d/conda.sh
conda activate iraod

CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py"
DATA_ROOT="/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR"
export SARCLIP_LORA="/mnt/SSD1_8TB/zechuan/IRAOD/lora_finetune/SARCLIP_LoRA_Interference.pt"
export CGA_PRECOMPUTED_DIR="/mnt/SSD1_8TB/zechuan/IRAOD/work_dirs/sarclip_features"

CORRUPTS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

echo "============================================================"
echo "  7 Corruptions × OrthoNet + SARCLIP + 12ep SFOD"
echo "  5 GPU, samp=8, test_interval=3 evaluation.interval=1, workers=8"
echo "  Precomputed features: $CGA_PRECOMPUTED_DIR"
echo "  Start: $(date)"
echo "============================================================"

PORT=29680
for corrupt in "${CORRUPTS[@]}"; do
    work_dir="work_dirs/exp_sfod_ortho_sarclip_${corrupt}"
    rm -rf "$work_dir"
    mkdir -p "$work_dir"

    echo ""
    echo "[$(date +%H:%M:%S)] === $corrupt ==="

    CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch \
        --nproc_per_node=5 \
        --master_port=$PORT \
        train.py "$CONFIG" \
        --work-dir "$work_dir" \
        --data-root "$DATA_ROOT" \
        --launcher pytorch \
        --samples-per-gpu 8 \
        --workers-per-gpu 8 \
        --cga-scorer sarclip \
        --sarclip-model ViT-L-14 \
        --sarclip-pretrained weights/sarclip/ViT-L-14/vit_l_14_model.safetensors \
        --cfg-options \
            corrupt="$corrupt" \
            test_interval=3 evaluation.interval=1 \
        > "$work_dir/train.log" 2>&1

    echo "[$(date +%H:%M:%S)] $corrupt done (exit=$?)"
    PORT=$((PORT + 1))

    python -c "
import json, glob
best = None
for lj in sorted(glob.glob('${work_dir}/*.log.json')):
    with open(lj) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
                if e.get('mode')=='val' and e.get('mAP') is not None:
                    if best is None or e['mAP']>best: best = e['mAP']
            except: pass
print(f'  Best mAP: {best:.4f}' if best else '  Best mAP: N/A')
" 2>/dev/null
done

echo ""
echo "=== Final Results ==="
for corrupt in "${CORRUPTS[@]}"; do
    work_dir="work_dirs/exp_sfod_ortho_sarclip_${corrupt}"
    python -c "
import json, glob
best = None
for lj in sorted(glob.glob('${work_dir}/*.log.json')):
    with open(lj) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
                if e.get('mode')=='val' and e.get('mAP') is not None:
                    if best is None or e['mAP']>best: best = e['mAP']
            except: pass
print(f'  ${corrupt}: mAP={best:.4f}' if best else f'  ${corrupt}: N/A')
" 2>/dev/null
done
echo "Done at $(date)"
