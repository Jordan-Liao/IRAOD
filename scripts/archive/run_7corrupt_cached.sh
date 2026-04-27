#!/bin/bash
# 7 种干扰 × OrthoNet + SARCLIP LoRA + 12ep SFOD
# 10 GPU + CGA DiskCache (第1 epoch 后全命中缓存)
set -e

cd /mnt/SSD1_8TB/zechuan/IRAOD
source ~/anaconda3/etc/profile.d/conda.sh
conda activate iraod

CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py"
DATA_ROOT="/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR"
export SARCLIP_LORA="/mnt/SSD1_8TB/zechuan/IRAOD/lora_finetune/SARCLIP_LoRA_Interference.pt"

# 启用 CGA 磁盘缓存 — 第1个epoch计算分数后缓存到磁盘，后续epoch直接读取
export CGA_CACHE_VERBOSE=1

CORRUPTS=(chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)

echo "============================================================"
echo "  7 Corruptions × OrthoNet + SARCLIP LoRA + 12ep SFOD"
echo "  10 GPU + CGA DiskCache"
echo "  Start: $(date)"
echo "============================================================"

PORT=29650
for corrupt in "${CORRUPTS[@]}"; do
    work_dir="work_dirs/exp_sfod_ortho_sarclip_${corrupt}"
    cache_dir="work_dirs/cga_cache_${corrupt}"
    rm -rf "$work_dir"
    mkdir -p "$work_dir" "$cache_dir"

    # 每种干扰用独立的缓存目录
    export CGA_CACHE_DIR="$cache_dir"

    echo ""
    echo "[$(date +%H:%M:%S)] === $corrupt === (10 GPU, cache=$cache_dir)"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -m torch.distributed.launch \
        --nproc_per_node=10 \
        --master_port=$PORT \
        train.py "$CONFIG" \
        --work-dir "$work_dir" \
        --data-root "$DATA_ROOT" \
        --launcher pytorch \
        --cga-scorer sarclip \
        --sarclip-model ViT-L-14 \
        --sarclip-pretrained weights/sarclip/ViT-L-14/vit_l_14_model.safetensors \
        --cfg-options \
            corrupt="$corrupt" \
        > "$work_dir/train.log" 2>&1

    exit_code=$?
    echo "[$(date +%H:%M:%S)] $corrupt done (exit=$exit_code)"

    # 统计缓存命中
    cache_files=$(find "$cache_dir" -name "*.json" | wc -l)
    echo "  Cache files: $cache_files"

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
