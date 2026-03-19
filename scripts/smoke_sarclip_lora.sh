#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# P0032+P0033 Smoke test: SARCLIP LoRA fine-tuning with L_ent
# Verifies: training runs, checkpoint saved, checkpoint loadable by CGA
# ============================================================================

cd /home/zechuan/IRAOD
export PATH=/home/zechuan/miniconda3/envs/iraod/bin:$PATH

OUTDIR="work_dirs/sanity/sarclip_lora_smoke"
mkdir -p "$OUTDIR"

echo "[smoke] Step 1: LoRA training (2 epochs, small batch, with L_ent)..."
python -m lora_finetune.lora_sarclip_train \
    --sarclip-dir third_party/SARCLIP \
    --sarclip-model RN50 \
    --data-root dataset/RSAR/train/images \
    --ann-file dataset/RSAR/train/annfiles \
    --classes ship aircraft car tank bridge harbor \
    --epochs 2 \
    --batch-size 8 \
    --lr 1e-4 \
    --r 4 --alpha 8 \
    --ent-weight 0.1 --ent-score-thr 0.5 \
    --output-dir "$OUTDIR"

CKPT="$OUTDIR/lora_final.pth"
if [ ! -f "$CKPT" ]; then
    echo "[smoke] FAIL: checkpoint not found at $CKPT"
    exit 1
fi
echo "[smoke] Step 1 PASSED: checkpoint saved at $CKPT"

echo "[smoke] Step 2: Verify checkpoint is loadable..."
python -c "
import sys, torch
sys.path.insert(0, 'third_party/SARCLIP')
import sar_clip
from tools.lora_utils import LoraConfig, inject_lora, load_lora_state_dict

# Load model
model = sar_clip.create_model_with_args('RN50', pretrained=None, precision='fp32',
                                         device='cpu', cache_dir=None, output_dict=True)

# Load LoRA checkpoint
ckpt = torch.load('$CKPT', map_location='cpu')
meta = ckpt.get('meta', {})
print(f'[smoke] LoRA meta: {meta}')

r = int(meta.get('r', 8))
alpha = float(meta.get('alpha', 16.0))

def vf(name, _m): return name.startswith('visual.')
n = inject_lora(model, config=LoraConfig(r=r, alpha=alpha), module_filter=vf)
load_lora_state_dict(model, ckpt['state_dict'])
print(f'[smoke] Injected LoRA into {n} layers, loaded state dict')

# Forward pass
x = torch.randn(1, 3, 224, 224)
out = model(image=x)
feats = out['image_features']
print(f'[smoke] encode_image OK, shape={feats.shape}')
print('[smoke] Step 2 PASSED')
"

echo "[smoke] Step 3: Verify SARCLIP_LORA env var in CGA..."
SARCLIP_LORA="$CKPT" python -c "
import os, sys, torch
os.environ['SARCLIP_LORA'] = '$CKPT'
os.environ['CGA_SCORER'] = 'sarclip'
os.environ['SARCLIP_MODEL'] = 'RN50'
sys.path.insert(0, 'third_party/SARCLIP')

# Test that CGA can load with LoRA
from sfod.cga import CGA
classes = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
cga = CGA(class_names=classes, model='RN50', pretrained=None,
          precision='fp32', templates=('an SAR image of a {}',))
print('[smoke] CGA with SARCLIP_LORA loaded successfully')
print('[smoke] Step 3 PASSED')
"

echo ""
echo "[smoke] ALL TESTS PASSED"
echo "[smoke] LoRA checkpoint: $CKPT"
echo "[smoke] Contains ent_weight=$(python -c "import torch; m=torch.load('$CKPT',map_location='cpu').get('meta',{}); print(m.get('ent_weight',0))")"

# Cleanup: keep the checkpoint as output artifact, don't self-delete
