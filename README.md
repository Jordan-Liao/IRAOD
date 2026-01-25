# IRAOD
IRAOD is a framework for oriented object detection in remote sensing images, with scripts that make DIOR/RSAR training + corruption robustness experiments reproducible.

For the full experiment ledger (smoke/full commands + logs + artifacts + results), see:
- `docs/experiment.md`
- `README_experiments.md`

## Getting Started (Reproducible)

### 1) Environment

Tested environment (this repo’s default scripts assume it):
- Python `3.10`
- PyTorch `2.0.1+cu118`
- MMCV `1.7.2`
- MMDetection `2.28.2`
- MMRotate `0.3.4`

Create a conda env (name is arbitrary; docs use `dino_sar`):
```bash
conda create -n dino_sar python=3.10 -y
conda activate dino_sar
```

Install PyTorch (example for CUDA 11.8; adjust if your CUDA differs):
```bash
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

Install OpenMMLab stack (MMCV must match your torch/cuda; example below is for torch2.0 + cu118):
```bash
pip install -U openmim
mim install "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install "mmdet==2.28.2" "mmrotate==0.3.4"
```

Install the remaining deps:
```bash
pip install -r requirements.txt
```

### 2) Datasets

All scripts assume datasets live under:
- `dataset/DIOR`
- `dataset/RSAR`

Expected layouts:
```text
dataset/DIOR/
  Annotations/
  JPEGImages/
  ImageSets/
  Corruption/
    JPEGImages-cloudy/
    JPEGImages-brightness/
    JPEGImages-contrast/
    ...
```

```text
dataset/RSAR/
  train/
    annfiles/   # *.txt
    images/     # *.jpg/*.png/*.bmp
  val/
    annfiles/
    images/
  test/
    annfiles/
    images/
```

Verify layouts:
```bash
conda run -n dino_sar python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR
```

### 3) Checkpoints / Weights

- Oriented-RCNN baseline weights (DIOR pretrain) go to `baseline/` (see `MODEL_ZOO.md`).
- SARCLIP weights go to `weights/sarclip/<MODEL>/` (see `weights/README.md`).

### 4) Quick Smoke (recommended first run)

```bash
bash scripts/smoke_dior.sh
bash scripts/smoke_rsar.sh
```

### 5) Core Experiments

DIOR:
```bash
bash scripts/exp_dior_baseline_eval.sh
bash scripts/exp_dior_ut.sh
bash scripts/exp_dior_ut_cga_clip.sh
```

RSAR:
```bash
bash scripts/exp_rsar_baseline.sh

# UT (no CGA)
CGA_SCORER=none bash scripts/exp_rsar_ut.sh

# UT + CGA (CLIP)
CGA_SCORER=clip bash scripts/exp_rsar_ut.sh

# UT + CGA (SARCLIP)
CGA_SCORER=sarclip SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors bash scripts/exp_rsar_ut.sh
```

### 6) RSAR Interference / Robustness

Generate interference directories (disk-heavy; see `docs/plan.md` for design rationale):
```bash
# test-only severity suites (interf_jamA_s1..s5 / interf_jamB_s1..s5)
bash scripts/prepare_rsar_interf_severity_test.sh

# representative training severity (interf_jamB_s3) for train/val
bash scripts/prepare_rsar_interf_jamB_s3_trainval.sh
```

The jamB_s3 robustness matrix (baseline / UT / UT+CGA; interf-only & mix) is recorded in `docs/experiment.md` (E0028–E0033).

### 7) Refresh Result Tables

After you run new experiments under `work_dirs/`, regenerate summary tables:
```bash
bash scripts/refresh_results.sh
```

## 💡 Acknowledgement

We thank the authors of the following works for their open-source contributions:

- [MMRotate](https://github.com/open-mmlab/mmrotate) - Rotated object detection framework
- [DOTA-C](https://github.com/hehaodong530/DOTA-C) - Corruption robustness benchmark
- [Unbiased Teacher](https://arxiv.org/abs/2102.05622) - Semi-supervised learning method


---

## 🖊️ Citation



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

