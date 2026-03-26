# IRAOD
IRAOD is a framework for oriented object detection in remote sensing images, with scripts that make DIOR/RSAR training + corruption robustness experiments reproducible.

For the full experiment ledger (smoke/full commands + logs + artifacts + results), see:
- `docs/experiment.md`
- `README_experiments.md`

## Getting Started (Reproducible)

### TL;DR (5 commands)

```bash
# 1) install env (CUDA 11.8 by default)
bash scripts/setup_env_iraod.sh
conda activate iraod

# 2) download datasets (optional helper)
conda run -n iraod python tools/download_datasets.py rsar --source gdrive
# (DIOR needs bypy: see "Datasets" below)

# 3) verify layout
conda run -n iraod python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR

# 4) smoke
bash scripts/smoke_dior.sh
bash scripts/smoke_rsar.sh

# 5) refresh tables
bash scripts/refresh_results.sh
```

### 1) Environment

Tested environment (this repo’s default scripts assume it):
- Python `3.10`
- PyTorch `2.0.1+cu118`
- MMCV `1.7.2`
- MMDetection `2.28.2`
- MMRotate `0.3.4`

Repo scripts default to conda env name `iraod`. If you use another env name, export `ENV_NAME=<your_env>` when running `scripts/*.sh`, and replace `conda run -n iraod ...` accordingly.

Recommended (one-shot):
```bash
bash scripts/setup_env_iraod.sh
conda activate iraod
```

CPU-only install:
```bash
CUDA_VARIANT=cpu bash scripts/setup_env_iraod.sh
conda activate iraod
```

Manual install (if you prefer step-by-step):

Create a conda env:
```bash
conda create -n iraod python=3.10 -y
conda activate iraod
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
```

Install the remaining deps:
```bash
pip install -r requirements.txt
```

### 2) Datasets

All scripts assume datasets live under:
- `dataset/DIOR`
- `dataset/RSAR`

Optional: auto download / organize (recommended for reproducibility):

Install and login `bypy` (only needed if you use Baidu NetDisk / `--source bypy`):
```bash
# bypy is installed via requirements.txt; login is a one-time step.
conda run -n iraod bypy info
```

Notes about `bypy`:
- `bypy` cannot download a Baidu share link directly; you must first save/transfer files into your own `/apps/bypy` via browser.
- If you are behind a proxy, this repo’s downloader will unset `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` for `bypy` calls.

Download datasets using the helper (see `tools/download_datasets.py --help`):
```bash
# DIOR: requires you already saved the dataset folder into /apps/bypy/<REMOTE_DIR>
conda run -n iraod python tools/download_datasets.py dior --source bypy --bypy-remote <REMOTE_DIR>

# RSAR: default is Google Drive; or use bypy if you already saved RSAR.tar under /apps/bypy/
conda run -n iraod python tools/download_datasets.py rsar --source gdrive
# conda run -n iraod python tools/download_datasets.py rsar --source bypy --bypy-remote RSAR.tar --dest dataset/RSAR
```

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
conda run -n iraod python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR
```

Prepare DIOR corruption folders (needed by DIOR corruption eval):
```bash
conda run -n iraod python tools/prepare_dior_corruption.py \
  --data-root dataset/DIOR --corrupt clean cloudy brightness contrast \
  --splits val,test --workers 8
```

### 3) Checkpoints / Weights

- Oriented-RCNN baseline weights (DIOR pretrain) go to `baseline/baseline.pth` (not tracked by git; see `MODEL_ZOO.md`).
- SARCLIP weights go to `weights/sarclip/<MODEL>/` (see `weights/README.md` for Baidu/bypy instructions).

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

Generate interference/corruption directories (disk-heavy; see `docs/plan.md` for design rationale):

**(Recommended / compliant)** RSAR 7-type corruption subsets under `dataset/RSAR/corruptions/`:
```bash
# Generates:
#   dataset/RSAR/corruptions/{chaff,gaussian_white_noise,point_target,noise_suppression,am_noise_horizontal,smart_suppression,am_noise_vertical}/{train,val,test}/images
# And creates legacy symlinks:
#   dataset/RSAR/<split>/images-<corrupt> -> dataset/RSAR/corruptions/<corrupt>/<split>/images
python tools/prepare_rsar_corruption.py --data-root dataset/RSAR --workers 8
```

**(Legacy)** `interf_jam*` severity suites:
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


## 🔬 Innovations (新加创新点)

### Innovation 1: OrthoNet + OCA-FPN

将正交滤波网络（[OrthoNets](https://github.com/hady1011/OrthoNets)）融入本项目，测试正交通道注意力对旋转目标检测的效果。

- **OrthoNet backbone**: 正交通道注意力网络，注册为 `OrthoNet`（`mmdet_extension/models/backbones/orthonet.py`）
- **OCA-FPN neck**: FPN + OrthoChannelAttention on each output level（`mmdet_extension/models/necks/oca_fpn.py`）
- **Config 文件**:
  - `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py`（RSAR）
  - `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o.py`（DIOR）
- **实验**: frontier-026 (OCA-FPN, 24ep schedule)，训练中（Epoch 13/24）
- **Val mAP**: 0.6511（Epoch 11）

### Innovation 2: SAR-CLIP LoRA 微调（干扰鲁棒 CGA 打分）

性能提升 SAR-CLIP（来自 [SAR-TEXT](https://github.com/YiguoHe/SAR-TEXT) 的 SAR-RS-CLIP.pt，基于 OpenCLIP ViT-L-14）在 RSAR 6 类 + 7 种干扰下的零样本分类准确率，从而让 IRAOD 的 CGA（语义打分纠正）模块对 SAR 更准、更鲁棒。

**核心逻辑**:
- 用 SARDet100k 的训练集标注精确裁剪目标 patch（水平框）+ 叠加干扰 → 构造高质量"干扰鲁棒"图像-文本对
- 用 LoRA（仅微调 vision encoder，参数量 <1%）在这些对上继续对比学习（contrastive loss）
- 得到 SARCLIP-Interference-LoRA 后，直接替换 SFOD-RS scorer，打分更准 → 伪标签质量更高 → 最终检测 mAP 再提升

**图像-文本对构造**: 裁剪的检测图像位于相应的类别文件夹，文本为 `[f"a SAR image of a {cls}" for cls in class_names]`

**代码**: `lora_finetune/lora_sarclip_train.py`, `lora_finetune/crop_sardet100k.py`

**LoRA 训练参数**: 4 linear layers, 122,880 trainable params (0.12%), 147,796 patches from RSAR train

**结果**:

| 指标 | 无 LoRA | 有 LoRA | 提升 |
|------|---------|---------|------|
| CGA 零样本分类精度 (RSAR 6类) | 0.6021 | **0.6513** | **+4.9%** |
| 端到端检测 mAP (最佳 SFOD 迭代 exp_ax) | 0.6842 | **0.6943** | **+1.0%** |

**SFOD 迭代链 LoRA 对比**:

| 迭代 | 无 LoRA mAP | 有 LoRA mAP | Delta |
|------|-------------|-------------|-------|
| exp_x | 0.6694 | 0.6797 | +0.0103 |
| exp_ab | 0.6706 | 0.6794 | +0.0088 |
| exp_an | 0.6800 | 0.6909 | +0.0109 |
| exp_ar | 0.6828 | 0.6935 | +0.0107 |
| exp_au | 0.6834 | 0.6938 | +0.0104 |
| **exp_ax** | **0.6842** | **0.6943** | **+0.0101** |

## 💡 Acknowledgement

We thank the authors of the following works for their open-source contributions:

- [MMRotate](https://github.com/open-mmlab/mmrotate) - Rotated object detection framework
- [DOTA-C](https://github.com/hehaodong530/DOTA-C) - Corruption robustness benchmark
- [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher) - Semi-supervised learning method


---

## 🖊️ Citation



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
