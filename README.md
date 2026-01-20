
# IRAOD



<p align="center">
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-acknowledgement">Acknowledgement</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📋 Overview

IRAOD is a comprehensive framework for oriented object detection in remote sensing images. It provides implementations of advanced semi-supervised learning methods including Unbiased Teacher, STAC, and CGA (Curriculum-Guided Augmentation) for detecting rotated objects in challenging conditions.

**Key Features:**
- 🔄 Semi-supervised learning with semi-supervised frameworks
- 🎯 Oriented object detection for rotated bounding boxes
- 📸 Support for DIOR and RSAR datasets
- 🌧️ Corruption robustness (Cloudy, Brightness, Contrast, etc.)
- 🚀 Easy training and testing pipeline

---
## 🎮 Getting Started

### 1️⃣ Install Environment

```bash
# Create conda environment
conda create --name IROAD python=3.8
conda activate IROAD

# Install PyTorch (adjust CUDA version as needed)
pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html --no-cache

# Clone and install dependencies
git clone https://github.com/Jordan-Liao/IRAOD.git
cd IRAOD
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

#### For DIOR Dataset
- Download the DIOR dataset from:
  - 🔗 [Google Drive](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
  - 🔗 [BaiduNetDisk](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg#list/path=%2F)

- Download the corruption images with cloud (DIOR-Cloudy) from [Google Drive](https://drive.google.com/drive/folders/11l2L5ScsFQ7FH64vd0mub9hVcO1BK1py)

  > **Note:** For more details about other corruptions and original cloudy images, please refer to [DOTA-C](https://github.com/hehaodong530/DOTA-C).

- Organize the dataset as follows:

  ```
  dataset/
  ├── DIOR/
  │   ├── Annotations              # Annotation files
  │   ├── JPEGImages               # Original images
  │   ├── ImageSets                # Train/val/test splits
  │   └── Corruption/
  │       ├── JPEGImages-brightness
  │       ├── JPEGImages-cloudy
  │       ├── JPEGImages-contrast
  │       └── ...
  ```

  - `JPEGImages`: All images in DIOR dataset
  - `ImageSets`: Train/val/test splits
  - `Corruption`: Corrupted images (brightness, cloudy, contrast, etc.)

#### For RSAR Dataset
- Download the `RSAR` dataset from:
  - 🔗 [Google Drive](https://drive.google.com/file/d/1v-HXUSmwBQCtrq0MlTOkCaBQ_vbz5_qs/view?usp=sharing)
  - 🔗 [BaiduNetDisk](https://pan.baidu.com/s/1g2NGfzf7Xgk_K9euKVjFEA?pwd=rsar)

- Extract files to `$DATAROOT` with the following structure:

  ```
  $DATAROOT
  ├── train
  │   ├── annfiles    # Annotation files (*.txt)
  │   └── images      # SAR images (*.jpg, *.bmp, *.png)
  ├── val
  │   ├── annfiles
  │   └── images
  └── test
      ├── annfiles
      └── images
  ```

---

### 3️⃣ Download Checkpoints

Before training, download the pretrained Oriented-RCNN model weights:

- **Baseline Model** (trained on DIOR): [baseline.pth](https://drive.google.com/file/d/1JOxD7eHrMkDFe9rBEgSTxBFAuTW1jXza/view?usp=drive_link)
  
  Save to the `baseline/` directory.

---

### 4️⃣ Training

#### Training on DIOR Dataset (with Corruption)

```bash
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py \
  --cfg-options corrupt="cloudy"
```

#### Training on RSAR Dataset

```bash
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py \
  --cfg-options corrupt="cloudy"
```

> **Note:** To retrain the Oriented-RCNN baseline model, please refer to [mmrotate](https://github.com/open-mmlab/mmrotate).

---

### 5️⃣ Testing

#### Testing on DIOR Dataset (with Corruption)

```bash
python test.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py \
  work_dirs/unbiased_teacher_oriented_rcnn_selftraining_cga/latest.pth \
  --eval mAP \
  --cfg-options corrupt="cloudy"
```

#### Testing on RSAR Dataset

```bash
python test.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftaining_cga_rsar.py \
  work_dirs/unbiased_teacher_oriented_rcnn_selftaining_cga_rsar/latest.pth \
  --eval mAP \
  --show-dir vis_rsar
```

---

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


