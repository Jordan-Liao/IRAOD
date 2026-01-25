# RSAR 多后缀图片解析审计记录（jpg/png/bmp…）

本 note 对应根目录 `plan.md` 的 Milestone 2 / Step 2.1–2.2：确认并落实“annfile -> image”解析对图片后缀无关（jpg/png/bmp/tif…），并给出可重复的验证命令。

## 1. 结论（当前仓库的实现点）

- RSAR 的“按 annfile 基名找到真实图片文件”的兼容逻辑集中在：`sfod/semi_dota_dataset.py` 的 `DOTADatasetAnySuffix.load_annotations()`。
- 全量对齐检查与缺失/冲突报告在：`tools/check_image_ann_alignment.py`。

## 2. 实现策略（工程实现）

### 2.1 一次性扫描建立索引（优先）

`DOTADatasetAnySuffix.load_annotations()` 会在 `img_prefix` 下扫描允许后缀：

- 允许后缀：`.jpg/.jpeg/.png/.bmp/.tif/.tiff`
- 建立 `stem -> relative_path` 的映射（stem 为不含扩展名的文件名）
- 将 `data_infos[i]["filename"]` 回写为真实存在的相对路径（不依赖标注里写死的后缀）

### 2.2 glob 兜底（避免漏网）

当索引未命中时，会尝试：

- `glob(img_dir/<base>.*)` 与 `glob(img_dir/<stem>.*)`
- 使用简单优先级（默认更偏向 `.jpg`）：见 `sfod/semi_dota_dataset.py::_pick_by_priority()`

## 3. 冲突与脏数据处理

- 若同名多后缀同时存在（例如同 stem 的 `.jpg` 与 `.png`），原则上应视为数据冲突并在对齐检查阶段暴露。
- 当前训练侧在索引命中时使用“首次遇到的文件”作为默认命中；因此建议在数据准备阶段保证 `conflict=0`。

## 4. 验证命令（必须可跑）

### 4.1 全量对齐检查（missing/conflict）

```bash
conda run -n iraod python tools/check_image_ann_alignment.py \
  --ann-dir dataset/RSAR/train/annfiles \
  --img-dir dataset/RSAR/train/images \
  --out-csv work_dirs/sanity/rsar_alignment_train.csv
```

### 4.2 dataloader 抽样可视化 sanity

```bash
conda run -n iraod python tools/sanity_check_rsar.py \
  --data-root dataset/RSAR --split train --num 20 \
  --out-dir work_dirs/sanity/rsar_vis
```

## 5. RSAR-Interference（扩展提示）

- 干扰版本建议按 `images-<corrupt>` 目录组织；本仓库验证脚本：`tools/verify_rsar_corrupt_switch.py`。
- 只要干扰图与 clean 保持同名（stem 一致），上述解析机制不依赖后缀即可直接工作。
