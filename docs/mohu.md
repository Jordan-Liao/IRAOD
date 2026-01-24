# Mohu

## 1) Not Implemented
## 2) Ambiguities
## Resolved (archive)
- [x] M0028 (plan: P0024): RSAR-Interference(interf_jamA) 已落盘真实扰动图像并通过校验
  - Evidence: `dataset/RSAR/{train,val,test}/images-interf_jamA/`（真实目录，文件数与 clean 一致）；`work_dirs/sanity/rsar_corrupt_switch/*_corrupt-interf_jamA.csv`（missing=0/conflict=0）；`bash scripts/prepare_rsar_interf_jamA.sh` 输出 `mean_abs_diff>0` 且 `identical=0`（抽样对比 clean != interfered）

- [x] M0027 (plan: P0023): RSAR UT+CGA teacher-init 尚未在 N_TEST=1000 上重新评估并刷新汇总
  - Evidence: 已新增 E0017 并完成评估：full mAP=0.2539（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_074641.json`）；同时刷新 `work_dirs/results/metrics.csv` 与 `experiments.csv`

- [x] M0026 (plan: P0022): SARCLIP 在 torch 1.7.1 下仍可能因 `batch_first` 报错
  - Evidence: 已在 `third_party/SARCLIP/sar_clip/transformer.py` 添加 MultiheadAttentionCompat，并补齐 torch1.7 依赖兼容；运行 `bash scripts/sarclip_torch17_smoke.sh` 生成 `work_dirs/sanity/sarclip_smoke_torch171.log`

- [x] M0025 (plan: P0021): 缺少 `notes/dataloader_extension_audit.md`（RSAR 多后缀解析审计记录）
  - Evidence: 已新增 `notes/dataloader_extension_audit.md`

- [x] M0024 (plan: P0020): 指标/图表/追踪表未纳入 teacher-init 实验
  - Evidence: 已刷新 `work_dirs/results/metrics.csv`（rows=17）与 `work_dirs/results/ablation_table.md`，并生成 `work_dirs/results/plots/`；同时更新 `experiments.csv`（rows=17）

- [x] M0023 (plan: P0019): RSAR UT+CGA(CLIP) teacher-init 实验未跑通/未产出 mAP
  - Evidence: 已支持 `TEACHER_CKPT` 并完成 smoke/full：smoke mAP=0.2790（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_070757.json`）；full mAP=0.3094（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_071054.json`）

- [x] M0022 (plan: P0018): RSAR UT teacher-init 支持未落盘（脚本缺少 `TEACHER_CKPT`）
  - Evidence: 已在 `scripts/exp_rsar_ut.sh` 支持 `TEACHER_CKPT`；并完成 smoke/full：smoke mAP=0.2836（`work_dirs/exp_rsar_ut_nocga_tinit/eval_20260121_063452.json`）；full mAP=0.1285（`work_dirs/exp_rsar_ut_nocga_tinit/eval_20260121_065058.json`）

- [x] M0021 (plan: P0017): `experiments.csv` / `README_experiments.md` / `MODEL_ZOO.md` 缺失
  - Evidence: 已新增 `tools/export_experiments.py` 并生成 `experiments.csv`（`conda run -n dino_sar python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv`）；同时补齐 `README_experiments.md` 与 `MODEL_ZOO.md`

- [x] M0020 (plan: P0016): `tools/plot_all.py` 缺失（mAP 图 + 训练曲线）
  - Evidence: 已新增 `tools/plot_all.py`；运行 `conda run -n dino_sar python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob 'work_dirs/exp_*/*.log.json' --out-dir work_dirs/results/plots` 生成 `work_dirs/results/plots/map_*.png` 与 `work_dirs/results/plots/curves/*.png`

- [x] M0019 (plan: P0015): `tools/vis_random_samples.py` 缺失（show-dir 抽样对比）
  - Evidence: 已新增 `tools/vis_random_samples.py`；运行 `conda run -n dino_sar python tools/vis_random_samples.py --vis-dirs work_dirs/exp_dior_baseline_eval/vis_clean work_dirs/exp_dior_ut/vis_clean work_dirs/exp_dior_ut_cga_clip/vis_clean --num 16 --out-dir work_dirs/results/vis_compare/dior_clean` 生成对比样例

- [x] M0018 (plan: P0014): 指标导出/对比汇总脚本缺失
  - Evidence: 已新增 `tools/export_metrics.py` 与 `tools/ablation_table.py`；运行 `conda run -n dino_sar python tools/export_metrics.py --work-dirs work_dirs/exp_* --out-csv work_dirs/results/metrics.csv` 生成 `work_dirs/results/metrics.csv`

- [x] M0017 (plan: P0013): RSAR supervised baseline config 与一键脚本缺失
  - Evidence: 已新增 `configs/experiments/rsar/baseline_oriented_rcnn_rsar.py` 与 `scripts/exp_rsar_baseline.sh`；运行 `bash scripts/exp_rsar_baseline.sh` 生成 `work_dirs/exp_rsar_baseline/latest.pth` 与 `work_dirs/exp_rsar_baseline/eval_20260121_021438.json`

- [x] M0016 (plan: P0012): CGA 后端切换（CLIP ↔ SARCLIP）与“失败不崩溃”尚未落盘
  - Evidence: 已修改 `sfod/cga.py` 与 `sfod/oriented_rcnn_cga.py`，并新增 `tools/cga_smoke.py`；运行 `conda run -n dino_sar python tools/cga_smoke.py --image dataset/RSAR/train/images/0000002.png --scorer clip --classes ship,aircraft,car,tank,bridge,harbor` 成功生成 `work_dirs/sanity/cga_smoke.json`

- [x] M0015 (plan: P0011): DIOR corruption 目录（clean/cloudy/brightness/contrast）未准备
  - Evidence: 已新增 `tools/prepare_dior_corruption.py` 并运行 `conda run -n dino_sar python tools/prepare_dior_corruption.py --data-root dataset/DIOR --corrupt clean cloudy brightness contrast --splits val,test --workers 8`，生成 `dataset/DIOR/Corruption/JPEGImages-clean|cloudy|brightness|contrast/`（ok=52803, failed=0）

- [x] M0014 (plan: P0009): CGA scorer 可插拔（CLIP ↔ SARCLIP）+ cache 尚未实现
  - Evidence: 已新增 `tools/cache_benchmark.py` 与 `sfod/scorers/`；运行 `conda run -n dino_sar python tools/cache_benchmark.py --scorer clip --image dataset/DIOR/JPEGImages/00001.jpg --prompt "an aerial image of airplane"`，生成 `work_dirs/sanity/cache_benchmark.json` 且第二次 run 命中 cache（hit=True）

- [x] M0011 (plan: P0008): SARCLIP 权重获取路径未定
  - Evidence: 统一权重目录为 `weights/sarclip/`（说明见 `weights/README.md`），`sfod/cga.py` 支持 `$SARCLIP_PRETRAINED` / `$SARCLIP_MODEL`；运行 `conda run -n dino_sar python tools/sarclip_smoke.py --image dataset/RSAR/train/images/0000002.png --prompts "an SAR image of ship"` 输出并写入 `work_dirs/sanity/sarclip_smoke.log`（无权重时给出下载与放置路径提示）

- [x] M0010 (plan: P0002): 数据下载源与自动化方式未定（bypy 无法直接处理 pan.baidu.com 分享链接）
  - Evidence: Decision 已写明（默认 Google Drive + `gdown`；bypy 需先转存到 `/apps/bypy`）；运行 `conda run -n dino_sar python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR` 返回 OK

- [x] M0009 (plan: P0010): RSAR-Interference 的 `corrupt` 切换校验脚本缺失
  - Evidence: 已新增 `tools/verify_rsar_corrupt_switch.py` 并运行 `conda run -n dino_sar python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA`，生成 `work_dirs/sanity/rsar_corrupt_switch/*_corrupt-*.csv` 且 missing=0/conflict=0（本机用 `dataset/RSAR/*/images-interf_jamA -> images` 软链做占位）

- [x] M0008 (plan: P0008): SARCLIP smoke 脚本缺失且 `sar_clip` 依赖未安装
  - Evidence: 已新增 `tools/sarclip_smoke.py` 并运行 `conda run -n dino_sar python tools/sarclip_smoke.py --image dataset/RSAR/train/images/0000002.png --prompts "an SAR image of ship"`，生成 `work_dirs/sanity/sarclip_smoke.log`（默认会自动 clone SARCLIP 到 `third_party/SARCLIP`；未提供权重时使用随机初始化并给出 warning）

- [x] M0007 (plan: P0007): RSAR smoke train/test 脚本缺失
  - Evidence: 已新增并运行 `bash scripts/smoke_rsar.sh`，生成 `work_dirs/exp_smoke_rsar/latest.pth`（log: `work_dirs/exp_smoke_rsar/20260121_003231.log`）且 `work_dirs/vis_rsar/` 生成可视化图片

- [x] M0001 (plan: P0001): 环境快照脚本与输出文件尚未落盘
  - Evidence: 已运行 `conda run -n dino_sar python tools/env_snapshot.py --out-dir docs`，生成 `docs/env_lock.txt` 与 `docs/system_info.md`

- [x] M0002 (plan: P0002): 数据目录校验脚本缺失
  - Evidence: 已新增 `tools/verify_dataset_layout.py` 与最小夹具 `tools/fixtures/`；运行 `conda run -n dino_sar python tools/verify_dataset_layout.py --dior tools/fixtures/DIOR --rsar tools/fixtures/RSAR` 返回 OK

- [x] M0012 (plan: P0002): 真实 RSAR 数据尚未落到 `dataset/RSAR`
  - Evidence: 已下载并解压 RSAR 到 `dataset/RSAR`；运行 `conda run -n dino_sar python tools/verify_dataset_layout.py --dior tools/fixtures/DIOR --rsar dataset/RSAR` 返回 OK

- [x] M0013 (plan: P0002): 真实 DIOR 数据尚未落到 `dataset/DIOR`
  - Evidence: 已下载并解压 DIOR 到 `dataset/DIOR`，并整理为 `JPEGImages/` + `ImageSets/` 结构；运行 `conda run -n dino_sar python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar tools/fixtures/RSAR` 返回 OK

- [x] M0003 (plan: P0003): DIOR sanity 脚本缺失
  - Evidence: 已新增 `tools/sanity_check_dior.py`；运行 `conda run -n dino_sar python tools/sanity_check_dior.py --data-root dataset/DIOR --split train --num 20 --out-dir work_dirs/sanity/dior_vis`，生成 `work_dirs/sanity/dior_sanity_report.json` 与可视化目录

- [x] M0004 (plan: P0004): DIOR smoke 脚本缺失（小样本闭环）
  - Evidence: 已新增 `scripts/smoke_dior.sh`；运行 `bash scripts/smoke_dior.sh` 生成 `work_dirs/exp_smoke_dior/latest.pth` 且日志输出 mAP

- [x] M0005 (plan: P0005): RSAR 对齐检查脚本缺失（missing/conflict 报告）
  - Evidence: 已新增 `tools/check_image_ann_alignment.py`；运行 `conda run -n dino_sar python tools/check_image_ann_alignment.py --ann-dir dataset/RSAR/train/annfiles --img-dir dataset/RSAR/train/images --out-csv work_dirs/sanity/rsar_alignment_train.csv`，输出 missing=0/conflict=0

- [x] M0006 (plan: P0006): RSAR sanity 脚本缺失
  - Evidence: 已新增 `tools/sanity_check_rsar.py`；运行 `conda run -n dino_sar python tools/sanity_check_rsar.py --data-root dataset/RSAR --split train --num 20 --out-dir work_dirs/sanity/rsar_vis`，生成 `work_dirs/sanity/rsar_sanity_report.json` 与可视化目录
