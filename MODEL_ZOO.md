# Model Zoo

说明：
- 本仓库的所有训练产物默认在 `work_dirs/`；
- 汇总指标见 `work_dirs/results/metrics.csv` 与 `work_dirs/results/ablation_table.md`；
- 下表列出当前已产出的主要 checkpoint（以 `latest.pth` 为主）。

## Checkpoints

| Dataset | Method | Checkpoint | Notes |
| --- | --- | --- | --- |
| DIOR | Baseline (supervised) | `baseline/baseline.pth` | README 提供的预训练权重；对应评估输出在 `work_dirs/exp_dior_baseline_eval/` |
| DIOR | UT | `work_dirs/exp_dior_ut/latest.pth` | 训练/评估/可视化：`work_dirs/exp_dior_ut/` |
| DIOR | UT + CGA(CLIP) | `work_dirs/exp_dior_ut_cga_clip/latest.pth` | 训练/评估/可视化：`work_dirs/exp_dior_ut_cga_clip/` |
| RSAR | Baseline (supervised) | `work_dirs/exp_rsar_baseline/latest.pth` | 训练/评估：`work_dirs/exp_rsar_baseline/`；可视化：`work_dirs/vis_rsar_baseline/` |
| RSAR | UT (no CGA) | `work_dirs/exp_rsar_ut_nocga/latest.pth` | 训练/评估：`work_dirs/exp_rsar_ut_nocga/`；可视化：`work_dirs/vis_rsar_ut_nocga/` |
| RSAR | UT + CGA(CLIP) | `work_dirs/exp_rsar_ut_cga_clip/latest.pth` | 训练/评估：`work_dirs/exp_rsar_ut_cga_clip/`；可视化：`work_dirs/vis_rsar_ut_cga_clip/` |
| RSAR | SFOD-RS clean source (Phase 5-8) | `work_dirs/rsar_sfodrs_full_fix_20260424_172627/source_train/latest.pth` | clean test mAP=0.5385；direct mean 7corr=0.3372 |
| RSAR | SFOD-RS corr-aug source (Phase 9) | `work_dirs/rsar_corraug_loose_20260504/source_train/latest.pth` | clean test mAP=0.5125；direct mean 7corr=0.4742（+40.6% vs clean source）；`RSAR_CORR_AUG=1 RSAR_CORR_AUG_PROB=0.5` |

## How to reproduce

复现入口见 `README_experiments.md`。

