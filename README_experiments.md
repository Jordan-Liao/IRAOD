# Experiments (Reproducibility Notes)

本仓库的实验组织目标是：
- 同一套入口（`train.py`/`test.py`）可在 DIOR/RSAR 上跑通；
- 所有实验产物落到 `work_dirs/`；
- `work_dirs/results/` 下可一键汇总指标、生成图表与定性对比样例。

## 0. 环境

本仓库默认使用 conda 环境 `iraod`（环境安装见 `README.md` / `scripts/setup_env_iraod.sh`）。

```bash
conda run -n iraod python tools/env_snapshot.py --out-dir docs
```

## 1. 数据

数据目录固定为：
- `dataset/DIOR`
- `dataset/RSAR`

校验命令：
```bash
conda run -n iraod python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR
```

如需自动下载/整理（需先将数据转存到你的 `/apps/bypy`，或使用 RSAR 的 Google Drive 源）：
```bash
conda run -n iraod python tools/download_datasets.py dior --source bypy --bypy-remote <REMOTE_DIR>
conda run -n iraod python tools/download_datasets.py rsar --source gdrive
```

## 2. 训练与评估脚本（核心实验）

DIOR：
- Baseline（监督）评估（clean + corruption）：`bash scripts/exp_dior_baseline_eval.sh`
- UT（无 CGA）：`bash scripts/exp_dior_ut.sh`
- UT + CGA(CLIP)：`bash scripts/exp_dior_ut_cga_clip.sh`

RSAR：
- Baseline（监督）：`bash scripts/exp_rsar_baseline.sh`
- UT（可通过 `CGA_SCORER=none|clip` 控制）：`bash scripts/exp_rsar_ut.sh`

说明：
- 所有脚本默认 `SMOKE=1` 并使用子集；调大 `N/N_TRAIN/N_TEST/MAX_EPOCHS` 即可扩充规模。
- `SAMPLES_PER_GPU` / `WORKERS_PER_GPU` 可通过环境变量覆盖（见各 `scripts/exp_*.sh`）。

## 3. 结果汇总与可视化

一键刷新指标/实验追踪表（推荐）：
```bash
bash scripts/refresh_results.sh
```

等价的手动命令（可自定义 work-dir 范围；默认不包含 `exp_rsar_severity` 这类“评估套件”目录）：
```bash
conda run -n iraod python tools/export_metrics.py \
  --work-dirs work_dirs/exp_smoke_* work_dirs/exp_dior_* work_dirs/exp_rsar_baseline* work_dirs/exp_rsar_ut* \
  --out-csv work_dirs/results/metrics.csv
conda run -n iraod python tools/ablation_table.py --csv work_dirs/results/metrics.csv --out-md work_dirs/results/ablation_table.md
conda run -n iraod python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv
```

生成图表：
```bash
conda run -n iraod python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob 'work_dirs/exp_*/*.log.json' --out-dir work_dirs/results/plots
```

定性对比抽样（基于 `--show-dir` 的输出）：
```bash
conda run -n iraod python tools/vis_random_samples.py --vis-dirs \
  work_dirs/exp_dior_baseline_eval/vis_clean \
  work_dirs/exp_dior_ut/vis_clean \
  work_dirs/exp_dior_ut_cga_clip/vis_clean \
  --num 8 --out-dir work_dirs/results/vis_compare/dior_clean
```

说明：
- `docs/experiment.md` 是“逐实验”的权威记录（cmd/log/artifacts/results）；`work_dirs/results/metrics.csv` 与 `experiments.csv` 是从产物自动汇总的表格视图。
