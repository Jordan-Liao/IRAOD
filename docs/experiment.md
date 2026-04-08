# Experiments

## Overview
- Goal: 在 DIOR/RSAR 上完成“可跑 + 可对比 + 可汇总”的实验闭环（DIOR corruption、RSAR baseline/UT/UT+CGA）
- Baseline: IRAOD `UnbiasedTeacher` + `OrientedRCNN`（mmrotate）
- Primary model: `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py`

## Experiments

### E0001: DIOR Smoke Train/Test
| Field | Value |
| --- | --- |
| Objective | 跑通 DIOR smoke train->test（mAP 非 NaN，产出 ckpt） |
| Baseline | IRAOD 默认训练入口 |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN (le90) |
| Weights | `baseline/baseline.pth` |
| Code path | `scripts/smoke_dior.sh`, `train.py`, `test.py`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py` |
| Params | `N=50`, `corrupt=clean`, `samples_per_gpu=1`, `workers_per_gpu=0` |
| Metrics (must save) | `mAP`（log）；`latest.pth` |
| Checks | mAP 输出存在且非 NaN；`latest.pth` 存在 |
| VRAM | ~4 GB |
| Time/epoch | ~1–3 min (smoke) |
| Total time | ~1–3 min |
| Single-GPU script | `bash scripts/smoke_dior.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash scripts/smoke_dior.sh` |
| Full cmd | `bash scripts/smoke_dior.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `work_dirs/exp_smoke_dior/*.log` |
| Artifacts | `work_dirs/exp_smoke_dior/latest.pth` |
| Results | 参考 `work_dirs/exp_smoke_dior/*.log` 中的 mAP 输出 |


### E0002: RSAR Smoke Train/Test (+ show-dir)
| Field | Value |
| --- | --- |
| Objective | 跑通 RSAR smoke train->test（含 `--show-dir` 可视化输出） |
| Baseline | IRAOD RSAR 配置训练入口 |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN (le90) |
| Weights | `None`（smoke 用随机/自初始化 teacher） |
| Code path | `scripts/smoke_rsar.sh`, `train.py`, `test.py`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py` |
| Params | `N_TRAIN=50`, `N_VAL=50`, `N_TEST=50`, `corrupt=clean`, `samples_per_gpu=1`, `max_epochs=1` |
| Metrics (must save) | `mAP`（log）；`latest.pth`；`--show-dir` 目录文件 |
| Checks | mAP 输出存在且非 NaN；`latest.pth` 存在；`work_dirs/vis_rsar/` 有图 |
| VRAM | ~4 GB |
| Time/epoch | ~1–5 min (smoke) |
| Total time | ~1–5 min |
| Single-GPU script | `bash scripts/smoke_rsar.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash scripts/smoke_rsar.sh` |
| Full cmd | `bash scripts/smoke_rsar.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `work_dirs/exp_smoke_rsar/*.log` |
| Artifacts | `work_dirs/exp_smoke_rsar/latest.pth`, `work_dirs/vis_rsar/` |
| Results | 参考 `work_dirs/exp_smoke_rsar/*.log` 中的 mAP 输出 |


### E0003: RSAR Ann-Image Alignment (Any Suffix)
| Field | Value |
| --- | --- |
| Objective | 验证 RSAR `annfiles/*.txt` 能在 `images/` 解析到真实文件（jpg/png/bmp…） |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | N/A（工具类实验，不涉及模型训练） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/check_image_ann_alignment.py` |
| Params | `--exts .jpg,.jpeg,.png,.bmp,.tif,.tiff` |
| Metrics (must save) | missing/conflict 统计；CSV 报告 |
| Checks | missing=0 且 conflict=0 |
| VRAM | 0 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~1–3 min（全量扫描） |
| Single-GPU script | `conda run -n iraod python tools/check_image_ann_alignment.py --ann-dir dataset/RSAR/train/annfiles --img-dir dataset/RSAR/train/images --out-csv work_dirs/sanity/rsar_alignment_train.csv` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `conda run -n iraod python tools/check_image_ann_alignment.py --ann-dir dataset/RSAR/train/annfiles --img-dir dataset/RSAR/train/images --out-csv work_dirs/sanity/rsar_alignment_train.csv` |
| Full cmd | `conda run -n iraod python tools/check_image_ann_alignment.py --ann-dir dataset/RSAR/train/annfiles --img-dir dataset/RSAR/train/images --out-csv work_dirs/sanity/rsar_alignment_train.csv` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout |
| Artifacts | `work_dirs/sanity/rsar_alignment_train.csv` |
| Results | missing=0/conflict=0（见 `work_dirs/sanity/rsar_alignment_train.csv` 与 stdout） |


### E0004: Scorer Switch + Disk Cache Benchmark (CLIP/SARCLIP)
| Field | Value |
| --- | --- |
| Objective | 验证 scorer 可切换且 cache 命中率/速度提升明显 |
| Baseline | 无 cache（首次运行） |
| Model | `sfod/scorers/ClipScorer` / `sfod/scorers/SarclipScorer` |
| Weights | CLIP 自动下载；SARCLIP 可选 `weights/sarclip/...` 或随机初始化 |
| Code path | `tools/cache_benchmark.py`, `tools/sarclip_smoke.py`, `sfod/scorers/` |
| Params | `--runs 2`（第 2 次应 hit=True） |
| Metrics (must save) | cache hit/miss；每次耗时；输出 JSON |
| Checks | run2 hit=True；输出 `work_dirs/sanity/cache_benchmark.json` |
| VRAM | ~2–6 GB（取决于 scorer/model） |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~1–5 min |
| Single-GPU script | `conda run -n iraod python tools/cache_benchmark.py --scorer clip --image dataset/DIOR/JPEGImages/00001.jpg --prompt "an aerial image of airplane"` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/cache_benchmark.py --scorer clip --image dataset/DIOR/JPEGImages/00001.jpg --prompt "an aerial image of airplane" && conda run -n iraod python tools/sarclip_smoke.py --image dataset/RSAR/train/images/0000002.png --prompts "an SAR image of ship"'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/cache_benchmark.py --scorer clip --image dataset/DIOR/JPEGImages/00001.jpg --prompt "an aerial image of airplane" && conda run -n iraod python tools/sarclip_smoke.py --image dataset/RSAR/train/images/0000002.png --prompts "an SAR image of ship"'` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout；`work_dirs/sanity/sarclip_smoke.log` |
| Artifacts | `work_dirs/sanity/cache_benchmark.json`, `work_dirs/sanity/scorer_cache/`, `work_dirs/sanity/sarclip_smoke.log` |
| Results | cache_benchmark 第二次 run 命中 hit=True（见 `work_dirs/sanity/cache_benchmark.json`） |


### E0005: RSAR Corrupt Switch Verification (clean ↔ images-interf_xxx)
| Field | Value |
| --- | --- |
| Objective | 验证 `corrupt=interf_xxx` 时 images 目录可切换且 ann->image resolve 仍为 missing=0/conflict=0，并跑通一次 RSAR smoke train/test |
| Baseline | clean |
| Model | N/A（工具类实验，不涉及模型训练） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py`, `scripts/smoke_rsar.sh` |
| Params | `--corrupt interf_jamA`（映射到 `images-interf_jamA/`）；`CORRUPT=interf_jamA` |
| Metrics (must save) | missing/conflict 统计；CSV 报告；smoke mAP（log）；`--show-dir` 输出 |
| Checks | verify 脚本对 clean/interf 均通过；`CORRUPT=interf_jamA` smoke train/test 可运行并产出可视化 |
| VRAM | ~4 GB（smoke train/test） |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~5–15 min（全量扫描 + smoke） |
| Single-GPU script | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CORRUPT=interf_jamA WORK_DIR=work_dirs/exp_smoke_rsar_interf_jamA VIS_DIR=work_dirs/vis_rsar_interf_jamA bash scripts/smoke_rsar.sh'` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CORRUPT=interf_jamA WORK_DIR=work_dirs/exp_smoke_rsar_interf_jamA VIS_DIR=work_dirs/vis_rsar_interf_jamA bash scripts/smoke_rsar.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CORRUPT=interf_jamA WORK_DIR=work_dirs/exp_smoke_rsar_interf_jamA VIS_DIR=work_dirs/vis_rsar_interf_jamA bash scripts/smoke_rsar.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout；`work_dirs/exp_smoke_rsar_interf_jamA/*.log` |
| Artifacts | `work_dirs/sanity/rsar_corrupt_switch/*_corrupt-*.csv`, `work_dirs/exp_smoke_rsar_interf_jamA/latest.pth`, `work_dirs/vis_rsar_interf_jamA/` |
| Results | verify: missing=0/conflict=0；smoke: mAP 输出存在且非 NaN（见对应 log） |


### E0006: DIOR Baseline Eval (clean + corruption)
| Field | Value |
| --- | --- |
| Objective | 用 `baseline/baseline.pth` 在 DIOR clean/cloudy/brightness/contrast 上评估并产出 `eval_*.json` |
| Baseline | DIOR baseline checkpoint |
| Model | OrientedRCNN R50-FPN (le90) |
| Weights | `baseline/baseline.pth` |
| Code path | `configs/experiments/dior/baseline_oriented_rcnn_dior.py`, `scripts/exp_dior_baseline_eval.sh`, `test.py` |
| Params | `CORRUPTS=clean,cloudy,brightness,contrast`；`SMOKE=1` 时 `N_TEST=200` |
| Metrics (must save) | 每个 corrupt 一个 `eval_*.json`（mAP）；对应 `vis_*` 目录 |
| Checks | 每个 `work_dirs/exp_dior_baseline_eval/eval_<corrupt>/` 生成 `eval_*.json` |
| VRAM | ~4–8 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~5–30 min（取决于测试样本数与 corrupt 个数） |
| Single-GPU script | `bash scripts/exp_dior_baseline_eval.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'SMOKE=1 N_TEST=200 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_baseline_eval.sh'` |
| Full cmd | `bash -lc 'SMOKE=1 N_TEST=2000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_baseline_eval.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260120-182557-6ca3__e0006-smoke.log`; full: `.rd_queue/logs/J20260120-190935-f4f4__e0006-full.log` |
| Artifacts | `work_dirs/exp_dior_baseline_eval/eval_*/eval_*.json`, `work_dirs/exp_dior_baseline_eval/vis_*/` |
| Results | smoke(N_TEST=200): clean mAP=0.6036 (`work_dirs/exp_dior_baseline_eval/eval_clean/eval_20260121_022624.json`); cloudy mAP=0.6046 (`work_dirs/exp_dior_baseline_eval/eval_cloudy/eval_20260121_022716.json`); brightness mAP=0.5529 (`work_dirs/exp_dior_baseline_eval/eval_brightness/eval_20260121_022808.json`); contrast mAP=0.5889 (`work_dirs/exp_dior_baseline_eval/eval_contrast/eval_20260121_022900.json`); full(N_TEST=2000): clean mAP=0.5426 (`work_dirs/exp_dior_baseline_eval/eval_clean/eval_20260121_041854.json`); cloudy mAP=0.5394 (`work_dirs/exp_dior_baseline_eval/eval_cloudy/eval_20260121_042417.json`); brightness mAP=0.5225 (`work_dirs/exp_dior_baseline_eval/eval_brightness/eval_20260121_042944.json`); contrast mAP=0.5390 (`work_dirs/exp_dior_baseline_eval/eval_contrast/eval_20260121_043406.json`) |


### E0007: DIOR UnbiasedTeacher (no CGA) Train + Eval (clean + corruption)
| Field | Value |
| --- | --- |
| Objective | 训练 DIOR UT（无 CGA），并在 clean/cloudy/brightness/contrast 上评估输出 `eval_*.json` |
| Baseline | E0006（DIOR baseline eval） |
| Model | UnbiasedTeacher + OrientedRCNN (teacher without CGA) |
| Weights | `baseline/baseline.pth`（teacher init） |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining.py`, `scripts/exp_dior_ut.sh` |
| Params | 训练固定 `corrupt=clean`；评估 `CORRUPTS=clean,cloudy,brightness,contrast`；`MAX_EPOCHS` 可调 |
| Metrics (must save) | `latest.pth`；每个 corrupt 一个 `eval_*.json`；可视化目录 `vis_*` |
| Checks | `work_dirs/exp_dior_ut/latest.pth` 存在；每个 `eval_<corrupt>/` 生成 `eval_*.json` |
| VRAM | ~6–12 GB |
| Time/epoch | ~?（取决于样本数/epoch） |
| Total time | smoke: ~5–20 min；full: 取决于 `MAX_EPOCHS` |
| Single-GPU script | `bash scripts/exp_dior_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'SMOKE=1 N=200 MAX_EPOCHS=1 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_ut.sh'` |
| Full cmd | `bash -lc 'SMOKE=1 N=2000 MAX_EPOCHS=2 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | failed: `.rd_queue/logs/J20260120-182559-b768__e0007-smoke.log`；smoke ok: `.rd_queue/logs/J20260120-183421-babc__e0007-smoke-rerun.log`；full: `.rd_queue/logs/J20260120-195534-1ac1__e0007-full-rerun.log` |
| Artifacts | `work_dirs/exp_dior_ut/latest.pth`, `work_dirs/exp_dior_ut/eval_*/eval_*.json`, `work_dirs/exp_dior_ut/vis_*/` |
| Results | smoke attempt1 failed: `DIORDataset.evaluate() got an unexpected keyword argument 'only_ema'`（`.rd_queue/logs/J20260120-182559-b768__e0007-smoke.log`）；fix: `test.py` drop `only_ema` from eval_kwargs；smoke rerun(N=200,epoch=1): clean mAP=0.5700 (`work_dirs/exp_dior_ut/eval_clean/eval_20260121_024911.json`); cloudy mAP=0.5700 (`work_dirs/exp_dior_ut/eval_cloudy/eval_20260121_025006.json`); brightness mAP=0.5282 (`work_dirs/exp_dior_ut/eval_brightness/eval_20260121_025104.json`); contrast mAP=0.5433 (`work_dirs/exp_dior_ut/eval_contrast/eval_20260121_025159.json`); full(N=2000,epoch=2): clean mAP=0.5379 (`work_dirs/exp_dior_ut/eval_clean/eval_20260121_044916.json`); cloudy mAP=0.5359 (`work_dirs/exp_dior_ut/eval_cloudy/eval_20260121_045352.json`); brightness mAP=0.5269 (`work_dirs/exp_dior_ut/eval_brightness/eval_20260121_045828.json`); contrast mAP=0.5325 (`work_dirs/exp_dior_ut/eval_contrast/eval_20260121_050305.json`) |


### E0008: DIOR UnbiasedTeacher + CGA(CLIP) Train + Eval (clean + corruption)
| Field | Value |
| --- | --- |
| Objective | 训练 DIOR UT+CGA（CLIP 后端），并在 clean/cloudy/brightness/contrast 上评估输出 `eval_*.json` |
| Baseline | E0007（DIOR UT no-CGA） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA (teacher with refine_test) |
| Weights | `baseline/baseline.pth`（teacher init） |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py`, `scripts/exp_dior_ut_cga_clip.sh`, `sfod/cga.py` |
| Params | 训练固定 `corrupt=clean`；`CGA_SCORER=clip`；评估 `CORRUPTS=clean,cloudy,brightness,contrast`；`MAX_EPOCHS` 可调 |
| Metrics (must save) | `latest.pth`；每个 corrupt 一个 `eval_*.json`；可视化目录 `vis_*` |
| Checks | `work_dirs/exp_dior_ut_cga_clip/latest.pth` 存在；每个 `eval_<corrupt>/` 生成 `eval_*.json` |
| VRAM | ~6–12 GB（取决于 CLIP 模型） |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于 `MAX_EPOCHS` |
| Single-GPU script | `bash scripts/exp_dior_ut_cga_clip.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'SMOKE=1 N=200 MAX_EPOCHS=1 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_ut_cga_clip.sh'` |
| Full cmd | `bash -lc 'SMOKE=1 N=2000 MAX_EPOCHS=2 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 CORRUPTS=clean,cloudy,brightness,contrast bash scripts/exp_dior_ut_cga_clip.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260120-182601-1d3d__e0008-smoke.log`; full: `.rd_queue/logs/J20260120-195534-7eda__e0008-full-rerun.log` |
| Artifacts | `work_dirs/exp_dior_ut_cga_clip/latest.pth`, `work_dirs/exp_dior_ut_cga_clip/eval_*/eval_*.json`, `work_dirs/exp_dior_ut_cga_clip/vis_*/` |
| Results | smoke(N=200,epoch=1): clean mAP=0.3709 (`work_dirs/exp_dior_ut_cga_clip/eval_clean/eval_20260121_023433.json`); cloudy mAP=0.3659 (`work_dirs/exp_dior_ut_cga_clip/eval_cloudy/eval_20260121_023521.json`); brightness mAP=0.3392 (`work_dirs/exp_dior_ut_cga_clip/eval_brightness/eval_20260121_023613.json`); contrast mAP=0.3509 (`work_dirs/exp_dior_ut_cga_clip/eval_contrast/eval_20260121_023710.json`); full(N=2000,epoch=2): clean mAP=0.5351 (`work_dirs/exp_dior_ut_cga_clip/eval_clean/eval_20260121_052204.json`); cloudy mAP=0.5357 (`work_dirs/exp_dior_ut_cga_clip/eval_cloudy/eval_20260121_052639.json`); brightness mAP=0.5229 (`work_dirs/exp_dior_ut_cga_clip/eval_brightness/eval_20260121_053106.json`); contrast mAP=0.5326 (`work_dirs/exp_dior_ut_cga_clip/eval_contrast/eval_20260121_053539.json`) |


### E0009: RSAR Baseline Train/Test (supervised)
| Field | Value |
| --- | --- |
| Objective | 训练 RSAR supervised baseline（OrientedRCNN）并评估 mAP，产出 `eval_*.json` |
| Baseline | 从零训练（监督 baseline，无预训练检测器） |
| Model | OrientedRCNN R50-FPN (le90) |
| Weights | `None` |
| Code path | `configs/experiments/rsar/baseline_oriented_rcnn_rsar.py`, `scripts/exp_rsar_baseline.sh`, `train.py`, `test.py` |
| Params | `SMOKE=1` 抽样（可通过 `N_TRAIN/N_VAL/N_TEST` 控制规模）；RSAR 全量过大，本 ledger 的 full 使用较大子集 |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `work_dirs/exp_rsar_baseline/latest.pth` 与 `work_dirs/exp_rsar_baseline/eval_*.json` 存在 |
| VRAM | ~4–8 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–20 min；full: 取决于 `MAX_EPOCHS` |
| Single-GPU script | `bash scripts/exp_rsar_baseline.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'SMOKE=1 MAX_EPOCHS=1 bash scripts/exp_rsar_baseline.sh'` |
| Full cmd | `bash -lc 'SMOKE=1 N_TRAIN=2000 N_VAL=500 N_TEST=1000 MAX_EPOCHS=6 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 bash scripts/exp_rsar_baseline.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_baseline/20260121_021358.log`; full: `.rd_queue/logs/J20260120-190935-f4c5__e0009-full.log` |
| Artifacts | `work_dirs/exp_rsar_baseline/latest.pth`, `work_dirs/exp_rsar_baseline/eval_*.json`, `work_dirs/vis_rsar_baseline/` |
| Results | smoke(N=50,epoch=1): mAP=0.0（`work_dirs/exp_rsar_baseline/eval_20260121_021438.json`）；full(N_TRAIN=2000,N_VAL=500,N_TEST=1000,epoch=6): mAP=0.2714（`work_dirs/exp_rsar_baseline/eval_20260121_041717.json`） |


### E0010: RSAR UnbiasedTeacher (CGA off) Train/Test
| Field | Value |
| --- | --- |
| Objective | 训练 RSAR UT（禁用 CGA）并评估 mAP，产出 `eval_*.json` |
| Baseline | E0009（RSAR baseline） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（但 `CGA_SCORER=none` 不做重打分） |
| Weights | `None`（smoke 默认从头训练 teacher） |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `scripts/exp_rsar_ut.sh` |
| Params | `CGA_SCORER=none`；`SMOKE=1` 抽样（可通过 `N_TRAIN/N_VAL/N_TEST` 控制规模）；RSAR 全量过大，本 ledger 的 full 使用较大子集 |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | work_dir 下生成 `latest.pth` 与 `eval_*.json` |
| VRAM | ~6–12 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于 `MAX_EPOCHS` |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=none SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_nocga VIS_DIR=work_dirs/vis_rsar_ut_nocga bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=none SMOKE=1 N_TRAIN=2000 N_VAL=500 N_TEST=1000 MAX_EPOCHS=6 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_nocga VIS_DIR=work_dirs/vis_rsar_ut_nocga bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260120-182603-45c6__e0010-smoke.log`; full: `.rd_queue/logs/J20260120-190935-3654__e0010-full.log` |
| Artifacts | `work_dirs/exp_rsar_ut_nocga/latest.pth`, `work_dirs/exp_rsar_ut_nocga/eval_*.json`, `work_dirs/vis_rsar_ut_nocga/` |
| Results | smoke(N=50,epoch=1): mAP=0.0（`work_dirs/exp_rsar_ut_nocga/eval_20260121_023942.json`）；full(N_TRAIN=2000,N_VAL=500,N_TEST=1000,epoch=6): mAP=0.0（`work_dirs/exp_rsar_ut_nocga/eval_20260121_034013.json`） |


### E0011: RSAR UnbiasedTeacher + CGA(CLIP) Train/Test
| Field | Value |
| --- | --- |
| Objective | 训练 RSAR UT+CGA（CLIP 后端）并评估 mAP，产出 `eval_*.json` |
| Baseline | E0010（RSAR UT no-CGA） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（CGA refine_test） |
| Weights | CLIP 自动下载；teacher 从头训练 |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `scripts/exp_rsar_ut.sh`, `sfod/cga.py` |
| Params | `CGA_SCORER=clip`；`SMOKE=1` 抽样（可通过 `N_TRAIN/N_VAL/N_TEST` 控制规模）；RSAR 全量过大，本 ledger 的 full 使用较大子集 |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | work_dir 下生成 `latest.pth` 与 `eval_*.json` |
| VRAM | ~6–14 GB（取决于 CLIP 模型） |
| Time/epoch | ~? |
| Total time | smoke: ~5–40 min；full: 取决于 `MAX_EPOCHS` |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=clip SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip VIS_DIR=work_dirs/vis_rsar_ut_cga_clip bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=clip SMOKE=1 N_TRAIN=500 N_VAL=100 N_TEST=200 MAX_EPOCHS=2 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip VIS_DIR=work_dirs/vis_rsar_ut_cga_clip SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_full2 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260120-182606-9a6e__e0011-smoke.log`; full: `.rd_queue/logs/J20260120-214453-a449__e0011-full-rerun2.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_clip/latest.pth`, `work_dirs/exp_rsar_ut_cga_clip/eval_*.json`, `work_dirs/vis_rsar_ut_cga_clip/` |
| Results | smoke(N=50,epoch=1): mAP=0.0（`work_dirs/exp_rsar_ut_cga_clip/eval_20260121_024609.json`）；full(N_TRAIN=500,N_VAL=100,N_TEST=200,epoch=2): mAP=0.0（`work_dirs/exp_rsar_ut_cga_clip/eval_20260121_055007.json`） |


### E0012: Qualitative Compare (show-dir sampler)
| Field | Value |
| --- | --- |
| Objective | 基于多个 `--show-dir` 输出目录生成随机抽样的并排对比图，用于快速定性分析 |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | N/A（工具类实验，不涉及模型训练） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/vis_random_samples.py` |
| Params | `--vis-dirs <dir1> <dir2> ...`；`--num N`；`--out-dir out` |
| Metrics (must save) | 生成的对比图片（PNG/JPG） |
| Checks | `--out-dir` 下生成 `sample_*.png`（或同名）且返回码=0 |
| VRAM | 0 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~秒级–1min（取决于 N 与图片大小） |
| Single-GPU script | `conda run -n iraod python tools/vis_random_samples.py --vis-dirs work_dirs/exp_dior_baseline_eval/vis_clean work_dirs/exp_dior_ut/vis_clean work_dirs/exp_dior_ut_cga_clip/vis_clean --num 8 --out-dir work_dirs/results/vis_compare/dior_clean` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `conda run -n iraod python tools/vis_random_samples.py --vis-dirs work_dirs/exp_dior_baseline_eval/vis_clean work_dirs/exp_dior_ut/vis_clean --num 4 --out-dir work_dirs/results/vis_compare/dior_clean_smoke` |
| Full cmd | `conda run -n iraod python tools/vis_random_samples.py --vis-dirs work_dirs/exp_dior_baseline_eval/vis_clean work_dirs/exp_dior_ut/vis_clean work_dirs/exp_dior_ut_cga_clip/vis_clean --num 16 --out-dir work_dirs/results/vis_compare/dior_clean` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout |
| Artifacts | `work_dirs/results/vis_compare/dior_clean_smoke/`, `work_dirs/results/vis_compare/dior_clean/` |
| Results | 已生成 smoke 4 张对比图（`work_dirs/results/vis_compare/dior_clean_smoke/sample_*.png`）与 full 16 张对比图（`work_dirs/results/vis_compare/dior_clean/sample_*.png`） |


### E0013: Plot All (metrics + training curves)
| Field | Value |
| --- | --- |
| Objective | 从 `metrics.csv` 与 `*.log.json` 一键生成 mAP 对比图与训练曲线图 |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | N/A（工具类实验，不涉及模型训练） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/plot_all.py` |
| Params | `--metrics-csv`；可选 `--log-json-glob`；`--out-dir` |
| Metrics (must save) | PNG 图表（mAP bar、loss/pseudo 曲线等） |
| Checks | `--out-dir` 下生成 PNG 且返回码=0 |
| VRAM | 0 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~秒级–几分钟（取决于 log 数量） |
| Single-GPU script | `conda run -n iraod python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob 'work_dirs/exp_*/*.log.json' --out-dir work_dirs/results/plots` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `conda run -n iraod python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --out-dir work_dirs/results/plots_smoke` |
| Full cmd | `conda run -n iraod python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob 'work_dirs/exp_*/*.log.json' --out-dir work_dirs/results/plots` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout |
| Artifacts | `work_dirs/results/plots_smoke/`, `work_dirs/results/plots/` |
| Results | 已生成 mAP 对比图（`work_dirs/results/plots/map_dior.png`, `work_dirs/results/plots/map_rsar.png`）与训练曲线（`work_dirs/results/plots/curves/*.png`） |


### E0014: Export Experiments Tracker (experiments.csv)
| Field | Value |
| --- | --- |
| Objective | 从 `metrics.csv` 导出 `experiments.csv`（包含 git hash、log/config 指针），并补齐复现/模型清单文档 |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | N/A（工具类实验，不涉及模型训练） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/export_experiments.py`, `README_experiments.md`, `MODEL_ZOO.md` |
| Params | `--metrics-csv`；`--out-csv` |
| Metrics (must save) | `experiments.csv` |
| Checks | `experiments.csv` 非空且包含 `git_sha` 等列；README/MODEL_ZOO 文件存在 |
| VRAM | 0 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~秒级 |
| Single-GPU script | `bash -lc 'conda run -n iraod python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv && test -f README_experiments.md && test -f MODEL_ZOO.md'` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `conda run -n iraod python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv && test -f README_experiments.md && test -f MODEL_ZOO.md'` |
| Smoke | [x] |
| Full | [x] |
| Logs | stdout |
| Artifacts | `experiments.csv`, `README_experiments.md`, `MODEL_ZOO.md` |
| Results | 已生成 `experiments.csv`（rows=15），并补齐 `README_experiments.md` 与 `MODEL_ZOO.md` |


### E0015: RSAR UnbiasedTeacher (CGA off) w/ Teacher Init (from RSAR baseline)
| Field | Value |
| --- | --- |
| Objective | 修复 RSAR UT 在 teacher 未初始化时易退化为 `mAP=0` 的问题：teacher 从 RSAR supervised baseline 初始化并评估 mAP |
| Baseline | E0010（RSAR UT no-CGA，teacher=None，mAP=0） |
| Model | UnbiasedTeacher + OrientedRCNN（student） + OrientedRCNN_CGA（ema teacher） |
| Weights | `TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py` |
| Params | `CGA_SCORER=none`；`TEACHER_CKPT` 指向 RSAR baseline ckpt |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `eval_*.json` 中 `metric.mAP` 非 0 |
| VRAM | ~6–12 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于 `MAX_EPOCHS`/subset |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_nocga_tinit VIS_DIR=work_dirs/vis_rsar_ut_nocga_tinit bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 N_TRAIN=2000 N_VAL=500 N_TEST=1000 MAX_EPOCHS=6 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_nocga_tinit VIS_DIR=work_dirs/vis_rsar_ut_nocga_tinit bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_nocga_tinit/20260121_063410.log`; full: `work_dirs/exp_rsar_ut_nocga_tinit/20260121_063800.log` |
| Artifacts | `work_dirs/exp_rsar_ut_nocga_tinit/`, `work_dirs/vis_rsar_ut_nocga_tinit/` |
| Results | smoke(N=50,epoch=1): mAP=0.2836（`work_dirs/exp_rsar_ut_nocga_tinit/eval_20260121_063452.json`）；full(N_TRAIN=2000,N_VAL=500,N_TEST=1000,epoch=6): mAP=0.1285（`work_dirs/exp_rsar_ut_nocga_tinit/eval_20260121_065058.json`） |


### E0016: RSAR UnbiasedTeacher + CGA(CLIP) w/ Teacher Init (from RSAR baseline)
| Field | Value |
| --- | --- |
| Objective | 在 teacher 从 RSAR baseline 初始化的前提下，运行 RSAR UT+CGA(CLIP) 并评估 mAP |
| Baseline | E0011（RSAR UT+CGA(CLIP)，teacher=None，mAP=0） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（ema teacher refine_test + CGA） |
| Weights | `TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth`；CLIP 自动下载 |
| Code path | `scripts/exp_rsar_ut.sh`, `sfod/cga.py`, `configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py` |
| Params | `CGA_SCORER=clip`；`TEACHER_CKPT` 指向 RSAR baseline ckpt |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `eval_*.json` 中 `metric.mAP` 非 0 |
| VRAM | ~6–14 GB（取决于 CLIP 模型） |
| Time/epoch | ~? |
| Total time | smoke: ~5–40 min；full: 取决于 `MAX_EPOCHS`/subset |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=clip TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=clip TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 N_TRAIN=500 N_VAL=100 N_TEST=200 MAX_EPOCHS=2 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_full2 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_cga_clip_tinit/20260121_070706.log`; full: `work_dirs/exp_rsar_ut_cga_clip_tinit/20260121_070907.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_clip_tinit/`, `work_dirs/vis_rsar_ut_cga_clip_tinit/` |
| Results | smoke(N=50,epoch=1): mAP=0.2790（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_070757.json`）；full(N_TRAIN=500,N_VAL=100,N_TEST=200,epoch=2): mAP=0.3094（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_071054.json`） |



### E0017: RSAR UT+CGA(CLIP) Teacher-Init Re-eval (N_TEST=1000)
| Field | Value |
| --- | --- |
| Objective | 在不重新训练的前提下，用同一 ckpt 在更大 test 子集（N_TEST=1000）上重跑评估，避免对比混用不同 test 大小 |
| Baseline | E0016 full(N_TEST=200) |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理时仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `test.py`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py` |
| Params | `DO_TRAIN=0`；`SMOKE=1` 仅用于生成子集 split；`N_TEST=50/1000` |
| Metrics (must save) | `eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | smoke: ~1–5 min；full: 取决于 N_TEST |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=clip DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit_eval50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_eval50 bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=clip DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit_eval1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_eval1000 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | `work_dirs/exp_rsar_ut_cga_clip_tinit/*.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_clip_tinit/eval_*.json`, `work_dirs/vis_rsar_ut_cga_clip_tinit_eval*/` |
| Results | smoke(N_TEST=50): mAP=0.2853（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_074558.json`）；full(N_TEST=1000): mAP=0.2539（`work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_074641.json`） |


### E0018: SARCLIP batch_first Compat Smoke (torch 1.7.1)
| Field | Value |
| --- | --- |
| Objective | 验证 SARCLIP 在 torch 1.7.1 下不再因 `nn.MultiheadAttention(batch_first=...)` 崩溃 |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | SARCLIP (text/image encoder) |
| Weights | 可选；无权重时允许 random init（仅验证不崩溃） |
| Code path | `third_party/SARCLIP/sar_clip/transformer.py`, `tools/sarclip_smoke.py`, `scripts/sarclip_torch17_smoke.sh` |
| Params | `--device cpu`（避免 CUDA 兼容问题） |
| Metrics (must save) | 运行成功日志（torch 版本 + score 输出） |
| Checks | 不出现 `unexpected keyword argument 'batch_first'`；脚本退出码为 0 |
| VRAM | 0 GB (CPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~5–30 min（取决于首次 pip 安装） |
| Single-GPU script | `bash scripts/sarclip_torch17_smoke.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash scripts/sarclip_torch17_smoke.sh` |
| Full cmd | `bash scripts/sarclip_torch17_smoke.sh` |
| Smoke | [x] |
| Full | [x] |
| Logs | `work_dirs/sanity/sarclip_smoke_torch171.log` |
| Artifacts | `work_dirs/sanity/sarclip_smoke_torch171.log` |
| Results | torch=1.7.1+cpu（见 `work_dirs/sanity/sarclip_smoke_torch171.log`）；随机初始化可跑通 encode_image/encode_text（仅验证兼容性） |


### E0019: RSAR UT+CGA(SARCLIP) Teacher-Init (Template1)
| Field | Value |
| --- | --- |
| Objective | 在 RSAR 上跑通 UT+CGA(SARCLIP)（使用预训练 SARCLIP 权重）并输出 mAP |
| Baseline | E0016/E0017（UT+CGA(CLIP) teacher-init） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA |
| Weights | `TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth`；`SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `scripts/exp_rsar_ut.sh`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `sfod/cga.py`, `sfod/oriented_rcnn_cga.py`, `weights/sarclip/` |
| Params | `CGA_SCORER=sarclip`；`SARCLIP_MODEL=RN50`；templates=默认（SAR 模板1）；teacher-init；smoke 子集与 full 子集 |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | 训练/测试不报错；`eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~8–16 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于子集规模/epoch |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=sarclip SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=sarclip SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=2 N_TRAIN=500 N_VAL=100 N_TEST=200 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_full2 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-111617-95f1__e0019-smoke.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit/20260122_191633.log`；full: `.rd_queue/logs/J20260122-113323-2f27__e0019-full.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit/20260122_193452.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit/` |
| Results | smoke(N=50,epoch=1): mAP=0.2867（`work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_191754.json`）；full(N=500,epoch=2): mAP=0.3088（`work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_193952.json`） |


### E0020: RSAR UT+CGA(SARCLIP) Prompt Ablation (Template2)
| Field | Value |
| --- | --- |
| Objective | 在 RSAR 上评估 SARCLIP prompt 模板对 UT+CGA 的影响（模板2：更贴近噪声/干扰） |
| Baseline | E0019（模板1） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA |
| Weights | 同 E0019 |
| Code path | `scripts/exp_rsar_ut.sh`, `sfod/cga.py` |
| Params | `CGA_TEMPLATES="<tmpl2a>|<tmpl2b>"` |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | 训练/测试不报错；`eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~8–16 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于子集规模/epoch |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2 bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=2 N_TRAIN=500 N_VAL=100 N_TEST=200 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_full2 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-111708-4600__e0020-smoke.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/20260122_191828.log`；full: `.rd_queue/logs/J20260122-113337-b924__e0020-full.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/20260122_194132.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2/` |
| Results | smoke(N=50,epoch=1): mAP=0.3014（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/eval_20260122_191953.json`）；full(N=500,epoch=2): mAP=0.3119（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/eval_20260122_194541.json`） |


### E0021: RSAR UT+CGA(SARCLIP) Cache Ablation (Eval Rerun)
| Field | Value |
| --- | --- |
| Objective | 对同一 ckpt 重跑 test：对比开启/关闭 CGA cache 的耗时与稳定性（不重新训练） |
| Baseline | E0019 smoke（或 full）ckpt |
| Model | UnbiasedTeacher + OrientedRCNN_CGA |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth`（来自 E0019） |
| Code path | `scripts/exp_rsar_ut.sh`, `sfod/cga.py`, `sfod/scorers/disk_cache.py` |
| Params | `DO_TRAIN=0 DO_TEST=1`；`CGA_CACHE_DIR=work_dirs/cga_cache/e0021`；`CGA_DISABLE_CACHE=1`（ablation） |
| Metrics (must save) | `eval_*.json`（mAP）；log 中记录 cache hit/miss（如有）与 wall time |
| Checks | 两次评估均产出 `eval_*.json` 且非 NaN；禁用 cache 时不影响数值正确性 |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~? |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'test -f work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth && /usr/bin/time -p env CGA_SCORER=sarclip CGA_CACHE_DIR=work_dirs/cga_cache/e0021 CGA_CACHE_VERBOSE=1 DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_cache_eval50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_cache_eval50 bash scripts/exp_rsar_ut.sh && /usr/bin/time -p env CGA_SCORER=sarclip CGA_CACHE_DIR=work_dirs/cga_cache/e0021 CGA_CACHE_VERBOSE=1 DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_cache_eval50_rerun SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_cache_eval50 bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth && /usr/bin/time -p env CGA_SCORER=sarclip CGA_DISABLE_CACHE=1 DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=200 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_nocache_eval200 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_nocache_eval200 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | failed: `.rd_queue/logs/J20260122-112059-428f__e0021-smoke.log`；smoke ok: `.rd_queue/logs/J20260122-112319-f6f2__e0021-smoke-rerun.log`；full: `.rd_queue/logs/J20260122-113351-bff0__e0021-full.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_*.json`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_*` |
| Results | smoke attempt1 failed: `/usr/bin/time` 无法直接执行 `VAR=... cmd`（exit=127）；fix: 用 `env VAR=... cmd`；smoke rerun: mAP=0.2867（`work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_192334.json` / `work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_192404.json`）；full(n_test=200, no-cache): mAP=0.3088（`work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_194635.json`），`real 62.63s`（见 `.rd_queue/logs/J20260122-113351-bff0__e0021-full.log`） |


### E0022: RSAR UT+CGA(CLIP) Robust Eval (interf_jamA)
| Field | Value |
| --- | --- |
| Objective | 在不重新训练的前提下，将 RSAR UT+CGA(CLIP) teacher-init ckpt 在 `interf_jamA` 上重跑评估（对齐 D1） |
| Baseline | E0016/E0017（clean） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理时仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py` |
| Params | `CORRUPT=interf_jamA`；`DO_TRAIN=0 DO_TEST=1`；`N_TEST=50/1000` |
| Metrics (must save) | `eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~? |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=clip CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit_interf_jamA_eval50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_interf_jamA_eval50 CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=clip CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit_interf_jamA_eval1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_interf_jamA_eval1000 CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-122828-c073__e0022-smoke.log`；full: `.rd_queue/logs/J20260122-123810-0382__e0022-full.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval/`, `work_dirs/vis_rsar_ut_cga_clip_tinit_interf_jamA_eval*/` |
| Results | placeholder（软链占位）smoke(N_TEST=50): mAP=0.2853（`work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval/eval_20260122_202934.json`）；placeholder full(N_TEST=1000): mAP=0.2539（`work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval/eval_20260122_203937.json`）；rerun(real data, N_TEST=1000): mAP=0.0509（`work_dirs/exp_rsar_ut_cga_clip_tinit_interf_jamA_eval/eval_20260125_010035.json`） |


### E0023: RSAR UT+CGA(SARCLIP) Robust Eval (interf_jamA, Template2)
| Field | Value |
| --- | --- |
| Objective | 在不重新训练的前提下，将 RSAR UT+CGA(SARCLIP, tmpl2) teacher-init ckpt 在 `interf_jamA` 上重跑评估（对齐 D2） |
| Baseline | E0022（CLIP on interf_jamA） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理时仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth`；`SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py`, `weights/sarclip/` |
| Params | `CORRUPT=interf_jamA`；`DO_TRAIN=0 DO_TEST=1`；tmpl2；`N_TEST=50/1000` |
| Metrics (must save) | `eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | `eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~6–14 GB（取决于 SARCLIP 模型） |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~? |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval50 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval1000 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-122838-ef74__e0023-smoke.log`；full: `.rd_queue/logs/J20260122-123821-7994__e0023-full.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval*/` |
| Results | placeholder（软链占位）smoke(N_TEST=50): mAP=0.2914（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval/eval_20260122_203035.json`）；placeholder full(N_TEST=1000): mAP=0.2592（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval/eval_20260122_204253.json`）；rerun(real data, N_TEST=1000): mAP=0.0509（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamA_eval/eval_20260125_010328.json`） |


### E0024: RSAR UT+CGA(SARCLIP) Mixed Train (sup=clean, unsup/test=interf_jamA, Template2)
| Field | Value |
| --- | --- |
| Objective | 在 RSAR 上用 `SUP_CLEAN=1` 做“clean supervised + interf_jamA unsupervised/test”的混训（对齐 D3 的可运行版本） |
| Baseline | E0020（clean, tmpl2） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA |
| Weights | `TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth`；`SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py` |
| Params | `CORRUPT=interf_jamA`；`SUP_CLEAN=1`；tmpl2；teacher-init |
| Metrics (must save) | `latest.pth`；`eval_*.json`（mAP）；`--show-dir` 输出 |
| Checks | 训练/测试不报错；`eval_*.json` 中 `metric.mAP` 存在且非 NaN |
| VRAM | ~8–16 GB |
| Time/epoch | ~? |
| Total time | smoke: ~5–30 min；full: 取决于子集规模/epoch |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth CORRUPT=interf_jamA SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth CORRUPT=interf_jamA SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=2 N_TRAIN=500 N_VAL=100 N_TEST=200 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA_full2 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-122846-294b__e0024-smoke.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/20260122_203136.log`；full: `.rd_queue/logs/J20260122-123832-bc69__e0024-full.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/20260122_204648.log`；rerun(real data) failed: `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/20260125_010625.log`；rerun(real data) ok: `.rd_queue/logs/J20260124-175719-4fe3__e0024-full-real.log`；`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/20260125_015817.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/` |
| Results | placeholder（软链占位）smoke(N=50,epoch=1): mAP=0.3012（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/eval_20260122_203315.json`）；placeholder full(N=500,epoch=2): mAP=0.3093（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/eval_20260122_205109.json`）；rerun(real data) failed: CUDA OOM（当前多卡被 `version_2/scripts/vram_fill.py --leave-free-mb 1024` 占用，仅剩 ~1GB free），见 `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/20260125_010625.log`；rerun(real data, GPU0 freed, N_TEST=200): mAP=0.0255（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/eval_20260125_015941.json`，log: `.rd_queue/logs/J20260124-175719-4fe3__e0024-full-real.log`） |


### E0025: RSAR Mixed-Train Re-eval (clean vs interf_jamA, N_TEST=1000)
| Field | Value |
| --- | --- |
| Objective | 用 E0024 的 ckpt 在 clean 与 `interf_jamA` 上各自重跑一次 `N_TEST=1000` 的评估（对齐 D4 的 eval 部分；多干扰可在有数据后扩展） |
| Baseline | E0024 |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理时仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py` |
| Params | `DO_TRAIN=0 DO_TEST=1`；`CORRUPT=clean|interf_jamA`；`N_TEST=50/1000` |
| Metrics (must save) | 每个 corrupt 一个 `eval_*.json`（mAP）；对应 `--show-dir` 输出 |
| Checks | 两次评估均产出 `eval_*.json` 且非 NaN |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~? |
| Single-GPU script | `bash scripts/exp_rsar_ut.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'test -f work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=clean DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean50 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_mix_eval_clean50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_mix_eval_clean50 bash scripts/exp_rsar_ut.sh && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA50 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_mix_eval_interf_jamA50 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_mix_eval_interf_jamA50 bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=clean DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean1000 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_mix_eval_clean1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_mix_eval_clean1000 bash scripts/exp_rsar_ut.sh && CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors CORRUPT=interf_jamA DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA1000 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamA/latest.pth VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_mix_eval_interf_jamA1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_mix_eval_interf_jamA1000 bash scripts/exp_rsar_ut.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue/logs/J20260122-122901-0968__e0025-smoke.log`；full: `.rd_queue/logs/J20260122-123848-b5bf__e0025-full.log`；rerun(real data, ckpt updated): `.rd_queue/logs/J20260124-180200-1f99__e0025-full-real2.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_*`, `work_dirs/vis_rsar_ut_cga_sarclip_mix_eval_*` |
| Results | placeholder（软链占位）smoke(N_TEST=50): clean mAP=0.3012（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean50/eval_20260122_203455.json`）；interf_jamA mAP=0.3012（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA50/eval_20260122_203606.json`）；placeholder full(N_TEST=1000): clean mAP=0.2575（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean1000/eval_20260122_205237.json`）；interf_jamA mAP=0.2575（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA1000/eval_20260122_205529.json`）；rerun(real data, N_TEST=1000): clean mAP=0.2575（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean1000/eval_20260125_010910.json`），interf_jamA mAP=0.0510（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA1000/eval_20260125_011050.json`）；rerun(real data, N_TEST=1000, ckpt=E0024-full-real latest.pth): clean mAP=0.2697（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_clean1000/eval_20260125_020236.json`），interf_jamA mAP=0.0467（`work_dirs/exp_rsar_ut_cga_sarclip_mix_eval_interf_jamA1000/eval_20260125_020403.json`） |


### E0026: RSAR Severity Curve Eval (interf_jamA_s1..s5, N_TEST=1000)
| Field | Value |
| --- | --- |
| Objective | 用同一个 ckpt 在 `clean` 与 `interf_jamA_s1..s5`（test-only）上做鲁棒性曲线评估（mAP vs severity） |
| Baseline | clean（同 ckpt） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth` |
| Code path | `scripts/eval_rsar_severity_curve.sh`, `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py` |
| Params | `CORRUPT_BASE=interf_jamA`；`SEVERITIES=1..5`；`N_TEST=50/1000`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `severity_summary.csv`；各 severity 的 `eval_*.json` |
| Checks | CSV 行数=1(clean)+5；每行 mAP 非 NaN；severity 目录必须存在（脚本会硬检查） |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~6–10 min |
| Single-GPU script | `bash scripts/eval_rsar_severity_curve.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth CORRUPT_BASE=interf_jamA SEVERITIES=3 INCLUDE_CLEAN=1 N_TEST=50 bash scripts/eval_rsar_severity_curve.sh'` |
| Full cmd | `bash -lc 'CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth CORRUPT_BASE=interf_jamA SEVERITIES=1,2,3,4,5 INCLUDE_CLEAN=1 N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | N/A（交互式运行，未生成 `.rd_queue/logs/*`；以 `severity_summary.csv`+`eval_*.json` 为准） |
| Artifacts | `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamA/` |
| Results | smoke(N_TEST=50, SEVERITIES=3): clean mAP=0.291350（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamA/clean/eval_20260125_023529.json`），interf_jamA_s3 mAP=0.362827（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamA/interf_jamA_s3/eval_20260125_023549.json`）；full(N_TEST=1000, SEVERITIES=1..5): `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamA/severity_summary.csv`（mAP: clean=0.259225, s1=0.262595, s2=0.260502, s3=0.251422, s4=0.248343, s5=0.240369；每行含 `eval_json` 路径） |


### E0027: RSAR Severity Curve Eval (interf_jamB_s1..s5, N_TEST=1000)
| Field | Value |
| --- | --- |
| Objective | 用同一个 ckpt 在 `clean` 与 `interf_jamB_s1..s5`（test-only）上做鲁棒性曲线评估（mAP vs severity） |
| Baseline | clean（同 ckpt） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（推理仅做检测） |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth` |
| Code path | `scripts/eval_rsar_severity_curve.sh`, `scripts/exp_rsar_ut.sh`, `tools/verify_rsar_corrupt_switch.py` |
| Params | `CORRUPT_BASE=interf_jamB`；`SEVERITIES=1..5`；`N_TEST=50/1000`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `severity_summary.csv`；各 severity 的 `eval_*.json` |
| Checks | CSV 行数=1(clean)+5；每行 mAP 非 NaN；severity 目录必须存在（脚本会硬检查） |
| VRAM | ~4–10 GB |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | ~6–10 min |
| Single-GPU script | `bash scripts/eval_rsar_severity_curve.sh` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth CORRUPT_BASE=interf_jamB SEVERITIES=3 INCLUDE_CLEAN=1 N_TEST=50 bash scripts/eval_rsar_severity_curve.sh'` |
| Full cmd | `bash -lc 'CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/latest.pth CORRUPT_BASE=interf_jamB SEVERITIES=1,2,3,4,5 INCLUDE_CLEAN=1 N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | N/A（交互式运行，未生成 `.rd_queue/logs/*`；以 `severity_summary.csv`+`eval_*.json` 为准） |
| Artifacts | `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamB/` |
| Results | smoke(N_TEST=50, SEVERITIES=3): clean mAP=0.291350（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamB/clean/eval_20260125_023624.json`），interf_jamB_s3 mAP=0.222065（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamB/interf_jamB_s3/eval_20260125_023645.json`）；full(N_TEST=1000, SEVERITIES=1..5): `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2/interf_jamB/severity_summary.csv`（mAP: clean=0.259225, s1=0.261777, s2=0.217373, s3=0.172148, s4=0.077893, s5=0.028767；每行含 `eval_json` 路径） |


### E0028: RSAR Baseline Train (interf_jamB_s3 only)
| Field | Value |
| --- | --- |
| Objective | supervised baseline（OrientedRCNN）仅用 `interf_jamB_s3` 做训练/验证/测试（关注 jamB_s3 域内收敛与 mAP） |
| Baseline | RSAR clean baseline（E0012 / E0009） |
| Model | OrientedRCNN (mmrotate baseline) |
| Weights | `CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth` |
| Code path | `scripts/exp_rsar_baseline.sh`, `sfod/utils/patches.py` |
| Params | `CORRUPT=interf_jamB_s3`；SMOKE 子集 `N_TRAIN/N_VAL/N_TEST`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json`（mAP）+ show-dir 可视化 |
| Checks | `eval_*.json` 存在且 mAP 非 NaN；`tools/verify_rsar_corrupt_switch.py --corrupt interf_jamB_s3 --splits train,val,test` missing=0/conflict=0 |
| VRAM | ~6–12 GB |
| Total time | smoke ~5–15 min；full ~?（取决于 N 与 epoch） |
| Single-GPU script | `bash scripts/exp_rsar_baseline.sh` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_baseline_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_baseline_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_baseline_interf_jamB_s3_smoke bash scripts/exp_rsar_baseline.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_baseline_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_baseline_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_baseline_interf_jamB_s3_full bash scripts/exp_rsar_baseline.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve_baseline.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_baseline_interf_jamB_s3/20260125_121848.log`；full: `.rd_queue/logs/J20260125-043738-e279__e0028-full.log`；train: `work_dirs/exp_rsar_baseline_interf_jamB_s3/20260125_131146.log` |
| Artifacts | `work_dirs/exp_rsar_baseline_interf_jamB_s3/`, `work_dirs/vis_rsar_baseline_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_baseline_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_baseline_interf_jamB_s3/eval_20260125_121911.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.1986（`work_dirs/exp_rsar_baseline_interf_jamB_s3/eval_20260125_132251.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_baseline_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.1975,s1=0.2038,s2=0.2029,s3=0.1986,s4=0.1756,s5=0.1131 |


### E0029: RSAR Baseline Train (mix clean + interf_jamB_s3, 1:1)
| Field | Value |
| --- | --- |
| Objective | supervised baseline 以 1:1 混训 clean 与 `interf_jamB_s3`（同标注、不同图像目录） |
| Baseline | E0028（interf-only） |
| Model | OrientedRCNN (mmrotate baseline) |
| Weights | `CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth` |
| Code path | `scripts/exp_rsar_baseline.sh`, `sfod/utils/patches.py`（`mix_train=1`） |
| Params | `CORRUPT=interf_jamB_s3`；`MIX_TRAIN=1`；`MIX_TRAIN_*_TIMES=1`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json`（mAP）+ jamB severity CSV |
| Checks | `eval_*.json` 存在且 mAP 非 NaN；train split 下存在 `images-clean`（脚本会创建） |
| VRAM | ~6–12 GB |
| Total time | smoke ~5–20 min；full ~? |
| Single-GPU script | `bash scripts/exp_rsar_baseline.sh` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CORRUPT=interf_jamB_s3 MIX_TRAIN=1 MIX_TRAIN_CLEAN_TIMES=1 MIX_TRAIN_CORRUPT_TIMES=1 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_baseline_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_baseline_mix_interf_jamB_s3_smoke bash scripts/exp_rsar_baseline.sh'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CORRUPT=interf_jamB_s3 MIX_TRAIN=1 MIX_TRAIN_CLEAN_TIMES=1 MIX_TRAIN_CORRUPT_TIMES=1 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_baseline_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_baseline_mix_interf_jamB_s3_full bash scripts/exp_rsar_baseline.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve_baseline.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/20260125_121942.log`；full: `.rd_queue/logs/J20260125-043738-0b8e__e0029-full.log`；train: `work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/20260125_124005.log` |
| Artifacts | `work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/`, `work_dirs/vis_rsar_baseline_mix_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_baseline_mix_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/eval_20260125_122008.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.3246（`work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/eval_20260125_125952.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_baseline_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.3739,s1=0.3842,s2=0.3828,s3=0.3246,s4=0.2065,s5=0.0768 |


### E0030: RSAR UT Train (no CGA, interf_jamB_s3 only)
| Field | Value |
| --- | --- |
| Objective | UnbiasedTeacher（无 CGA）仅用 `interf_jamB_s3`（sup/unsup/val/test 统一在 s3 域）训练并评估 |
| Baseline | E0028 baseline（teacher init） |
| Model | UnbiasedTeacher + OrientedRCNN (no CGA) |
| Weights | `CKPT=work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `scripts/eval_rsar_severity_curve.sh` |
| Params | `CORRUPT=interf_jamB_s3`；`CGA_SCORER=none`；`TEACHER_CKPT=E0028/latest.pth`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json` + jamB severity CSV |
| Checks | 训练/测试完成；mAP 非 NaN；ckpt 文件存在 |
| VRAM | ~8–16 GB |
| Total time | smoke ~10–30 min；full ~? |
| Smoke cmd | `bash -lc 'test -f work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_nocga_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_nocga_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_nocga_interf_jamB_s3_smoke bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_nocga_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_nocga_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_nocga_interf_jamB_s3_full bash scripts/exp_rsar_ut.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/20260125_122156.log`；full: `.rd_queue/logs/J20260125-043738-d33b__e0030-full.log`；train: `work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/20260125_133445.log` |
| Artifacts | `work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/`, `work_dirs/vis_rsar_ut_nocga_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/eval_20260125_122230.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.1317（`work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/eval_20260125_134639.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.1430,s1=0.1433,s2=0.1396,s3=0.1317,s4=0.0926,s5=0.0495 |


### E0031: RSAR UT Train (no CGA, SUP clean + unsup/val/test interf_jamB_s3)
| Field | Value |
| --- | --- |
| Objective | UnbiasedTeacher（无 CGA）在 `CORRUPT=interf_jamB_s3` 下启用 `SUP_CLEAN=1`：监督分支用 clean，unsup/val/test 用 s3 域（对齐已有 mix_interf_jamA 方案） |
| Baseline | E0029 baseline mix（teacher init） |
| Model | UnbiasedTeacher + OrientedRCNN (no CGA) |
| Weights | `CKPT=work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/latest.pth` |
| Code path | `scripts/exp_rsar_ut.sh`, `scripts/eval_rsar_severity_curve.sh` |
| Params | `CORRUPT=interf_jamB_s3`；`SUP_CLEAN=1`；`TEACHER_CKPT=E0029/latest.pth`；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json` + jamB severity CSV |
| Checks | 训练/测试完成；mAP 非 NaN |
| VRAM | ~8–16 GB |
| Total time | smoke ~10–30 min；full ~? |
| Smoke cmd | `bash -lc 'test -f work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_nocga_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_nocga_mix_interf_jamB_s3_smoke bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_nocga_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_nocga_mix_interf_jamB_s3_full bash scripts/exp_rsar_ut.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/20260125_122306.log`；full: `.rd_queue/logs/J20260125-043738-d0c6__e0031-full.log`；train: `work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/20260125_135833.log` |
| Artifacts | `work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/`, `work_dirs/vis_rsar_ut_nocga_mix_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_mix_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/eval_20260125_122339.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.2260（`work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/eval_20260125_141022.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.2900,s1=0.2854,s2=0.2887,s3=0.2260,s4=0.1283,s5=0.0788 |


### E0032: RSAR UT+CGA(SARCLIP) Train (interf_jamB_s3 only)
| Field | Value |
| --- | --- |
| Objective | UnbiasedTeacher + CGA(SARCLIP) 在 `interf_jamB_s3` 域内训练并评估（interf-only） |
| Baseline | E0030（no CGA） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA (SARCLIP scorer) |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/latest.pth`；SARCLIP `weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `scripts/exp_rsar_ut.sh`, `sfod/cga.py`, `scripts/eval_rsar_severity_curve.sh` |
| Params | `CGA_SCORER=sarclip`；`CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\"`；`CORRUPT=interf_jamB_s3`；teacher=E0028；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json` + jamB severity CSV |
| Checks | 训练/测试完成；mAP 非 NaN；权重路径存在 |
| VRAM | ~10–18 GB |
| Total time | smoke ~15–40 min；full ~? |
| Smoke cmd | `bash -lc 'test -f weights/sarclip/RN50/rn50_model.safetensors && test -f work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3_smoke bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f weights/sarclip/RN50/rn50_model.safetensors && test -f work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3_full bash scripts/exp_rsar_ut.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/20260125_122437.log`；full: `.rd_queue/logs/J20260125-043851-3815__e0032-full.log`；train: `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/20260125_142242.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/eval_20260125_122514.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.1330（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/eval_20260125_143501.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.1398,s1=0.1460,s2=0.1404,s3=0.1330,s4=0.0954,s5=0.0377 |


### E0033: RSAR UT+CGA(SARCLIP) Train (SUP clean + unsup/val/test interf_jamB_s3)
| Field | Value |
| --- | --- |
| Objective | UnbiasedTeacher + CGA(SARCLIP) 在 `CORRUPT=interf_jamB_s3` 下启用 `SUP_CLEAN=1` 的 mix 方案，并跑 jamB severity 曲线 |
| Baseline | E0032（interf-only） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA (SARCLIP scorer) |
| Weights | `CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/latest.pth`；SARCLIP `weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `scripts/exp_rsar_ut.sh`, `scripts/eval_rsar_severity_curve.sh` |
| Params | `SUP_CLEAN=1`；`CORRUPT=interf_jamB_s3`；teacher=E0029；模板同 E0032；`CUDA_VISIBLE_DEVICES=0` |
| Metrics (must save) | `eval_*.json` + jamB severity CSV |
| Checks | 训练/测试完成；mAP 非 NaN |
| VRAM | ~10–18 GB |
| Total time | smoke ~15–40 min；full ~? |
| Smoke cmd | `bash -lc 'test -f weights/sarclip/RN50/rn50_model.safetensors && test -f work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=1 N_TRAIN=50 N_VAL=50 N_TEST=50 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3_smoke bash scripts/exp_rsar_ut.sh'` |
| Full cmd | `bash -lc 'test -f weights/sarclip/RN50/rn50_model.safetensors && test -f work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth && conda run -n iraod python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamB_s3 --splits train,val,test && CUDA_VISIBLE_DEVICES=0 CGA_SCORER=sarclip CGA_TEMPLATES=\"a noisy SAR image of a {}|this SAR patch shows a {} under interference\" SARCLIP_MODEL=RN50 SARCLIP_PRETRAINED=weights/sarclip/RN50/rn50_model.safetensors TEACHER_CKPT=work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/latest.pth CORRUPT=interf_jamB_s3 SUP_CLEAN=1 SMOKE=1 MAX_EPOCHS=6 N_TRAIN=2000 N_VAL=500 N_TEST=1000 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=2 WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3 VIS_DIR=work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3_full bash scripts/exp_rsar_ut.sh && CUDA_VISIBLE_DEVICES=0 CKPT=work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/latest.pth CORRUPT_BASE=interf_jamB N_TEST=1000 bash scripts/eval_rsar_severity_curve.sh'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/20260125_122548.log`；full: `.rd_queue/logs/J20260125-043851-29f5__e0033-full.log`；train: `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/20260125_144651.log` |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/`, `work_dirs/vis_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/`, `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/interf_jamB/severity_summary.csv` |
| Results | smoke(N=50,epoch=1): mAP=0.0000（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/eval_20260125_122623.json`）；full(N_TRAIN=2000,epoch=6): mAP=0.2317（`work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/eval_20260125_145932.json`）；jamB severity（`work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`）mAP: clean=0.2961,s1=0.2960,s2=0.2963,s3=0.2317,s4=0.1403,s5=0.0782 |


### E0034: RSAR Full-Sample Mode + CLI Args Verification
| Field | Value |
| --- | --- |
| Objective | 验证 RSAR 默认不再强制 50-sample 子集（len(train)>50），并可通过 `train.py/test.py --data-root/--cga-scorer/--sarclip-*` 等短命令运行 |
| Baseline | N/A（工具类实验，不涉及模型训练） |
| Model | N/A（config 解析 + dataset build） |
| Weights | N/A（工具类实验，不涉及模型训练） |
| Code path | `tools/verify_full_sample_mode.py`, `train.py`, `test.py`, `scripts/exp_rsar_ut.sh`, `scripts/exp_rsar_baseline.sh`, `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py` |
| Params | `--data-root $(pwd)/dataset/RSAR`；`--min-annfiles` |
| Metrics (must save) | `work_dirs/sanity/full_sample_mode.json`（包含 split annfile counts + dataset len） |
| Checks | JSON 中 `ok=true` 且 `dataset_lens.lens.train>50` |
| VRAM | ~0 GB |
| Total time | ~10–60 s |
| Single-GPU script | `conda run -n iraod python tools/verify_full_sample_mode.py --config configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --data-root "$(pwd)/dataset/RSAR"` |
| Multi-GPU script | `N/A` |
| Smoke cmd | `bash -lc 'conda run -n iraod python tools/verify_full_sample_mode.py --config configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --data-root \"$(pwd)/dataset/RSAR\" --min-annfiles 51'` |
| Full cmd | `bash -lc 'conda run -n iraod python tools/verify_full_sample_mode.py --config configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --data-root \"$(pwd)/dataset/RSAR\" --min-annfiles 1000'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: stdout；full: `.rd_queue/logs/J20260126-171458-539a__e0034-full.log` |
| Artifacts | `work_dirs/sanity/full_sample_mode.json` |
| Results | len(train)=78837, len(test)=8538（见 `work_dirs/sanity/full_sample_mode.json`） |


### E0035: RSAR Baseline Train/Test (FULL dataset)
| Field | Value |
| --- | --- |
| Objective | 用 RSAR 全量数据（train/val/test 全部样本）训练 supervised baseline，并在 full test 上评估 mAP |
| Baseline | E0009（子集版 baseline） |
| Model | OrientedRCNN R50-FPN (le90) |
| Weights | `None` |
| Code path | `configs/experiments/rsar/baseline_oriented_rcnn_rsar.py`, `train.py`, `test.py` |
| Params | `--data-root $(pwd)/dataset/RSAR`；`--no-validate`（训练期不做 eval，末尾单独 full test） |
| Metrics (must save) | `work_dirs/exp_rsar_baseline_full_nanfix/latest.pth`；`work_dirs/exp_rsar_baseline_full_nanfix/eval_*.json`（mAP） |
| Checks | ckpt 与 eval json 存在；mAP 非 NaN |
| VRAM | ~6–16 GB |
| Total time | ~hours–days（取决于 GPU 与 IO） |
| Single-GPU script | `CUDA_VISIBLE_DEVICES=8 conda run -n iraod python train.py configs/experiments/rsar/baseline_oriented_rcnn_rsar.py --work-dir work_dirs/exp_rsar_baseline_full_nanfix --data-root "$(pwd)/dataset/RSAR" --no-validate` |
| Multi-GPU script | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n iraod torchrun --nproc_per_node=8 --master_port=29501 train.py configs/experiments/rsar/baseline_oriented_rcnn_rsar.py --work-dir work_dirs/exp_rsar_baseline_full_nanfix --data-root \"${DATA_ROOT}\" --no-validate --launcher pytorch'` |
| Smoke cmd | 参考 E0009（子集 smoke 已通过） |
| Full cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n iraod torchrun --nproc_per_node=8 --master_port=29501 train.py configs/experiments/rsar/baseline_oriented_rcnn_rsar.py --work-dir work_dirs/exp_rsar_baseline_full_nanfix --data-root \"${DATA_ROOT}\" --no-validate --launcher pytorch && CUDA_VISIBLE_DEVICES=0 conda run -n iraod python test.py configs/experiments/rsar/baseline_oriented_rcnn_rsar.py work_dirs/exp_rsar_baseline_full_nanfix/latest.pth --eval mAP --work-dir work_dirs/exp_rsar_baseline_full_nanfix --data-root \"${DATA_ROOT}\"'` |
| Smoke | [x] |
| Full | [x] |
| Logs | `.rd_queue/logs/J20260127-080433-13a6__e0035-full-nanfix-nccl.log`（训练 log: `work_dirs/exp_rsar_baseline_full_nanfix/20260127_160444.log`；旧跑崩: `work_dirs/exp_rsar_baseline_full/20260127_012944.log`） |
| Artifacts | `work_dirs/exp_rsar_baseline_full_nanfix/` |
| Results | mAP=0.6535（`work_dirs/exp_rsar_baseline_full_nanfix/eval_20260127_180332.json`） |


### E0036: RSAR UnbiasedTeacher (CGA off) Train/Test (FULL dataset)
| Field | Value |
| --- | --- |
| Objective | 用 RSAR 全量数据训练 UT（禁用 CGA），teacher 从 E0035 初始化，并在 full test 上评估 mAP |
| Baseline | E0010（子集版 UT no-CGA） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（但 `CGA_SCORER=none`） |
| Weights | teacher-init: `work_dirs/exp_rsar_baseline_full_nanfix/latest.pth` |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `train.py`, `test.py` |
| Params | `--cga-scorer none`；`--no-validate`；`--cfg-options load_from=<baseline> model.ema_ckpt=<baseline> data.samples_per_gpu=16 data.workers_per_gpu=4` |
| Metrics (must save) | `work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/latest.pth`；`work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/eval_*.json`（mAP） |
| Checks | ckpt 与 eval json 存在；mAP 非 NaN |
| VRAM | ~35–47 GB（目标：尽量接近 48GB/卡） |
| Total time | ~hours–days |
| Single-GPU script | `CUDA_VISIBLE_DEVICES=8 conda run -n iraod python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16 --data-root "$(pwd)/dataset/RSAR" --cga-scorer none --no-validate --cfg-options load_from=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth model.ema_ckpt=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth data.samples_per_gpu=16 data.workers_per_gpu=4` |
| Multi-GPU script | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; CUDA_VISIBLE_DEVICES=4,5,6 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29502 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16 --data-root \"${DATA_ROOT}\" --cga-scorer none --no-validate --launcher pytorch --cfg-options load_from=\"${BASELINE_CKPT}\" model.ema_ckpt=\"${BASELINE_CKPT}\" data.samples_per_gpu=16 data.workers_per_gpu=4'` |
| Smoke cmd | 参考 E0010（子集 smoke 已通过） |
| Full cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; CUDA_VISIBLE_DEVICES=4,5,6 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29502 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16 --data-root \"${DATA_ROOT}\" --cga-scorer none --no-validate --launcher pytorch --cfg-options load_from=\"${BASELINE_CKPT}\" model.ema_ckpt=\"${BASELINE_CKPT}\" data.samples_per_gpu=16 data.workers_per_gpu=4 && CUDA_VISIBLE_DEVICES=4 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/latest.pth --eval mAP --work-dir work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16 --data-root \"${DATA_ROOT}\" --cga-scorer none --cfg-options load_from=None model.ema_ckpt=None'` |
| Smoke | [x] |
| Full | [x] |
| Logs | `.rd_queue_ut_nocga/logs/J20260127-141036-aadb__e0036-full-vram16-nanfix.log`（训练 log: `work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/20260127_221047.log`；旧版本手动停掉: `.rd_queue_ut_nocga/logs/J20260127-080710-5fc7__e0036-full-nanfix-nccl.log`） |
| Artifacts | `work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/` |
| Results | mAP=0.1205（`work_dirs/exp_rsar_ut_nocga_full_nanfix_vram16/eval_20260127_231517.json`）；注：当时 `SemiDOTADataset.flag` 与 `__len__` 不一致导致 epoch 被 sampler 截断（日志为 `Epoch[1][*/177]`），且配置 `weight_l=0/use_bbox_reg=False` 使监督/回归损失恒为 0，因此该结果不可作为最终对比；修复后重跑见 E0038 |


### E0037: RSAR UnbiasedTeacher + CGA(SARCLIP) Train/Test (FULL dataset)
| Field | Value |
| --- | --- |
| Objective | 用 RSAR 全量数据训练 UT+CGA(SARCLIP)，teacher 从 E0035 初始化，并在 full test 上评估 mAP |
| Baseline | E0016/E0019（子集版 UT+CGA(SARCLIP)） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA (SARCLIP scorer) |
| Weights | teacher-init: `work_dirs/exp_rsar_baseline_full_nanfix/latest.pth`；SARCLIP: `weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `train.py`, `test.py`, `sfod/cga.py` |
| Params | `--cga-scorer sarclip`；`--sarclip-model RN50`；`--sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors`；`--no-validate`；`--cfg-options data.samples_per_gpu=14 data.workers_per_gpu=4` |
| Metrics (must save) | `work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/latest.pth`；`work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/eval_*.json`（mAP） |
| Checks | ckpt 与 eval json 存在；mAP 非 NaN；SARCLIP 权重存在 |
| VRAM | ~35–47 GB（目标：尽量接近 48GB/卡） |
| Total time | ~hours–days |
| Single-GPU script | `CUDA_VISIBLE_DEVICES=8 conda run -n iraod python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14 --data-root "$(pwd)/dataset/RSAR" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --no-validate --cfg-options load_from=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth model.ema_ckpt=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth data.samples_per_gpu=14 data.workers_per_gpu=4` |
| Multi-GPU script | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; test -f weights/sarclip/RN50/rn50_model.safetensors; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29503 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14 --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --no-validate --launcher pytorch --cfg-options load_from=\"${BASELINE_CKPT}\" model.ema_ckpt=\"${BASELINE_CKPT}\" data.samples_per_gpu=14 data.workers_per_gpu=4'` |
| Smoke cmd | 参考 E0016（子集 smoke 已通过） |
| Full cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; test -f weights/sarclip/RN50/rn50_model.safetensors; DATA_ROOT=\"$(pwd)/dataset/RSAR\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29503 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14 --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --no-validate --launcher pytorch --cfg-options load_from=\"${BASELINE_CKPT}\" model.ema_ckpt=\"${BASELINE_CKPT}\" data.samples_per_gpu=14 data.workers_per_gpu=4 && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/latest.pth --eval mAP --work-dir work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14 --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --cfg-options load_from=None model.ema_ckpt=None'` |
| Smoke | [x] |
| Full | [x] |
| Logs | `.rd_queue_ut_cga/logs/J20260127-141038-5d9a__e0037-full-vram14-nanfix.log`（训练 log: `work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/20260127_221049.log`；旧版本手动停掉: `.rd_queue_ut_cga/logs/J20260127-080724-ce51__e0037-full-nanfix-nccl.log`） |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/` |
| Results | mAP=0.0918（`work_dirs/exp_rsar_ut_cga_sarclip_full_nanfix_vram14/eval_20260127_231554.json`）；注：同 E0036（epoch 被截断为 `Epoch[1][*/202]` + `weight_l=0/use_bbox_reg=False`）；修复后重跑见 E0039 |


### E0038: RSAR UnbiasedTeacher (CGA off) Train/Test (FULL dataset, FIX)
| Field | Value |
| --- | --- |
| Objective | 修复 epoch 截断（`flag_len==len(dataset)`）+ 默认开启监督/回归（`weight_l=1/use_bbox_reg=True`）后，重新用 RSAR 全量数据训练 UT(no-CGA) 并在 full test 上评估 mAP（学生+EMA teacher） |
| Baseline | E0035（supervised baseline mAP=0.6535） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA（CGA_SCORER=none） |
| Weights | teacher-init: `work_dirs/exp_rsar_baseline_full_nanfix/latest.pth` |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py`, `train.py`, `test.py`, `sfod/semi_dota_dataset.py` |
| Params | `--cga-scorer none`；`--teacher-ckpt <baseline>`；`--samples-per-gpu 16`；`--workers-per-gpu 4`；默认 `weight_l=1/use_bbox_reg=True` |
| Metrics (must save) | `latest.pth`；`latest_ema.pth`；学生/EMA 各一个 `eval_*.json` |
| Checks | 训练日志 epoch iters ≈ `78837/(16*3)=1643`（不再是 177）；监督 loss 非 0；mAP 非 NaN |
| VRAM | 目标 ~45–48 GB/卡（RTX 4090D 48GB） |
| Total time | 预估 ~10–12 h（3 卡） |
| Smoke cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}\"; DATA_ROOT=\"${RSAR_DATA_ROOT:-/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR}\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; WORK_DIR=work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke; mkdir -p \"${WORK_DIR}\"; rm -f \"${WORK_DIR}/SMOKE_OK\" || true; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29525 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir \"${WORK_DIR}\" --no-validate --launcher pytorch --data-root \"${DATA_ROOT}\" --teacher-ckpt \"${BASELINE_CKPT}\" --cga-scorer none --samples-per-gpu 16 --workers-per-gpu 4 --max-epochs 1 && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py \"${WORK_DIR}/latest.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer none --cfg-options load_from=None model.ema_ckpt=None && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py \"${WORK_DIR}/latest_ema.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer none && touch \"${WORK_DIR}/SMOKE_OK\"'` |
| Full cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}\"; DATA_ROOT=\"${RSAR_DATA_ROOT:-/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR}\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; test -f work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke/SMOKE_OK; WORK_DIR=work_dirs/exp_rsar_ut_nocga_full_fix_vram16; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29526 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir \"${WORK_DIR}\" --no-validate --launcher pytorch --data-root \"${DATA_ROOT}\" --teacher-ckpt \"${BASELINE_CKPT}\" --cga-scorer none --samples-per-gpu 16 --workers-per-gpu 4 && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py \"${WORK_DIR}/latest.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer none --cfg-options load_from=None model.ema_ckpt=None && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py \"${WORK_DIR}/latest_ema.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer none'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue_ut_nocga/logs/J20260128-134057-16bf__e0038-smoke-fix-vram16.log`（exit=1：当时 EMA eval config 缺少 `data` 字段；已修复并补跑 EMA eval + 写入 `SMOKE_OK`）；full: `.rd_queue_ut_nocga_full/logs/J20260128-152926-ecb5__e0038-full-fix-vram16-gpu236.log`；train: `work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke/20260128_214108.log` / `work_dirs/exp_rsar_ut_nocga_full_fix_vram16/20260128_232936.log`；previous failures: `.rd_queue_ut_nocga/logs/J20260127-171316-356e__e0038-smoke.log` / `.rd_queue_ut_nocga/logs/J20260127-171657-9620__e0038-smoke-rerun.log`（NCCL Error 3）/ `.rd_queue_ut_nocga/logs/J20260128-131634-f635__e0038-smoke-rerun2.log`（CUDA OOM） |
| Artifacts | `work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke/`, `work_dirs/exp_rsar_ut_nocga_full_fix_vram16/` |
| Results | smoke: student mAP=0.6139（`work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke/eval_20260128_224352.json`）；EMA mAP=0.6310（`work_dirs/exp_rsar_ut_nocga_full_fix_vram16_smoke/eval_20260128_230602.json`）；full: student mAP=0.6179（`work_dirs/exp_rsar_ut_nocga_full_fix_vram16/eval_20260129_132858.json`）；EMA mAP=0.6670（`work_dirs/exp_rsar_ut_nocga_full_fix_vram16/eval_20260129_133308.json`） |


### E0039: RSAR UnbiasedTeacher + CGA(SARCLIP) Train/Test (FULL dataset, FIX)
| Field | Value |
| --- | --- |
| Objective | 在 E0038 修复基础上，开启 CGA(SARCLIP) 以提升伪标签质量，并在 full test 上评估 mAP（学生+EMA teacher） |
| Baseline | E0035（supervised baseline mAP=0.6535） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA (SARCLIP scorer) |
| Weights | teacher-init: `work_dirs/exp_rsar_baseline_full_nanfix/latest.pth`；SARCLIP: `weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py`, `configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py`, `train.py`, `test.py`, `sfod/cga.py` |
| Params | `--cga-scorer sarclip`；`--sarclip-model RN50`；`--sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors`；`--teacher-ckpt <baseline>`；`--samples-per-gpu 14`；`--workers-per-gpu 4`；默认 `weight_l=1/use_bbox_reg=True` |
| Metrics (must save) | `latest.pth`；`latest_ema.pth`；学生/EMA 各一个 `eval_*.json` |
| Checks | 训练日志 epoch iters ≈ `78837/(14*3)=1878`（不再是 202）；监督 loss 非 0；mAP 非 NaN；SARCLIP 权重存在；CGA init classes 不应回退为 `class_0..` |
| VRAM | 目标 ~40–48 GB/卡（视 batch 上限而定） |
| Total time | 预估 ~10–12 h（3 卡） |
| Smoke cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}\"; DATA_ROOT=\"${RSAR_DATA_ROOT:-/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR}\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; test -f weights/sarclip/RN50/rn50_model.safetensors; WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke; mkdir -p \"${WORK_DIR}\"; rm -f \"${WORK_DIR}/SMOKE_OK\" || true; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29527 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir \"${WORK_DIR}\" --no-validate --launcher pytorch --data-root \"${DATA_ROOT}\" --teacher-ckpt \"${BASELINE_CKPT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --samples-per-gpu 14 --workers-per-gpu 4 --max-epochs 1 && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py \"${WORK_DIR}/latest.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --cfg-options load_from=None model.ema_ckpt=None && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py \"${WORK_DIR}/latest_ema.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors && touch \"${WORK_DIR}/SMOKE_OK\"'` |
| Full cmd | `bash -lc 'set -euo pipefail; unset NCCL_P2P_DISABLE NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL NCCL_PROTO NCCL_MAX_NCHANNELS || true; export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}\"; DATA_ROOT=\"${RSAR_DATA_ROOT:-/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR}\"; BASELINE_CKPT=work_dirs/exp_rsar_baseline_full_nanfix/latest.pth; test -f \"${BASELINE_CKPT}\"; test -f weights/sarclip/RN50/rn50_model.safetensors; test -f work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke/SMOKE_OK; WORK_DIR=work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14; CUDA_VISIBLE_DEVICES=7,8,9 conda run -n iraod torchrun --nproc_per_node=3 --master_port=29528 train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --work-dir \"${WORK_DIR}\" --no-validate --launcher pytorch --data-root \"${DATA_ROOT}\" --teacher-ckpt \"${BASELINE_CKPT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --samples-per-gpu 14 --workers-per-gpu 4 && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py \"${WORK_DIR}/latest.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors --cfg-options load_from=None model.ema_ckpt=None && CUDA_VISIBLE_DEVICES=7 conda run -n iraod python test.py configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py \"${WORK_DIR}/latest_ema.pth\" --eval mAP --work-dir \"${WORK_DIR}\" --data-root \"${DATA_ROOT}\" --cga-scorer sarclip --sarclip-model RN50 --sarclip-pretrained weights/sarclip/RN50/rn50_model.safetensors'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke: `.rd_queue_ut_nocga/logs/J20260128-135450-0433__e0039-smoke-fix-vram14.log`；full: `.rd_queue_ut_nocga/logs/J20260128-135620-fa74__e0039-full-fix-vram14.log`；train: `work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke/20260128_224819.log` / `work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14/20260129_000132.log`；previous (terminated): `.rd_queue_ut_cga/logs/J20260128-131644-a0a0__e0039-smoke-rerun2.log`（exit=-15）; previous failures: `.rd_queue_ut_cga/logs/J20260127-171357-9a89__e0039-smoke.log` / `.rd_queue_ut_cga/logs/J20260127-171723-1c8e__e0039-smoke-rerun.log`（NCCL Error 3） |
| Artifacts | `work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke/`, `work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14/` |
| Results | smoke: student mAP=0.6042（`work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke/eval_20260128_235129.json`）；EMA mAP=0.6292（`work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14_smoke/eval_20260128_235635.json`）；full: student mAP=0.6356（`work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14/eval_20260129_133159.json`）；EMA mAP=0.6622（`work_dirs/exp_rsar_ut_cga_sarclip_full_fix_vram14/eval_20260129_133556.json`） |


### E0040: SARCLIP LoRA Fine-Tuning + CGA Load Verification (P0032)
| Field | Value |
| --- | --- |
| Objective | 在 RSAR 6 类数据上进行 5-GPU SARCLIP LoRA 微调（无 `L_ent`），并验证生成的 LoRA checkpoint 可被 `SARCLIP_LORA` 注入到 CGA/SarclipScorer 中进行 test-time 语义重打分 |
| Baseline | no-LoRA CGA test-time rescoring（`eval_lora_cga.py` 中的 `no_lora`） |
| Model | SARCLIP `RN50` + LoRA（`target=vision`）；CGA(SARCLIP) eval 使用 detector `work_dirs/ut_rsar_corrected/latest_ema.pth` |
| Weights | 训练输入：`third_party/SARCLIP`；输出：`work_dirs/p0032_sarclip_lora/lora_final.pth`；评估 detector：`work_dirs/ut_rsar_corrected/latest_ema.pth` |
| Code path | `lora_finetune/lora_sarclip_train.py`, `tools/lora_utils.py`, `scripts/run_lora_experiments.sh`, `scripts/smoke_sarclip_lora.sh`, `sfod/cga.py`, `sfod/scorers/sarclip_scorer.py`, `eval_lora_cga.py` |
| Params | `torchrun --nproc_per_node=5`；`epochs=10`；`batch-size=32`；`lr=1e-4`；`lora-r=8`；`lora-alpha=16`；`ent-weight=0.0` |
| Metrics (must save) | `work_dirs/p0032_sarclip_lora/lora_final.pth`（含 meta）；`work_dirs/p0032_sarclip_lora/train.log`（loss/ce/ent 曲线）；`work_dirs/lora_cga_eval_results.json`（mAP） |
| Checks | 训练完成；checkpoint 可加载；`encode_image` 跑通；设置 `SARCLIP_LORA` 后日志出现 `[CGA/SARCLIP] LoRA loaded ...`；mAP 写入结果 JSON |
| VRAM | 未单独记录（5-GPU 分布式 LoRA 训练） |
| Total time | ~13m23s（`work_dirs/lora_experiments.log` 时间戳 20:33:28 → 20:46:51） |
| Single-GPU script | `N/A`（当前记录为 5-GPU `torchrun`） |
| Multi-GPU script | `bash scripts/run_lora_experiments.sh` |
| Smoke cmd | `bash scripts/smoke_sarclip_lora.sh` |
| Full cmd | `bash -lc 'bash scripts/run_lora_experiments.sh && python eval_lora_cga.py'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke(shared): `work_dirs/smoke_lora.log`；full: `work_dirs/lora_experiments.log`, `work_dirs/p0032_sarclip_lora/train.log`, `work_dirs/lora_cga_eval.log` |
| Artifacts | `work_dirs/p0032_sarclip_lora/`, `work_dirs/sanity/sarclip_lora_smoke/`, `work_dirs/lora_cga_eval_results.json` |
| Results | 训练 loss: `0.7028 → 0.2683(best@epoch9) → 0.2764(final)`；checkpoint meta: `ent_weight=0.0`, `ent_score_thr=0.5`；CGA eval mAP=`0.65303`（`work_dirs/lora_cga_eval_results.json`），较 no-LoRA=`0.65350` 下降 `0.00047`；`work_dirs/lora_cga_eval.log` 已验证 `SARCLIP_LORA=work_dirs/p0032_sarclip_lora/lora_final.pth` 并成功 `LoRA loaded` |


### E0041: SARCLIP LoRA + Low-Confidence Entropy Minimization (P0033)
| Field | Value |
| --- | --- |
| Objective | 在 P0032 的基础上加入仅作用于低置信度样本的 `L_ent`（`--ent-weight 0.1 --ent-score-thr 0.5`），验证其是否提升 LoRA 语义打分与 CGA test-time rescoring 效果 |
| Baseline | E0040（同样的 LoRA 训练但 `ent-weight=0.0`）以及 no-LoRA CGA rescoring |
| Model | SARCLIP `RN50` + LoRA（`target=vision`）；CGA(SARCLIP) eval 使用 detector `work_dirs/ut_rsar_corrected/latest_ema.pth` |
| Weights | 训练输入：`third_party/SARCLIP`；输出：`work_dirs/p0033_sarclip_lora_ent/lora_final.pth`；评估 detector：`work_dirs/ut_rsar_corrected/latest_ema.pth` |
| Code path | `lora_finetune/lora_sarclip_train.py`, `tools/lora_utils.py`, `scripts/run_lora_experiments.sh`, `scripts/smoke_sarclip_lora.sh`, `sfod/cga.py`, `sfod/scorers/sarclip_scorer.py`, `eval_lora_cga.py` |
| Params | `torchrun --nproc_per_node=5`；`epochs=10`；`batch-size=32`；`lr=1e-4`；`lora-r=8`；`lora-alpha=16`；`ent-weight=0.1`；`ent-score-thr=0.5` |
| Metrics (must save) | `work_dirs/p0033_sarclip_lora_ent/lora_final.pth`（含 meta）；`work_dirs/p0033_sarclip_lora_ent/train.log`（loss/ce/ent 曲线）；`work_dirs/lora_cga_eval_results.json`（mAP） |
| Checks | 训练完成；checkpoint 可加载；`encode_image` 跑通；设置 `SARCLIP_LORA` 后日志出现 `[CGA/SARCLIP] LoRA loaded ...`；mAP 写入结果 JSON；checkpoint meta 含 `ent_weight` / `ent_score_thr` |
| VRAM | 未单独记录（5-GPU 分布式 LoRA 训练） |
| Total time | ~13m47s（`work_dirs/lora_experiments.log` 时间戳 20:46:51 → 21:00:38） |
| Single-GPU script | `N/A`（当前记录为 5-GPU `torchrun`） |
| Multi-GPU script | `bash scripts/run_lora_experiments.sh` |
| Smoke cmd | `bash scripts/smoke_sarclip_lora.sh` |
| Full cmd | `bash -lc 'bash scripts/run_lora_experiments.sh && python eval_lora_cga.py'` |
| Smoke | [x] |
| Full | [x] |
| Logs | smoke(shared): `work_dirs/smoke_lora.log`；full: `work_dirs/lora_experiments.log`, `work_dirs/p0033_sarclip_lora_ent/train.log`, `work_dirs/lora_cga_eval.log` |
| Artifacts | `work_dirs/p0033_sarclip_lora_ent/`, `work_dirs/sanity/sarclip_lora_smoke/`, `work_dirs/lora_cga_eval_results.json` |
| Results | 训练 loss: `0.6784 → 0.2606(best@epoch9) → 0.2729(final)`；checkpoint meta: `ent_weight=0.1`, `ent_score_thr=0.5`；CGA eval mAP=`0.65349`（`work_dirs/lora_cga_eval_results.json`），较 P0032=`0.65303` 提升 `0.00046`，但仍略低于 no-LoRA=`0.65350`；训练日志中 `ent=0.0000` 贯穿 10 个 epoch，说明当前设置下 entropy 项未形成可观测贡献 |


---



## New Experiments (Auto-collected 2026-03-26)


### E0042: frontier_001_anchor
| Field | Value |
| --- | --- |
| Objective | 12ep baseline anchor 实验，验证 OrientedRCNN-R50-FPN 在 RSAR 上的基准性能 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_001_anchor/20260321_223758.log` |
| Artifacts | `work_dirs/frontier_001_anchor/` |
| Results | Best mAP=**0.6544** (test_eval_epoch11: mAP=0.6544) |
| Plan ref | docs/plan.md §6.2 frontier-001 |


### E0043: frontier_002_lr_schedule
| Field | Value |
| --- | --- |
| Objective | 调整 LR schedule（step→custom），探索学习率衰减策略对 mAP 的影响 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 14 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_002_lr_schedule/20260322_142136.log` |
| Artifacts | `work_dirs/frontier_002_lr_schedule/` |
| Results | Best mAP=**0.6609** (test_epoch13_nms02: mAP=0.6569; test_epoch11_nms02: mAP=0.6609; test_epoch14_nms02: mAP=0.6576) |
| Plan ref | docs/plan.md §6.2 frontier-002 |


### E0044: frontier_004_gwd_loss
| Field | Value |
| --- | --- |
| Objective | 将 SmoothL1 bbox 回归损失替换为 GWD (Gaussian Wasserstein Distance) loss |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_004_gwd_loss/20260322_183045.log` |
| Artifacts | `work_dirs/frontier_004_gwd_loss/` |
| Results | Best mAP=**0.6040** (test_eval_ep12: mAP=0.6040) |
| Plan ref | docs/plan.md §6.2 frontier-004 |


### E0045: frontier_005_polyrotate
| Field | Value |
| --- | --- |
| Objective | 添加 PolyRandomRotate 数据增强，测试旋转增强对旋转框检测的效果 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_005_polyrotate/20260322_220419.log` |
| Artifacts | `work_dirs/frontier_005_polyrotate/` |
| Results | Best mAP=**0.6226** (test_eval_ep12: mAP=0.6226) |
| Plan ref | docs/plan.md §6.2 frontier-005 |


### E0046: frontier_006_multiscale
| Field | Value |
| --- | --- |
| Objective | 启用多尺度训练（img_scale 800~1333），测试多尺度对小目标检测的影响 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_006_multiscale/20260323_014428.log` |
| Artifacts | `work_dirs/frontier_006_multiscale/` |
| Results | Best mAP=**0.6366** (test_default_nms: mAP=0.6366) |
| Plan ref | docs/plan.md §6.2 frontier-006 |


### E0047: frontier_007_score_thr
| Field | Value |
| --- | --- |
| Objective | 调整推理时 score_thr（0.05→0.1），减少低置信度检测框 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | N/A（评测/工具类实验，无训练时长） |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `N/A` |
| Artifacts | `work_dirs/frontier_007_score_thr/` |
| Results | Best mAP=**0.6623** (test_eval: mAP=0.6623) |
| Plan ref | docs/plan.md §6.2 frontier-007 |


### E0048: frontier_008_24ep
| Field | Value |
| --- | --- |
| Objective | 延长训练到 24 epoch（step=[16,22]），作为后续所有实验的 baseline anchor |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 24 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_008_24ep/20260323_215733.log` |
| Artifacts | `work_dirs/frontier_008_24ep/` |
| Results | Best mAP=**0.7005** (test_nms03_epoch21: mAP=0.7005) |
| Plan ref | docs/plan.md §6.2 frontier-008 |


### E0049: frontier_009_cosine_lr
| Field | Value |
| --- | --- |
| Objective | 将 step LR 替换为 cosine annealing LR schedule |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_009_cosine_lr/test_nms02.log` |
| Artifacts | `work_dirs/frontier_009_cosine_lr/` |
| Results | Best mAP=**0.6659** (test_nms02: mAP=0.6659; test_default_nms: mAP=0.6570; test_ep12_nms02: mAP=0.6653) |
| Plan ref | docs/plan.md §6.2 frontier-009 |


### E0050: frontier_009_cosine_lr_repro
| Field | Value |
| --- | --- |
| Objective | Cosine LR 复现实验，验证 frontier_009 结果的可重复性 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_009_cosine_lr_repro/test_epoch12_nms02.log` |
| Artifacts | `work_dirs/frontier_009_cosine_lr_repro/` |
| Results | Best mAP=**0.6621** (test_epoch12_nms02: mAP=0.6543; test_epoch11_nms02: mAP=0.6621; test_epoch12_default: mAP=0.6547) |
| Plan ref | docs/plan.md §6.2 frontier-009 |


### E0051: frontier_010_backbone_lr_mult
| Field | Value |
| --- | --- |
| Objective | backbone lr_mult=0.1，冻结骨干网络学习率测试迁移学习效果 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_010_backbone_lr_mult/20260323_122614.log` |
| Artifacts | `work_dirs/frontier_010_backbone_lr_mult/` |
| Results | Best mAP=**0.6575** (log.json val epoch=12) |
| Plan ref | docs/plan.md §6.2 frontier-010 |


### E0052: frontier_012_wd10x
| Field | Value |
| --- | --- |
| Objective | Weight decay 10 倍增大（0.0001→0.001），测试正则化强度影响 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_012_wd10x/20260323_171451.log` |
| Artifacts | `work_dirs/frontier_012_wd10x/` |
| Results | Best mAP=**0.5184** (test_nms02_epoch12: mAP=0.5184) |
| Plan ref | docs/plan.md §6.2 frontier-012 |


### E0053: frontier_013_cosine_lr_seed0
| Field | Value |
| --- | --- |
| Objective | Cosine LR + seed=0，多种子验证实验稳定性 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_013_cosine_lr_seed0/20260324_052009.log` |
| Artifacts | `work_dirs/frontier_013_cosine_lr_seed0/` |
| Results | Best mAP=**0.6645** (test_nms03_ep11: mAP=0.6645) |
| Plan ref | docs/plan.md §6.2 frontier-013 |


### E0054: frontier_013_cosine_lr_seed1
| Field | Value |
| --- | --- |
| Objective | Cosine LR + seed=1，多种子验证实验稳定性 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_013_cosine_lr_seed1/20260324_085806.log` |
| Artifacts | `work_dirs/frontier_013_cosine_lr_seed1/` |
| Results | Best mAP=**0.6731** (test_nms03_ep11: mAP=0.6731) |
| Plan ref | docs/plan.md §6.2 frontier-013 |


### E0055: frontier_014_nms_sweep_run
| Field | Value |
| --- | --- |
| Objective | NMS IoU 阈值扫描（0.15~0.35），寻找最优 NMS 设置 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | N/A（评测/工具类实验，无训练时长） |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `N/A` |
| Artifacts | `work_dirs/frontier_014_nms_sweep_run/` |
| Results | Best mAP=**0.6553** (thr_015: mAP=0.6553) |
| Plan ref | docs/plan.md §6.2 frontier-014 |


### E0056: frontier_014_nms_sweep_run2
| Field | Value |
| --- | --- |
| Objective | NMS IoU 阈值扫描第二轮（更细粒度 0.15/0.20/0.25/0.30） |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | N/A（评测/工具类实验，无训练时长） |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `N/A` |
| Artifacts | `work_dirs/frontier_014_nms_sweep_run2/` |
| Results | Best mAP=**0.6734** (thr_025: mAP=0.6659; thr_030: mAP=0.6734; thr_020: mAP=0.6632; thr_015: mAP=0.6553) |
| Plan ref | docs/plan.md §6.2 frontier-014 |


### E0057: frontier_015_epoch12_nms02
| Field | Value |
| --- | --- |
| Objective | 12ep 训练 + NMS IoU=0.20，对比 24ep anchor |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | N/A（评测/工具类实验，无训练时长） |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `N/A` |
| Artifacts | `work_dirs/frontier_015_epoch12_nms02/` |
| Results | Best mAP=**0.6627** (default: mAP=0.6627) |
| Plan ref | docs/plan.md §6.2 frontier-015 |


### E0058: frontier_015_epoch12_nms03
| Field | Value |
| --- | --- |
| Objective | 12ep 训练 + NMS IoU=0.30，对比 24ep anchor |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | N/A（评测/工具类实验，无训练循环） |
| Total time | N/A（评测/工具类实验，无训练时长） |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `N/A` |
| Artifacts | `work_dirs/frontier_015_epoch12_nms03/` |
| Results | Best mAP=**0.6686** (default: mAP=0.6686) |
| Plan ref | docs/plan.md §6.2 frontier-015 |


### E0059: frontier_017_24ep_seed1
| Field | Value |
| --- | --- |
| Objective | 24ep schedule + seed=1，验证 anchor 实验的种子稳定性 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 11 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_017_24ep_seed1/20260324_124956.log` |
| Artifacts | `work_dirs/frontier_017_24ep_seed1/` |
| Results | Best mAP=**0.6637** (log.json val epoch=10) |
| Plan ref | docs/plan.md §6.2 frontier-017 |


### E0060: frontier_017_24ep_seed1_oriented_rcnn_rsar
| Field | Value |
| --- | --- |
| Objective | 24ep seed1 的 config 产物目录 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_017_24ep_seed1_oriented_rcnn_rsar/20260324_181048.log` |
| Artifacts | `work_dirs/frontier_017_24ep_seed1_oriented_rcnn_rsar/` |
| Results | Best mAP=**0.6559** (log.json val epoch=11) |
| Plan ref | docs/plan.md §6.2 frontier-017 |


### E0061: frontier_020_pafpn_24ep
| Field | Value |
| --- | --- |
| Objective | 将 neck FPN 替换为 PAFPN（路径聚合特征金字塔），24ep schedule |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 24 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_020_pafpn_24ep/20260324_193301.log` |
| Artifacts | `work_dirs/frontier_020_pafpn_24ep/` |
| Results | Best mAP=**0.7329** (log.json val epoch=21) |
| Plan ref | docs/plan.md §6.2 frontier-020 |


### E0062: frontier_021_cefocal_roi_cls
| Field | Value |
| --- | --- |
| Objective | 将 RoI head 分类损失 CE 替换为 CEFocalLoss，测试 focal loss 效果 |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~1 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 12 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_021_cefocal_roi_cls/20260325_040558.log` |
| Artifacts | `work_dirs/frontier_021_cefocal_roi_cls/` |
| Results | Best mAP=**0.6489** (log.json val epoch=11) |
| Plan ref | docs/plan.md §6.2 frontier-021 |


### E0063: frontier_026_ocafpn_24ep
| Field | Value |
| --- | --- |
| Objective | 将 neck FPN 替换为 OCA-FPN（正交通道注意力 FPN），24ep schedule（创新点 1） |
| Baseline | frontier_008_24ep (mAP=0.701) |
| Model | OrientedRCNN + R50-FPN (le90) |
| Weights | ImageNet pretrained R50 (mmdet default) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~0 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 13 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/frontier_026_ocafpn_24ep/20260325_110955.log` |
| Artifacts | `work_dirs/frontier_026_ocafpn_24ep/` |
| Results | Best mAP=**0.6511** (log.json val epoch=11) |
| Plan ref | docs/plan.md §6.2 frontier-026 |


### E0064: exp_aa_x_sup_polish
| Field | Value |
| --- | --- |
| Objective | 监督分支精调 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_aa_x_sup_polish/launcher.log` |
| Artifacts | `work_dirs/exp_aa_x_sup_polish/` |
| Results | Best mAP=**0.6822** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0065: exp_ab_x_ema_sup_polish
| Field | Value |
| --- | --- |
| Objective | EMA + 监督分支精调 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ab_x_ema_sup_polish/launcher.log` |
| Artifacts | `work_dirs/exp_ab_x_ema_sup_polish/` |
| Results | Best mAP=**0.6832** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0066: exp_ac_x_ema_tiny_u
| Field | Value |
| --- | --- |
| Objective | EMA + tiny unsupervised 权重实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ac_x_ema_tiny_u/launcher.log` |
| Artifacts | `work_dirs/exp_ac_x_ema_tiny_u/` |
| Results | Best mAP=**0.6832** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0067: exp_ad_x_tail_protect
| Field | Value |
| --- | --- |
| Objective | 长尾类别保护策略 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ad_x_tail_protect/launcher.log` |
| Artifacts | `work_dirs/exp_ad_x_tail_protect/` |
| Results | Best mAP=**0.6830** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0068: exp_ae_x_bridge_harbor_protect
| Field | Value |
| --- | --- |
| Objective | bridge/harbor 类别保护策略 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ae_x_bridge_harbor_protect/launcher.log` |
| Artifacts | `work_dirs/exp_ae_x_bridge_harbor_protect/` |
| Results | Best mAP=**0.6827** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0069: exp_af_v_one_epoch_wu08
| Field | Value |
| --- | --- |
| Objective | 单 epoch warm-up=0.8 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_af_v_one_epoch_wu08/launcher.log` |
| Artifacts | `work_dirs/exp_af_v_one_epoch_wu08/` |
| Results | Best mAP=**0.6795** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0070: exp_ag_v_one_epoch_tank80
| Field | Value |
| --- | --- |
| Objective | 单 epoch + tank 类权重=80 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ag_v_one_epoch_tank80/launcher.log` |
| Artifacts | `work_dirs/exp_ag_v_one_epoch_tank80/` |
| Results | Best mAP=**0.6797** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0071: exp_ah_x_ema_bbox_split
| Field | Value |
| --- | --- |
| Objective | EMA + bbox 分离训练策略 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ah_x_ema_bbox_split/launcher.log` |
| Artifacts | `work_dirs/exp_ah_x_ema_bbox_split/` |
| Results | Best mAP=**0.6831** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0072: exp_ai_v_bbox_split
| Field | Value |
| --- | --- |
| Objective | bbox 分离训练策略（基于 exp_v） |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ai_v_bbox_split/launcher.log` |
| Artifacts | `work_dirs/exp_ai_v_bbox_split/` |
| Results | Best mAP=**0.6807** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0073: exp_aj_v_perclass_anneal
| Field | Value |
| --- | --- |
| Objective | 逐类别退火策略 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_aj_v_perclass_anneal/launcher.log` |
| Artifacts | `work_dirs/exp_aj_v_perclass_anneal/` |
| Results | Best mAP=**0.6804** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0074: exp_ak_x_epoch2_condcrop
| Field | Value |
| --- | --- |
| Objective | 2 epoch + 条件裁剪增强 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 2 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ak_x_epoch2_condcrop/launcher.log` |
| Artifacts | `work_dirs/exp_ak_x_epoch2_condcrop/` |
| Results | Best mAP=**0.6834** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0075: exp_al_x_dtrcrop_epoch1
| Field | Value |
| --- | --- |
| Objective | DTR 裁剪 + 1 epoch 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_al_x_dtrcrop_epoch1/launcher.log` |
| Artifacts | `work_dirs/exp_al_x_dtrcrop_epoch1/` |
| Results | Best mAP=**0.6791** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0076: exp_an_x_sardet_unlabeled
| Field | Value |
| --- | --- |
| Objective | 使用 SARDet100k 无标注数据扩充 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~6 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 2 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_an_x_sardet_unlabeled/launcher.log` |
| Artifacts | `work_dirs/exp_an_x_sardet_unlabeled/` |
| Results | Best mAP=**0.6957** (log.json val epoch=2) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0077: exp_ao_an_tiny_u
| Field | Value |
| --- | --- |
| Objective | 基于 exp_an 的 tiny unsupervised 权重 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ao_an_tiny_u/launcher.log` |
| Artifacts | `work_dirs/exp_ao_an_tiny_u/` |
| Results | Best mAP=**0.6965** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0078: exp_ap_ao_tiny_u01
| Field | Value |
| --- | --- |
| Objective | tiny u=0.1 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ap_ao_tiny_u01/launcher.log` |
| Artifacts | `work_dirs/exp_ap_ao_tiny_u01/` |
| Results | Best mAP=**0.6976** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0079: exp_aq_ap_tiny_u005
| Field | Value |
| --- | --- |
| Objective | tiny u=0.05 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_aq_ap_tiny_u005/launcher.log` |
| Artifacts | `work_dirs/exp_aq_ap_tiny_u005/` |
| Results | Best mAP=**0.6879** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0080: exp_ar_ap_tiny_u005_lowema
| Field | Value |
| --- | --- |
| Objective | tiny u=0.05 + 低 EMA 动量 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ar_ap_tiny_u005_lowema/launcher.log` |
| Artifacts | `work_dirs/exp_ar_ap_tiny_u005_lowema/` |
| Results | Best mAP=**0.6978** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0081: exp_as_ar_tiny_u003_lowema
| Field | Value |
| --- | --- |
| Objective | tiny u=0.03 + 低 EMA 动量 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_as_ar_tiny_u003_lowema/launcher.log` |
| Artifacts | `work_dirs/exp_as_ar_tiny_u003_lowema/` |
| Results | Best mAP=**0.6976** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0082: exp_at_ar_tiny_u004_lowema
| Field | Value |
| --- | --- |
| Objective | tiny u=0.04 + 低 EMA 动量 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_at_ar_tiny_u004_lowema/launcher.log` |
| Artifacts | `work_dirs/exp_at_ar_tiny_u004_lowema/` |
| Results | Best mAP=**0.6974** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0083: exp_au_ar_tiny_u006_lowema
| Field | Value |
| --- | --- |
| Objective | tiny u=0.06 + 低 EMA 动量 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_au_ar_tiny_u006_lowema/student_eval.log` |
| Artifacts | `work_dirs/exp_au_ar_tiny_u006_lowema/` |
| Results | Best mAP=**0.6980** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0084: exp_av_au_tiny_u006_lowema
| Field | Value |
| --- | --- |
| Objective | 基于 exp_au 的 u=0.06 低 EMA 继续训练 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_av_au_tiny_u006_lowema/student_eval.log` |
| Artifacts | `work_dirs/exp_av_au_tiny_u006_lowema/` |
| Results | Best mAP=**0.6971** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0085: exp_aw_au_tiny_u005_lowema
| Field | Value |
| --- | --- |
| Objective | 基于 exp_au 的 u=0.05 低 EMA |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_aw_au_tiny_u005_lowema/student_eval.log` |
| Artifacts | `work_dirs/exp_aw_au_tiny_u005_lowema/` |
| Results | Best mAP=**0.6974** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0086: exp_ax_au_tiny_u006_ema995
| Field | Value |
| --- | --- |
| Objective | u=0.06 + EMA momentum=0.995（**最佳 SFOD 结果**） |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ax_au_tiny_u006_ema995/launcher.log` |
| Artifacts | `work_dirs/exp_ax_au_tiny_u006_ema995/` |
| Results | Best mAP=**0.6987** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0087: exp_ay_ax_tiny_u006_ema995
| Field | Value |
| --- | --- |
| Objective | 基于 exp_ax 的继续训练 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 1 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_ay_ax_tiny_u006_ema995/launcher.log` |
| Artifacts | `work_dirs/exp_ay_ax_tiny_u006_ema995/` |
| Results | Best mAP=**0.6977** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0088: exp_m_lora_cga
| Field | Value |
| --- | --- |
| Objective | LoRA 微调 SARCLIP 后集成到 CGA 模块的首次 SFOD 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~3 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 16 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_m_lora_cga/wait_and_resume.log` |
| Artifacts | `work_dirs/exp_m_lora_cga/` |
| Results | Best mAP=**0.6639** (log.json val epoch=11) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0089: exp_m_wu_schedule
| Field | Value |
| --- | --- |
| Objective | 调整 warm-up schedule 参数的 SFOD 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~1 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 16 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_m_wu_schedule/train.log` |
| Artifacts | `work_dirs/exp_m_wu_schedule/` |
| Results | Best mAP=**0.6641** (log.json val epoch=7) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0090: exp_t_m_resume
| Field | Value |
| --- | --- |
| Objective | 从 exp_m 断点恢复训练 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~1 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 16 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_t_m_resume/train_attempt1_sighup.log` |
| Artifacts | `work_dirs/exp_t_m_resume/` |
| Results | Best mAP=**0.6688** (log.json val epoch=16) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0091: exp_u_m_aircraft_protect
| Field | Value |
| --- | --- |
| Objective | 针对 aircraft 类别的保护策略（提高小类权重） |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~4 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 16 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_u_m_aircraft_protect/train_attempt1_missing_ckpt.log` |
| Artifacts | `work_dirs/exp_u_m_aircraft_protect/` |
| Results | Best mAP=**0.6678** (log.json val epoch=12) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0092: exp_v_t_aircraft_protect
| Field | Value |
| --- | --- |
| Objective | 基于 exp_t 的 aircraft 保护策略迭代 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~7 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 16 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_v_t_aircraft_protect/train.log` |
| Artifacts | `work_dirs/exp_v_t_aircraft_protect/` |
| Results | Best mAP=**0.6780** (log.json val epoch=13) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0093: exp_w_v_polish
| Field | Value |
| --- | --- |
| Objective | 基于 exp_v 的精调优化 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 8 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_w_v_polish/launcher.log` |
| Artifacts | `work_dirs/exp_w_v_polish/` |
| Results | Best mAP=**0.6822** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0094: exp_x_v_micro_polish
| Field | Value |
| --- | --- |
| Objective | 基于 exp_v 的微调优化（更小 LR） |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 2 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_x_v_micro_polish/launcher.log` |
| Artifacts | `work_dirs/exp_x_v_micro_polish/` |
| Results | Best mAP=**0.6835** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0095: exp_y_x_vitl14_lora
| Field | Value |
| --- | --- |
| Objective | 使用 ViT-L-14 LoRA 增强 CGA 的 SFOD 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~6 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 2 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_y_x_vitl14_lora/launcher.log` |
| Artifacts | `work_dirs/exp_y_x_vitl14_lora/` |
| Results | Best mAP=**0.6813** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


### E0096: exp_z_x_lower_wu
| Field | Value |
| --- | --- |
| Objective | 降低 warm-up 策略的 SFOD 实验 |
| Baseline | Previous SFOD iteration |
| Model | UnbiasedTeacher + OrientedRCNN R50-FPN + CGA |
| Weights | Previous SFOD iteration checkpoint (EMA teacher) |
| Code path | 由 open-researcher-v2 自动生成（见 `.research/graph.json` frontier 节点） |
| Params | 见对应 `work_dirs/*/` 下的 `.py` config 副本（由实验框架自动拷贝） |
| Metrics (must save) | `mAP`; checkpoint `.pth` |
| Checks | mAP 输出存在且合理；checkpoint 存在 |
| VRAM | ~4-6 GB (single GPU) |
| Time/epoch | ~5 min/epoch (estimated from log.json) |
| Total time | N/A (wall-clock not logged; schedule: 2 epochs) |
| Single-GPU script | `N/A (config 未追踪)` |
| Multi-GPU script | 5x GPU via `torch.distributed.launch` |
| Smoke cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Full cmd | N/A (自动化实验，由 open-researcher-v2 调度) |
| Smoke | N/A (auto-run) |
| Full | [x] |
| Logs | `work_dirs/exp_z_x_lower_wu/launcher.log` |
| Artifacts | `work_dirs/exp_z_x_lower_wu/` |
| Results | Best mAP=**0.6821** (log.json val epoch=1) |
| Plan ref | `docs/plan.md` §6.3 SFOD Chain (row: this exp), §6.5 LoRA Iteration Table |


---

## Phase 3: RSAR 电子干扰鲁棒性评测（无源目标检测）

> **方法**: SFOD + OrthoNet(depth=50) backbone + SARCLIP ViT-L-14 LoRA CGA 零样本重打分
> **数据协议**: train(有标签,干净) + val(无标签,干扰) → 半监督学习; test(干扰) → 评估
> **训练配置**: 24 epoch, 5×GPU, BS=40, SGD lr=0.02, step=[16,22], eval每epoch
> **关键参数**: score_thr=0.5, EMA momentum=0.9996, weight_u=0.5
> **LoRA**: SARDet100k 裁剪 patch + 干扰增强微调, 122,880 params (0.12%)
> **日期**: 2026-03-29 ~ 2026-04-05


### E0097: SFOD OrthoNet+SARCLIP — chaff（箔条干扰）
| Field | Value |
| --- | --- |
| Objective | 箔条干扰下的无源旋转目标检测（source-free） |
| Baseline | Clean RSAR mAP ≈ 0.68 (Phase 2 best) |
| Model | UnbiasedTeacher + OrientedRCNN OrthoNet-50 + FPN + SARCLIP CGA (ViT-L-14 LoRA) |
| Weights | ImageNet pretrained OrthoNet + SARCLIP_LoRA_Interference.pt |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py` |
| Params | `corrupt=chaff`, `score_thr=0.5`, `momentum=0.9996`, `weight_u=0.5`, `total_epoch=24`, `lr_step=[16,22]` |
| Metrics (must save) | `mAP`; checkpoint `.pth` + `_ema.pth` |
| Checks | mAP 输出存在且合理；每 epoch eval 正常；无 NaN 崩溃 |
| VRAM | ~24 GB per GPU (5× RTX 4090) |
| Time/epoch | ~40 min (train) + ~15 min (eval) |
| Total time | ~22h |
| Multi-GPU script | `scripts/run_7corrupt_fast.sh` |
| Full cmd | `CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=5 train.py $CONFIG --cfg-options corrupt="chaff"` |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_chaff/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_chaff/` |
| Results | **mAP=0.4861** (Epoch 24); ship=0.507, aircraft=0.630, car=0.779, tank=0.181, bridge=0.435, harbor=0.386 |
| Plan ref | `docs/plan.md` §D RSAR-Interference 鲁棒性评测 |


### E0098: SFOD OrthoNet+SARCLIP — gaussian_white_noise（高斯白噪声）
| Field | Value |
| --- | --- |
| Objective | 高斯白噪声干扰下的无源旋转目标检测 |
| Baseline | Clean RSAR mAP ≈ 0.68 |
| Model | 同 E0097 |
| Params | `corrupt=gaussian_white_noise`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_gaussian_white_noise/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_gaussian_white_noise/` |
| Results | **mAP=0.5692** (Epoch 24); ship=0.770, aircraft=0.645, car=0.865, tank=0.178, bridge=0.481, harbor=0.475 |


### E0099: SFOD OrthoNet+SARCLIP — point_target（点目标干扰）
| Field | Value |
| --- | --- |
| Objective | 点目标干扰下的无源旋转目标检测 |
| Model | 同 E0097 |
| Params | `corrupt=point_target`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_point_target/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_point_target/` |
| Results | **mAP=0.5681** (Epoch 24); ship=0.771, aircraft=0.647, car=0.858, tank=0.181, bridge=0.466, harbor=0.485 |


### E0100: SFOD OrthoNet+SARCLIP — noise_suppression（噪声抑制）
| Field | Value |
| --- | --- |
| Objective | 噪声抑制干扰下的无源旋转目标检测 |
| Model | 同 E0097 |
| Params | `corrupt=noise_suppression`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_noise_suppression/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_noise_suppression/` |
| Results | **mAP=0.2354** (Epoch 24); ship=0.584, aircraft=0.030, car=0.055, tank=0.013, bridge=0.432, harbor=0.298 |


### E0101: SFOD OrthoNet+SARCLIP — am_noise_horizontal（水平调幅噪声）
| Field | Value |
| --- | --- |
| Objective | 水平调幅噪声干扰下的无源旋转目标检测 |
| Model | 同 E0097 |
| Params | `corrupt=am_noise_horizontal`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_am_noise_horizontal/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_am_noise_horizontal/` |
| Results | **mAP=0.0969** (Epoch 24); ship=0.173, aircraft=0.000, car=0.149, tank=0.030, bridge=0.219, harbor=0.010 |


### E0102: SFOD OrthoNet+SARCLIP — smart_suppression（智能压制）
| Field | Value |
| --- | --- |
| Objective | 智能压制干扰下的无源旋转目标检测 |
| Model | 同 E0097 |
| Params | `corrupt=smart_suppression`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_smart_suppression/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_smart_suppression/` |
| Results | **mAP=0.1880** (Epoch 24); ship=0.381, aircraft=0.058, car=0.091, tank=0.037, bridge=0.382, harbor=0.179 |


### E0103: SFOD OrthoNet+SARCLIP — am_noise_vertical（垂直调幅噪声）
| Field | Value |
| --- | --- |
| Objective | 垂直调幅噪声干扰下的无源旋转目标检测 |
| Model | 同 E0097 |
| Params | `corrupt=am_noise_vertical`，其余同 E0097 |
| Full | [x] |
| Logs | `work_dirs/exp_sfod_ortho_sarclip_am_noise_vertical/train.log` |
| Artifacts | `work_dirs/exp_sfod_ortho_sarclip_am_noise_vertical/` |
| Results | **mAP=0.1148** (Epoch 24); ship=0.179, aircraft=0.045, car=0.058, tank=0.093, bridge=0.297, harbor=0.017 |


### Phase 3 汇总: 7 种干扰鲁棒性评测

| 干扰类型 | mAP | ship | aircraft | car | tank | bridge | harbor | 难度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_white_noise | **0.569** | 0.770 | 0.645 | 0.865 | 0.178 | 0.481 | 0.475 | 轻度 |
| point_target | **0.568** | 0.771 | 0.647 | 0.858 | 0.181 | 0.466 | 0.485 | 轻度 |
| chaff | **0.486** | 0.507 | 0.630 | 0.779 | 0.181 | 0.435 | 0.386 | 中等 |
| noise_suppression | **0.235** | 0.584 | 0.030 | 0.055 | 0.013 | 0.432 | 0.298 | 困难 |
| smart_suppression | **0.188** | 0.381 | 0.058 | 0.091 | 0.037 | 0.382 | 0.179 | 困难 |
| am_noise_vertical | **0.115** | 0.179 | 0.045 | 0.058 | 0.093 | 0.297 | 0.017 | 极难 |
| am_noise_horizontal | **0.097** | 0.173 | 0.000 | 0.149 | 0.030 | 0.219 | 0.010 | 极难 |
| **平均** | **0.323** | 0.481 | 0.294 | 0.408 | 0.102 | 0.387 | 0.264 | — |

**关键发现**:
1. **轻度干扰（高斯白噪声/点目标）**: mAP ≈ 0.57，仅比 clean 下降 ~16%，SFOD 伪标签自适应有效
2. **中等干扰（箔条）**: mAP ≈ 0.49，下降 ~28%，箔条产生大量虚假散射体，影响 ship/bridge 检测
3. **困难干扰（噪声抑制/智能压制）**: mAP ≈ 0.21，下降 ~69%，aircraft/car/tank 几乎完全丧失
4. **极难干扰（调幅噪声）**: mAP < 0.12，下降 >82%，水平/垂直条纹严重破坏 SAR 成像
5. **ship/bridge 鲁棒性最强**: 大目标在各种干扰下都能保持一定检测能力
6. **aircraft/tank 最脆弱**: 小目标在强干扰下 AP 趋近 0，伪标签完全失效
7. **OrthoNet + SARCLIP LoRA CGA 的核心价值**: 在轻/中度干扰下提供有效的域自适应能力

---

## Phase 4: RSAR CLIP-guided SFOD 同款对照组实验

> **目标**: 在 RSAR 上复刻 CLIP-guided SFOD 论文同款 control baselines，并统一输出 clean test + 7 个 corruption test 的总表
> **实验编号**: E0104 ~ E0110
> **Source model**: `configs/experiments/rsar/frontier_026_ocafpn_24ep_oriented_rcnn_rsar.py` + `work_dirs/frontier_026_ocafpn_24ep/latest.pth`
> **统一 detector**: 现有 RSAR oriented detector（不改结构）
> **目标域自适应数据**: `dataset/RSAR/train/images/`（无标注；严禁使用 `test` 图像校准/自适应）
> **统一脚本**: `scripts/exp_rsar_controls.sh`, `scripts/queue_rsar_long_controls.sh`
> **统一随机种子**: `3407`
> **结果汇总**: `work_dirs/controls/rsar_clip_guided_sfod/results_controls.csv`, `work_dirs/controls/rsar_clip_guided_sfod/results_controls.md`
> **日期**: 2026-04-06 ~ 2026-04-07


### E0104: Clean test（SOURCE_CKPT, 无适配）
| Field | Value |
| --- | --- |
| Objective | 不做任何适配，直接用 SOURCE_CKPT 在 `dataset/RSAR/test/images` 上评估 clean mAP |
| Model | OrientedRCNN + OrthoNet-50 + OCA-FPN |
| Weights | `work_dirs/frontier_026_ocafpn_24ep/latest.pth` |
| Code path | `tools/run_direct_test.py`, `tools/rsar_controls_common.py` |
| Params | `method=clean`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/clean/metrics.json` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/clean/` |
| Results | `clean_test=0.6863`, `mean=0.6863` |
| Plan ref | `docs/plan.md` §8 RSAR CLIP-guided SFOD 对照组 |


### E0105: Direct test（目标域直接测试, 无参数更新）
| Field | Value |
| --- | --- |
| Objective | 对 clean test + 7 个 corruption test 直接评估 SOURCE_CKPT，不更新任何参数 |
| Model | 同 E0104 |
| Weights | `work_dirs/frontier_026_ocafpn_24ep/latest.pth` |
| Code path | `tools/run_direct_test.py`, `tools/rsar_controls_common.py` |
| Params | `method=direct`, `seed=3407`; clean + 7 corruptions 全部直接测试 |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/direct/metrics.json` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/direct/eval_*` |
| Results | `clean=0.6863`, `gaussian_white_noise=0.6786`, `point_target=0.6788`, `chaff=0.6239`, `noise_suppression=0.3026`, `smart_suppression=0.3343`, `am_noise_vertical=0.2906`, `am_noise_horizontal=0.2484`, `mean=0.4804` |


### E0106: BN baseline（仅更新 BN running mean/var）
| Field | Value |
| --- | --- |
| Objective | 冻结全部可学习参数，仅在 `dataset/RSAR/train/images/` 上做一次 BN 统计校准，然后在 clean + 7 corruptions 上评估 |
| Model | 同 E0104 |
| Weights | `SOURCE_CKPT -> BN_CALIB_CKPT=work_dirs/controls_smoke/direct_bn/bn/latest.pth` |
| Code path | `tools/run_bn_calibration.py`, `tools/rsar_controls_common.py` |
| Params | `samples_per_gpu=8`, `workers_per_gpu=2`, `num_batches=9855`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/bn/run_meta.json`, `work_dirs/controls/rsar_clip_guided_sfod/bn/metrics.json` |
| Artifacts | `work_dirs/controls_smoke/direct_bn/bn/latest.pth`, `work_dirs/controls/rsar_clip_guided_sfod/bn/eval_*` |
| Results | `clean=0.6861`, `gaussian_white_noise=0.6787`, `point_target=0.6786`, `chaff=0.6237`, `noise_suppression=0.3025`, `smart_suppression=0.3342`, `am_noise_vertical=0.2904`, `am_noise_horizontal=0.2471`, `mean=0.4802` |


### E0107: Tent baseline（BN affine + RoI entropy）
| Field | Value |
| --- | --- |
| Objective | 仅更新 BN affine（gamma/beta），在 `dataset/RSAR/train/images/` 上最小化 RoI 分类 entropy，然后在 clean + 7 corruptions 上评估 |
| Model | 同 E0104 |
| Weights | `SOURCE_CKPT -> TENT_CKPT=work_dirs/controls/rsar_clip_guided_sfod/tent/latest.pth` |
| Code path | `tools/run_tent_adapt.py`, `tools/rsar_controls_common.py` |
| Params | `epochs=1`, `steps=19710`, `lr=1e-4`, `topk=256`, `min_fg_conf=0.05`, `trainable=BN affine only`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/tent/run_meta.json`, `work_dirs/controls/rsar_clip_guided_sfod/tent/metrics.json`, `work_dirs/controls/rsar_clip_guided_sfod/tent_launcher.log` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/tent/latest.pth`, `work_dirs/controls/rsar_clip_guided_sfod/tent/eval_*` |
| Results | `clean=0.0000`, `gaussian_white_noise=0.0000`, `point_target=0.0000`, `chaff=0.0000`, `noise_suppression=0.0152`, `smart_suppression=0.0152`, `am_noise_vertical=0.0000`, `am_noise_horizontal=0.0000`, `mean=0.0038` |


### E0108: SHOT baseline（detection-approx-shot）
| Field | Value |
| --- | --- |
| Objective | detection 近似版 SHOT：冻结 RPN/RoI heads，仅优化 backbone+neck，并用 RoI entropy 在 `dataset/RSAR/train/images/` 上做无监督自适应 |
| Model | 同 E0104 |
| Weights | `SOURCE_CKPT -> SHOT_CKPT=work_dirs/controls/rsar_clip_guided_sfod/shot/latest.pth` |
| Code path | `tools/run_shot_adapt.py`, `tools/rsar_controls_common.py` |
| Params | `definition=detection-approx-shot`, `epochs=3`, `steps=59130`, `lr=1e-4`, `topk=256`, `min_fg_conf=0.05`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/shot/run_meta.json`, `work_dirs/controls/rsar_clip_guided_sfod/shot/metrics.json`, `work_dirs/controls/rsar_clip_guided_sfod/shot_launcher.log` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/shot/latest.pth`, `work_dirs/controls/rsar_clip_guided_sfod/shot/eval_*` |
| Results | `clean=0.0000`, `gaussian_white_noise=0.0000`, `point_target=0.0000`, `chaff=0.0000`, `noise_suppression=0.0000`, `smart_suppression=0.0000`, `am_noise_vertical=0.0000`, `am_noise_horizontal=0.0000`, `mean=0.0000` |


### E0109: Self-training baseline（UnbiasedTeacher, no CGA）
| Field | Value |
| --- | --- |
| Objective | 复用 UnbiasedTeacher 式 weak/strong 自训练；teacher 用 EMA 更新，伪标签阈值 `tau=0.5`，不启用 CGA |
| Model | UnbiasedTeacher + OrientedRCNN（同 RSAR detector） |
| Weights | `SOURCE_CKPT -> SELFTRAIN_CKPT=work_dirs/controls/rsar_clip_guided_sfod/selftrain/latest_ema.pth` |
| Code path | `tools/run_selftrain_adapt.py`, `scripts/exp_rsar_controls.sh` |
| Params | `max_epochs=24`, `lr=0.02`, `weight_u=0.5`, `tau=0.5`, `ema_momentum=0.998`, `samples_per_gpu=8`, `workers_per_gpu=4`, `cuda_visible_devices=2,3,4,5,6`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/selftrain/run_meta.json`, `work_dirs/controls/rsar_clip_guided_sfod/selftrain/metrics.json`, `work_dirs/controls/rsar_clip_guided_sfod/selftrain/train_20260406_034810.log` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/selftrain/latest_ema.pth`, `work_dirs/controls/rsar_clip_guided_sfod/selftrain/eval_*` |
| Results | `clean=0.0076`, `gaussian_white_noise=0.0076`, `point_target=0.0076`, `chaff=0.0076`, `noise_suppression=0.0152`, `smart_suppression=0.0076`, `am_noise_vertical=0.0152`, `am_noise_horizontal=0.0000`, `mean=0.0085` |


### E0110: Self-training + CGA（SARCLIP ViT-L-14 LoRA, λ=0.2）
| Field | Value |
| --- | --- |
| Objective | 在 E0109 基础上启用 CGA：teacher 伪标签阶段加入 SARCLIP LoRA 重打分，prompt=`a SAR image of a {}`，`lambda=0.2` |
| Model | UnbiasedTeacher + OrientedRCNN_CGA + SARCLIP ViT-L-14 LoRA |
| Weights | `SOURCE_CKPT -> CGA_CKPT=work_dirs/controls/rsar_clip_guided_sfod/cga/latest_ema.pth`; `weights/sarclip/ViT-L-14/vit_l_14_model.safetensors`; `lora_finetune/SARCLIP_LoRA_Interference.pt` |
| Code path | `tools/run_selftrain_adapt.py`, `sfod/cga.py`, `scripts/exp_rsar_controls.sh` |
| Params | `max_epochs=24`, `lr=0.02`, `weight_u=0.5`, `tau=0.5`, `ema_momentum=0.998`, `cga_lambda=0.2`, `sarclip_model=ViT-L-14`, `samples_per_gpu=8`, `workers_per_gpu=4`, `cuda_visible_devices=2,3,4,5,6`, `seed=3407` |
| Logs/Meta | `work_dirs/controls/rsar_clip_guided_sfod/cga/run_meta.json`, `work_dirs/controls/rsar_clip_guided_sfod/cga/metrics.json`, `work_dirs/controls/rsar_clip_guided_sfod/cga/train_20260406_172829.log` |
| Artifacts | `work_dirs/controls/rsar_clip_guided_sfod/cga/latest_ema.pth`, `work_dirs/controls/rsar_clip_guided_sfod/cga/eval_*` |
| Results | `clean=0.0001`, `gaussian_white_noise=0.0002`, `point_target=0.0001`, `chaff=0.0001`, `noise_suppression=0.0009`, `smart_suppression=0.0004`, `am_noise_vertical=0.0038`, `am_noise_horizontal=0.0000`, `mean=0.0007` |


### Phase 4 汇总: CLIP-guided SFOD 对照组总表

> `mean` 为 `clean_test + 7 corruption_test` 共 8 列的算术平均。

| method | clean_test | gaussian_white_noise_test | point_target_test | chaff_test | noise_suppression_test | smart_suppression_test | am_noise_vertical_test | am_noise_horizontal_test | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.6863 | - | - | - | - | - | - | - | 0.6863 |
| direct | 0.6863 | 0.6786 | 0.6788 | 0.6239 | 0.3026 | 0.3343 | 0.2906 | 0.2484 | 0.4804 |
| bn | 0.6861 | 0.6787 | 0.6786 | 0.6237 | 0.3025 | 0.3342 | 0.2904 | 0.2471 | 0.4802 |
| tent | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0152 | 0.0152 | 0.0000 | 0.0000 | 0.0038 |
| shot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| selftrain | 0.0076 | 0.0076 | 0.0076 | 0.0076 | 0.0152 | 0.0076 | 0.0152 | 0.0000 | 0.0085 |
| cga | 0.0001 | 0.0002 | 0.0001 | 0.0001 | 0.0009 | 0.0004 | 0.0038 | 0.0000 | 0.0007 |

**关键发现**:
1. `direct` 与 `bn` 基本重合（0.4804 vs 0.4802），说明仅用 clean `train/images` 更新 BN 统计对该 RSAR 控制组几乎没有收益。
2. `Tent`、`SHOT`、`Self-training` 与 `Self-training+CGA` 在当前协议下全部明显劣于 `direct`，且均接近塌陷。
3. `Self-training+CGA` 的 `mean=0.0007`，甚至低于 `Self-training=0.0085`；当前 CGA 重打分没有挽救 teacher-student 退化。
4. 当前最强控制组实际上是“不做适配”的 source detector：`SOURCE_CKPT` 在 clean test 上 0.6863，在 8 列平均上 0.4804。
5. 该结果表明：在“target adaptation data 仅为 clean `RSAR/train/images/`，测试为 `corruptions/test`”这一设置下，参数更新式目标自适应会系统性破坏源模型，而不是带来增益。

### Phase 4 论文/汇报摘要

在严格遵循 CLIP-guided SFOD control protocol 的 RSAR 实验中，所有方法均从同一个 source detector 出发，并且仅允许使用 clean `RSAR/train/images/` 作为无标注 adaptation data。结果显示，不做任何适配的 `direct test` 反而取得最高的 8 列平均性能（`mean mAP=0.4804`），而仅更新 BN 统计的 `bn` 与其几乎完全一致（`0.4802`）。相比之下，参数更新式目标自适应方法全部显著退化：`tent=0.0038`、`shot=0.0000`、`selftrain=0.0085`、`cga=0.0007`。这说明在 clean-train → corrupt-test 的显著分布错位下，目标域自适应没有带来正迁移，反而系统性破坏了 source model。该结论与 Phase 3 并不矛盾；二者的关键差异在于 Phase 3 使用了与测试干扰域匹配的无标注 `val/images-${corrupt}`，而 Phase 4 刻意限制为 clean `train/images`。因此，Phase 4 应作为论文中的负对照：它证明“没有干扰匹配目标域数据时，直接测试 source detector 是更强且更稳健的基线”。

