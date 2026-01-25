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
| Baseline | N/A |
| Model | N/A |
| Weights | N/A |
| Code path | `tools/check_image_ann_alignment.py` |
| Params | `--exts .jpg,.jpeg,.png,.bmp,.tif,.tiff` |
| Metrics (must save) | missing/conflict 统计；CSV 报告 |
| Checks | missing=0 且 conflict=0 |
| VRAM | 0 GB |
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Model | N/A |
| Weights | N/A |
| Code path | `tools/verify_rsar_corrupt_switch.py`, `sfod/utils/patches.py`, `scripts/smoke_rsar.sh` |
| Params | `--corrupt interf_jamA`（映射到 `images-interf_jamA/`）；`CORRUPT=interf_jamA` |
| Metrics (must save) | missing/conflict 统计；CSV 报告；smoke mAP（log）；`--show-dir` 输出 |
| Checks | verify 脚本对 clean/interf 均通过；`CORRUPT=interf_jamA` smoke train/test 可运行并产出可视化 |
| VRAM | ~4 GB（smoke train/test） |
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Baseline | N/A |
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
| Baseline | N/A |
| Model | N/A |
| Weights | N/A |
| Code path | `tools/vis_random_samples.py` |
| Params | `--vis-dirs <dir1> <dir2> ...`；`--num N`；`--out-dir out` |
| Metrics (must save) | 生成的对比图片（PNG/JPG） |
| Checks | `--out-dir` 下生成 `sample_*.png`（或同名）且返回码=0 |
| VRAM | 0 GB |
| Time/epoch | N/A |
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
| Baseline | N/A |
| Model | N/A |
| Weights | N/A |
| Code path | `tools/plot_all.py` |
| Params | `--metrics-csv`；可选 `--log-json-glob`；`--out-dir` |
| Metrics (must save) | PNG 图表（mAP bar、loss/pseudo 曲线等） |
| Checks | `--out-dir` 下生成 PNG 且返回码=0 |
| VRAM | 0 GB |
| Time/epoch | N/A |
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
| Baseline | N/A |
| Model | N/A |
| Weights | N/A |
| Code path | `tools/export_experiments.py`, `README_experiments.md`, `MODEL_ZOO.md` |
| Params | `--metrics-csv`；`--out-csv` |
| Metrics (must save) | `experiments.csv` |
| Checks | `experiments.csv` 非空且包含 `git_sha` 等列；README/MODEL_ZOO 文件存在 |
| VRAM | 0 GB |
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Baseline | N/A |
| Model | SARCLIP (text/image encoder) |
| Weights | 可选；无权重时允许 random init（仅验证不崩溃） |
| Code path | `third_party/SARCLIP/sar_clip/transformer.py`, `tools/sarclip_smoke.py`, `scripts/sarclip_torch17_smoke.sh` |
| Params | `--device cpu`（避免 CUDA 兼容问题） |
| Metrics (must save) | 运行成功日志（torch 版本 + score 输出） |
| Checks | 不出现 `unexpected keyword argument 'batch_first'`；脚本退出码为 0 |
| VRAM | 0 GB (CPU) |
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
| Time/epoch | N/A |
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
