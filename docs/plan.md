# Plan

## Items
- [x] P0001: 固化环境与可复现快照
  - Summary: 记录当前可运行的 conda 环境（依赖版本、CUDA/GPU、git hash），并把快照落盘。
  - Rationale: 后续任何训练/评估问题都能快速定位到“代码版本+环境版本”。
  - Scope: `tools/env_snapshot.py`, `docs/env_lock.txt`, `docs/system_info.md`
  - Acceptance: 在目标 conda 环境下可运行；生成的文件包含 torch/mmcv/mmdet/mmrotate 版本与 GPU/CUDA 信息。
  - Verification: `conda run -n dino_sar python tools/env_snapshot.py --out-dir docs`
  - Outputs: `docs/env_lock.txt`, `docs/system_info.md`
  - Dependencies: conda env `dino_sar`

- [x] P0002: 下载并整理 DIOR/RSAR 数据集到统一目录
  - Summary: 把 DIOR 与 RSAR 按 README 目录结构放到 `dataset/DIOR` 与 `dataset/RSAR`，并提供自动校验脚本。
  - Rationale: 训练/测试入口强依赖数据目录结构；先把数据整理好再做 smoke 与后续实验。
  - Scope: `tools/verify_dataset_layout.py`, `dataset/`
  - Acceptance: `dataset/DIOR` 与 `dataset/RSAR` 均满足 README 结构；校验脚本返回 0。
  - Verification: `conda run -n dino_sar python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR`
  - Outputs: 数据目录 + 校验输出
  - Dependencies: 数据源（Baidu/Google Drive；或 bypy: `conda run -n dino_sar python tools/download_datasets.py rsar --source bypy --bypy-remote RSAR.tar --dest dataset/RSAR`）；磁盘空间

- [x] P0003: DIOR dataloader sanity（抽样可视化+统计）
  - Summary: 随机抽样图像，读取标注并输出可视化与统计报告。
  - Rationale: 避免训练前因路径/标注格式问题浪费时间。
  - Scope: `tools/sanity_check_dior.py`, `work_dirs/sanity/dior_vis/`
  - Acceptance: 运行无报错；输出 20 张叠框图；统计报告包含图片数/标注数/空标注数/异常框数。
  - Verification: `conda run -n dino_sar python tools/sanity_check_dior.py --data-root dataset/DIOR --split train --num 20 --out-dir work_dirs/sanity/dior_vis`
  - Outputs: `work_dirs/sanity/dior_vis/`, `work_dirs/sanity/dior_sanity_report.json`
  - Dependencies: `dataset/DIOR`

- [x] P0004: DIOR smoke train（小样本闭环）
  - Summary: 用极小样本跑通 train->eval 的完整链路，产出 checkpoint 与 mAP（不追求数值）。
  - Rationale: 证明训练栈可跑、日志与评估可复现。
  - Scope: `scripts/smoke_dior.sh`, `work_dirs/exp_smoke_dior/`
  - Acceptance: 训练不报错；产生 `latest.pth`；评估输出 mAP 且非 NaN。
  - Verification: `bash scripts/smoke_dior.sh`
  - Outputs: `work_dirs/exp_smoke_dior/`（log/ckpt/eval）
  - Dependencies: `dataset/DIOR`, baseline 权重（如需要）

- [x] P0005: RSAR ann->image 对齐检查（支持任意后缀）
  - Summary: 全量遍历 annfiles，解析出 image_id，并在 images/ 中解析真实文件（jpg/png/bmp）。
  - Rationale: 先解决“找不到图片/同名多后缀冲突”类工程问题。
  - Scope: `tools/check_image_ann_alignment.py`, `sfod/semi_dota_dataset.py`
  - Acceptance: missing=0；conflict=0（或给出明确的冲突处理规则）；输出 CSV 报告。
  - Verification: `conda run -n dino_sar python tools/check_image_ann_alignment.py --ann-dir dataset/RSAR/train/annfiles --img-dir dataset/RSAR/train/images --out-csv work_dirs/sanity/rsar_alignment_train.csv`
  - Outputs: `work_dirs/sanity/rsar_alignment_train.csv`
  - Dependencies: `dataset/RSAR`

- [x] P0006: RSAR dataloader sanity（抽样可视化+统计）
  - Summary: 抽样可视化 RSAR 旋转框，输出统计报告。
  - Rationale: 确认标注解析、角度版本与可视化一致。
  - Scope: `tools/sanity_check_rsar.py`, `work_dirs/sanity/rsar_vis/`
  - Acceptance: 运行无报错；输出样例可视化；统计报告 missing=0/conflict=0。
  - Verification: `conda run -n dino_sar python tools/sanity_check_rsar.py --data-root dataset/RSAR --split train --num 20 --out-dir work_dirs/sanity/rsar_vis`
  - Outputs: `work_dirs/sanity/rsar_vis/`, `work_dirs/sanity/rsar_sanity_report.json`
  - Dependencies: `dataset/RSAR`

- [x] P0007: RSAR smoke train/test（含可视化输出）
  - Summary: 用极小样本跑通 RSAR train 与 test，并生成 `--show-dir` 可视化。
  - Rationale: 证明 RSAR 训练/测试入口可用。
  - Scope: `scripts/smoke_rsar.sh`, `work_dirs/exp_smoke_rsar/`, `work_dirs/vis_rsar/`
  - Acceptance: 训练/测试不报错；mAP 输出正常；可视化目录生成图片。
  - Verification: `bash scripts/smoke_rsar.sh`
  - Outputs: `work_dirs/exp_smoke_rsar/`, `work_dirs/vis_rsar/`
  - Dependencies: `dataset/RSAR`

- [x] P0008: SARCLIP scorer 依赖与 smoke（兼容性兜底）
  - Summary: 引入/封装 SARCLIP scorer，并提供单图打分 smoke（后续用于 CGA）。
  - Rationale: RSAR 域内 VLM 评分更合理；先把 scorer 跑通避免训练时才暴雷。
  - Scope: `tools/sarclip_smoke.py`, `sfod/cga.py`
  - Acceptance: smoke 脚本在目标环境下可运行并输出 score；缺失权重时给出明确报错与下载指引。
  - Verification: `conda run -n dino_sar python tools/sarclip_smoke.py --image <path> --prompts \"an SAR image of ship\"`
  - Outputs: `work_dirs/sanity/sarclip_smoke.log`
  - Dependencies: SARCLIP 代码与权重

- [x] P0009: CGA scorer 可插拔（CLIP ↔ SARCLIP）+ cache
  - Summary: 把当前 CGA 评分器封装成统一接口，支持按 cfg 切换 encoder，并加入磁盘 cache。
  - Rationale: DIOR 用 CLIP；RSAR 用 SARCLIP；cache 可避免训练中重复打分导致极慢。
  - Scope: `sfod/cga.py`（或新模块）, configs（新增 scorer 配置项）
  - Acceptance: 同一批输入两种 scorer 输出维度一致；cache 第二次命中率接近 100% 且速度提升明显。
  - Verification: `conda run -n dino_sar python tools/cache_benchmark.py --scorer clip --image <path> --prompt \"an aerial image of ship\"`
  - Outputs: cache 文件 + benchmark 日志
  - Dependencies: CLIP/SARCLIP 权重

- [x] P0010: RSAR-Interference “corrupt” 切换机制
  - Summary: 允许 RSAR 通过 `--cfg-options corrupt=interf_xxx` 切换到 `images-interf_xxx/`，并保证 ann->image resolve 正常。
  - Rationale: 未来加入干扰集时无需改训练代码，仅新增数据目录与配置项。
  - Scope: RSAR dataset 配置/封装（`sfod/semi_dota_dataset.py` 与 configs）
  - Acceptance: clean 与 interfered 均可被 resolve；sanity 脚本对两者均通过。
  - Verification: `conda run -n dino_sar python tools/verify_rsar_corrupt_switch.py --data-root dataset/RSAR --corrupt interf_jamA`
  - Outputs: 校验日志/报告
  - Dependencies: `dataset/RSAR` 干扰版本目录

- [x] P0011: 生成/准备 DIOR corruption 图像目录（clean/cloudy/brightness/contrast）
  - Summary: 在 `dataset/DIOR/Corruption/` 下准备 `JPEGImages-clean`（软链到 clean）以及 `JPEGImages-cloudy|brightness|contrast`（从 clean 生成或从 bypy 下载）。
  - Rationale: 根目录 `plan.md` 的实验组 B 需要 DIOR corruption 可被 `--cfg-options corrupt=...` 直接评估/训练。
  - Scope: `tools/prepare_dior_corruption.py`, `dataset/DIOR/Corruption/`
  - Acceptance: 目标目录存在；对 `ImageSets/val.txt` 与 `ImageSets/test.txt` 涉及的所有 image_id，都能在对应 `JPEGImages-<corrupt>/` 下找到同名 `.jpg`。
  - Verification: `conda run -n dino_sar python tools/prepare_dior_corruption.py --data-root dataset/DIOR --corrupt clean cloudy brightness contrast --splits val,test --workers 8`
  - Outputs: `dataset/DIOR/Corruption/JPEGImages-*/`
  - Dependencies: `dataset/DIOR/JPEGImages/`

- [x] P0012: CGA 后端可切换（CLIP ↔ SARCLIP）且失败不崩溃
  - Summary: 让 `OrientedRCNN_CGA` 的 `refine_test()` 支持通过环境变量切换使用 CLIP 或 SARCLIP；当 SARCLIP 权重缺失/依赖异常时自动降级为不启用 CGA 或切回 CLIP，保证训练/测试不中断。
  - Rationale: DIOR 默认用 CLIP；RSAR 可选 SARCLIP；并确保长跑训练时不会因权重/依赖问题中途崩溃。
  - Scope: `sfod/cga.py`, `sfod/oriented_rcnn_cga.py`, `tools/cga_smoke.py`
  - Acceptance: `CGA_SCORER=clip` 时能输出分数；`CGA_SCORER=sarclip` 且权重缺失时给出 warning 并不中断；`CGA_SCORER=sarclip` 且权重存在时正常输出分数。
  - Verification: `conda run -n dino_sar python tools/cga_smoke.py --image dataset/RSAR/train/images/0000002.png --scorer clip --classes ship,aircraft,car,tank,bridge,harbor`
  - Outputs: `work_dirs/sanity/cga_smoke.json`
  - Dependencies: `clip`；可选 `SARCLIP` + 权重（`weights/README.md`）

- [x] P0013: RSAR supervised baseline 配置与可复现训练/评估脚本
  - Summary: 增加一个 RSAR 的 supervised baseline config（OrientedRCNN）与一键训练/测试脚本，产出 mAP 与 eval json。
  - Rationale: 根目录 `plan.md` 的实验组 C 需要 RSAR supervised 基线用于对照。
  - Scope: `configs/experiments/rsar/baseline_oriented_rcnn_rsar.py`, `scripts/exp_rsar_baseline.sh`
  - Acceptance: 训练与测试无报错；`--work-dir` 下生成 `latest.pth` 与 `eval_*.json`。
  - Verification: `bash scripts/exp_rsar_baseline.sh`
  - Outputs: `work_dirs/exp_rsar_baseline/`（log/ckpt/eval json）
  - Dependencies: `dataset/RSAR`

- [x] P0014: 指标导出与对比汇总（CSV/表格）
  - Summary: 提供脚本把各实验 `eval_*.json`（或 log）汇总成 CSV，并生成一个对比表（DIOR: clean+corrupt；RSAR: baseline/UT/UT+CGA）。
  - Rationale: 根目录 `plan.md` 需要“可分析/可复现”的实验追踪与最终验收材料。
  - Scope: `tools/export_metrics.py`, `tools/ablation_table.py`, `work_dirs/results/`
  - Acceptance: 能从指定 `work_dirs/*/eval_*.json` 解析出 mAP 并生成 CSV；CSV 行包含 exp_id/method/dataset/corrupt/seed/work_dir/mAP。
  - Verification: `conda run -n dino_sar python tools/export_metrics.py --work-dirs work_dirs/exp_* --out-csv work_dirs/results/metrics.csv`
  - Outputs: `work_dirs/results/metrics.csv`
  - Dependencies: `pandas`

- [x] P0015: 随机抽样可视化对比脚本（show-dir 侧）
  - Summary: 基于 `test.py --show-dir` 的输出目录，对同名图片做随机抽样并生成并排对比图（便于快速做定性对比）。
  - Rationale: 根目录 `plan.md` 的“必备可视化（定性）”需要一个稳定的抽样对比工具。
  - Scope: `tools/vis_random_samples.py`, `work_dirs/results/vis_compare/`
  - Acceptance: 支持传入多个 `--vis-dirs`；自动取同名文件交集并采样；输出对比图到 `--out-dir`；无交集时给出明确报错并返回非 0。
  - Verification: `conda run -n dino_sar python tools/vis_random_samples.py --vis-dirs work_dirs/exp_dior_baseline_eval/vis_clean work_dirs/exp_dior_ut/vis_clean work_dirs/exp_dior_ut_cga_clip/vis_clean --num 8 --out-dir work_dirs/results/vis_compare/dior_clean`
  - Outputs: `work_dirs/results/vis_compare/dior_clean/`
  - Dependencies: `Pillow`

- [x] P0016: 一键生成图表（mAP 条形图 + 训练曲线）
  - Summary: 从 `work_dirs/results/metrics.csv` 与各实验的 `*.log.json` 解析指标/曲线，一键输出常用图表（mAP 对比、loss/pseudo 统计曲线等）。
  - Rationale: 根目录 `plan.md` 的“必备可视化（定量）”需要一个可复用的绘图入口。
  - Scope: `tools/plot_all.py`, `work_dirs/results/plots/`
  - Acceptance: 在仅提供 `metrics.csv` 时也能生成 mAP 图；提供 `--log-json-glob` 时额外生成训练曲线；输出 PNG 到 `--out-dir`。
  - Verification: `conda run -n dino_sar python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob 'work_dirs/exp_*/*.log.json' --out-dir work_dirs/results/plots`
  - Outputs: `work_dirs/results/plots/`
  - Dependencies: `matplotlib`, `pandas`

- [x] P0017: 实验追踪表与复现材料（experiments.csv / README / MODEL_ZOO）
  - Summary: 生成 `experiments.csv` 作为 run-level 追踪表，并补齐复现说明与模型清单文档。
  - Rationale: 根目录 `plan.md` 的“实验追踪与复现材料”需要统一落盘与可重建的生成脚本。
  - Scope: `tools/export_experiments.py`, `experiments.csv`, `README_experiments.md`, `MODEL_ZOO.md`
  - Acceptance: `tools/export_experiments.py` 可从 `metrics.csv` 生成 `experiments.csv`（包含 git hash、log/config 指针等）；`README_experiments.md` 与 `MODEL_ZOO.md` 可直接指引复现与查找 ckpt。
  - Verification: `bash -lc 'conda run -n dino_sar python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv && test -f README_experiments.md && test -f MODEL_ZOO.md'`
  - Outputs: `experiments.csv`, `README_experiments.md`, `MODEL_ZOO.md`
  - Dependencies: `pandas`

- [x] P0018: RSAR UnbiasedTeacher teacher 初始化（从 RSAR supervised baseline）
  - Summary: 让 RSAR 的 UT/UT+CGA 支持从 `exp_rsar_baseline` checkpoint 初始化 teacher（必要时也初始化 student），避免 `mAP=0` 的退化训练。
  - Rationale: 现有 RSAR UT 配置 `weight_l=0` 时若 teacher 不初始化，几乎不会产生 pseudo-label，导致检测性能为 0。
  - Scope: `scripts/exp_rsar_ut.sh`, `docs/experiment.md`
  - Acceptance: 使用 `TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth` 运行 UT，评估 mAP 非 0。
  - Verification: `bash -lc 'CGA_SCORER=none TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_nocga_tinit VIS_DIR=work_dirs/vis_rsar_ut_nocga_tinit bash scripts/exp_rsar_ut.sh'`
  - Outputs: `work_dirs/exp_rsar_ut_nocga_tinit/`（log/ckpt/eval json）
  - Dependencies: `work_dirs/exp_rsar_baseline/latest.pth`, `dataset/RSAR`

- [x] P0019: RSAR UT+CGA(CLIP) teacher 初始化（从 RSAR supervised baseline）
  - Summary: 基于 P0018 的 teacher 初始化机制，在 RSAR 上跑通 UT+CGA(CLIP) 并产出可对比的 mAP。
  - Rationale: 确保 RSAR 上的 CGA 实验不是“随机 teacher”导致的无效对比。
  - Scope: `scripts/exp_rsar_ut.sh`, `docs/experiment.md`
  - Acceptance: UT+CGA(CLIP) smoke/full 均可运行并产出 `eval_*.json`（mAP 非 0）。
  - Verification: `bash -lc 'CGA_SCORER=clip TEACHER_CKPT=work_dirs/exp_rsar_baseline/latest.pth SMOKE=1 MAX_EPOCHS=1 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit bash scripts/exp_rsar_ut.sh'`
  - Outputs: `work_dirs/exp_rsar_ut_cga_clip_tinit/`（log/ckpt/eval json）
  - Dependencies: `work_dirs/exp_rsar_baseline/latest.pth`, `dataset/RSAR`, `clip`

- [x] P0020: 指标/图表汇总纳入 RSAR teacher-init 实验
  - Summary: 把新增 RSAR teacher-init 实验纳入 `metrics.csv`/表格/图表与 `experiments.csv` 追踪表。
  - Rationale: 确保对比结论可复现且可视化材料齐全。
  - Scope: `tools/export_metrics.py`, `tools/ablation_table.py`, `tools/plot_all.py`, `tools/export_experiments.py`
  - Acceptance: `metrics.csv` 包含 teacher-init 的 RSAR 行；图表刷新；`experiments.csv` 行数增加。
  - Verification: `bash -lc 'conda run -n dino_sar python tools/export_metrics.py --work-dirs work_dirs/exp_* --out-csv work_dirs/results/metrics.csv && conda run -n dino_sar python tools/ablation_table.py --csv work_dirs/results/metrics.csv --out-md work_dirs/results/ablation_table.md && conda run -n dino_sar python tools/plot_all.py --metrics-csv work_dirs/results/metrics.csv --log-json-glob \"work_dirs/exp_*/*.log.json\" --out-dir work_dirs/results/plots && conda run -n dino_sar python tools/export_experiments.py --metrics-csv work_dirs/results/metrics.csv --out-csv experiments.csv'`
  - Outputs: `work_dirs/results/metrics.csv`, `work_dirs/results/ablation_table.md`, `work_dirs/results/plots/`, `experiments.csv`
  - Dependencies: `pandas`, `matplotlib`

- [x] P0021: RSAR 多后缀解析实现审计记录（notes）
  - Summary: 补齐根目录 `plan.md` 里要求的 `notes/dataloader_extension_audit.md`，记录 RSAR 图片后缀解析的代码位置、实现策略与验证方法。
  - Rationale: 便于后续引入 RSAR-Interference 或新增后缀/目录结构时快速定位与复查。
  - Scope: `notes/dataloader_extension_audit.md`, `sfod/semi_dota_dataset.py`, `tools/check_image_ann_alignment.py`
  - Acceptance: note 文件存在且清晰说明“ann->image resolve”的索引逻辑/优先级/冲突处理；并引用可运行的对齐检查命令。
  - Verification: `bash -lc 'test -f notes/dataloader_extension_audit.md && rg -n \"DOTADatasetAnySuffix\" sfod/semi_dota_dataset.py >/dev/null'`
  - Outputs: `notes/dataloader_extension_audit.md`
  - Dependencies: N/A

- [x] P0022: SARCLIP 在 torch 1.7.1 下 batch_first 兼容补丁
  - Summary: 给 `third_party/SARCLIP` 的 `nn.MultiheadAttention(batch_first=...)` 加兼容层，使其在 torch<1.9（无 batch_first 参数）时也能运行。
  - Rationale: 根目录 `plan.md` 的关键风险 A；不修复会导致 SARCLIP 在老训练栈直接崩溃。
  - Scope: `third_party/SARCLIP/sar_clip/transformer.py`, `scripts/sarclip_torch17_smoke.sh`
  - Acceptance: 在 torch 1.7.1 环境下可 import 并执行 SARCLIP smoke（至少跑通一次 encode_image/encode_text 或最小前向），不再报 `unexpected keyword argument 'batch_first'`。
  - Verification: `bash scripts/sarclip_torch17_smoke.sh`
  - Outputs: `work_dirs/sanity/sarclip_smoke_torch171.log`
  - Dependencies: conda（可创建 env）

- [x] P0023: RSAR UT+CGA teacher-init 统一 N_TEST=1000 评估并刷新汇总
  - Summary: 增加 RSAR UT 脚本的 eval-only 模式，并将 teacher-init 的 UT+CGA(CLIP) 在 `N_TEST=1000` 上重新评估，刷新 `metrics.csv`/表格/追踪表与 docs 证据链接。
  - Rationale: 避免 RSAR 对比里混用不同大小的 test 子集导致误读。
  - Scope: `scripts/exp_rsar_ut.sh`, `docs/experiment.md`, `docs/plan.md`, `work_dirs/results/metrics.csv`, `experiments.csv`
  - Acceptance: `work_dirs/exp_rsar_ut_cga_clip_tinit/` 产生新的 `eval_*.json`（N_TEST=1000）；`metrics.csv` 的 RSAR ut+cga 行更新为最新 eval。
  - Verification: `bash -lc 'CGA_SCORER=clip DO_TRAIN=0 DO_TEST=1 SMOKE=1 N_TRAIN=50 N_VAL=50 N_TEST=1000 SAMPLES_PER_GPU=4 WORKERS_PER_GPU=4 WORK_DIR=work_dirs/exp_rsar_ut_cga_clip_tinit VIS_DIR=work_dirs/vis_rsar_ut_cga_clip_tinit_eval1000 SPLIT_DIR=work_dirs/smoke_splits/rsar_ut_cga_clip_tinit_eval1000 CKPT=work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth bash scripts/exp_rsar_ut.sh && conda run -n dino_sar python tools/export_metrics.py --work-dirs work_dirs/exp_* --out-csv work_dirs/results/metrics.csv'`
  - Outputs: `work_dirs/exp_rsar_ut_cga_clip_tinit/eval_*.json`, `work_dirs/results/metrics.csv`
  - Dependencies: `work_dirs/exp_rsar_ut_cga_clip_tinit/latest.pth`, `dataset/RSAR`

- [x] P0024: 生成 RSAR-Interference(interf_jamA) 扰动图像目录（真实数据）
  - Summary: 将 `dataset/RSAR/*/images-interf_jamA/` 从占位软链替换为真实目录，并用可复现的干扰模拟脚本从 clean 批量生成对应图像。
  - Rationale: 之前 `interf_jamA` 使用软链占位会导致“干扰评估=clean”，无法验证鲁棒性结论；必须落盘真实扰动图像。
  - Scope: `tools/prepare_rsar_interference.py`, `tools/verify_rsar_interference_diff.py`, `scripts/prepare_rsar_interf_jamA.sh`, `dataset/RSAR/*/images-interf_jamA/`
  - Acceptance: `tools/verify_rsar_corrupt_switch.py` 对 clean/interf_jamA 均 `missing=0 conflict=0`；`tools/verify_rsar_interference_diff.py` 抽样对比确认干扰图与 clean 不全相同。
  - Verification: `bash scripts/prepare_rsar_interf_jamA.sh`
  - Outputs: `dataset/RSAR/{train,val,test}/images-interf_jamA/`（真实扰动数据）
  - Dependencies: `opencv-python`, `numpy`, `scipy`（可选；当前实现未强依赖）
  - Evidence: `dataset/RSAR/{train,val,test}/images-interf_jamA/`, `work_dirs/sanity/rsar_corrupt_switch/*_corrupt-interf_jamA.csv`

- [x] P0025: 生成 RSAR-Interference(interf_jamB) 扰动图像目录（真实数据）
  - Summary: 生成 `dataset/RSAR/*/images-interf_jamB/`（AM 条带干扰 `noise_am_jamming`），同名映射 clean 批量生成。
  - Rationale: 对齐 `plan.md` 的 jamA/jamB 目录规范，便于后续多干扰评估/混训扩展。
  - Scope: `tools/prepare_rsar_interference.py`, `tools/verify_rsar_interference_diff.py`, `scripts/prepare_rsar_interf_jamB.sh`, `dataset/RSAR/*/images-interf_jamB/`
  - Acceptance: `tools/verify_rsar_corrupt_switch.py --corrupt interf_jamB` 返回 `missing=0 conflict=0`；`tools/verify_rsar_interference_diff.py` 抽样对比 `identical < checked`。
  - Verification: `bash scripts/prepare_rsar_interf_jamB.sh`
  - Outputs: `dataset/RSAR/{train,val,test}/images-interf_jamB/`（真实扰动数据）
  - Dependencies: `opencv-python`, `numpy`
  - Evidence: `dataset/RSAR/{train,val,test}/images-interf_jamB/`, `work_dirs/sanity/rsar_corrupt_switch/*_corrupt-interf_jamB.csv`

- [x] P0026: 生成 RSAR-Interference severity 套件（test-only，interf_jamA_s1..s5 / interf_jamB_s1..s5）
  - Summary: 为鲁棒性曲线准备轻量级扰动版本：只生成 `test` split 的 `images-interf_jamA_s{1..5}` 与 `images-interf_jamB_s{1..5}`。
  - Rationale: 先用 test-only 版本快速做 mAP vs severity 曲线，筛选最有代表性的干扰强度/类型，再决定是否扩展到 train/val（避免磁盘爆炸）。
  - Scope: `scripts/prepare_rsar_interf_severity_test.sh`, `tools/verify_rsar_corrupt_switch.py`, `dataset/RSAR/test/images-interf_jamA_s*/`, `dataset/RSAR/test/images-interf_jamB_s*/`
  - Acceptance: 每个 `corrupt` 在 `--splits test` 下 `missing=0 conflict=0`；`tools/verify_rsar_interference_diff.py` 抽样对比不全相同（`identical < checked`）。
  - Verification: `bash scripts/prepare_rsar_interf_severity_test.sh`
  - Outputs: `dataset/RSAR/test/images-interf_jamA_s{1..5}/`, `dataset/RSAR/test/images-interf_jamB_s{1..5}/`
  - Dependencies: 磁盘空间（噪声型扰动 PNG 体积会显著增大）
  - Evidence: `dataset/RSAR/test/images-interf_jamA_s*/`, `dataset/RSAR/test/images-interf_jamB_s*/`, `work_dirs/sanity/rsar_corrupt_switch/test_corrupt-interf_jamA_s*.csv`, `work_dirs/sanity/rsar_corrupt_switch/test_corrupt-interf_jamB_s*.csv`

- [x] P0027: 扩展代表性 severity（interf_jamB_s3）到 train/val（用于混训/鲁棒训练）
  - Summary: 基于 `E0027` 的 severity 曲线（jamB 随 severity 单调下降明显），选择中等强度 `interf_jamB_s3`，生成 `train/val` 的 `images-interf_jamB_s3/`。
  - Rationale: 只扩展一个代表性等级，控制磁盘开销，同时为后续“混训/鲁棒训练 + 再跑 severity 曲线”提供数据基础。
  - Scope: `scripts/prepare_rsar_interf_jamB_s3_trainval.sh`, `tools/prepare_rsar_interference.py`, `tools/verify_rsar_corrupt_switch.py`, `dataset/RSAR/{train,val}/images-interf_jamB_s3/`
  - Acceptance: 在 `--splits train,val` 下 `missing=0 conflict=0`；抽样 diff check 通过（非全相同）。
  - Verification: `bash scripts/prepare_rsar_interf_jamB_s3_trainval.sh`
  - Outputs: `dataset/RSAR/train/images-interf_jamB_s3/`, `dataset/RSAR/val/images-interf_jamB_s3/`
  - Dependencies: 磁盘空间（约 +数 GB）；CPU/IO（一次性生成 78k+ 图）
  - Evidence: `dataset/RSAR/train/images-interf_jamB_s3/`（78837 files, 7.8G）, `dataset/RSAR/val/images-interf_jamB_s3/`（8467 files, 1.1G）, `work_dirs/sanity/rsar_corrupt_switch/train_corrupt-interf_jamB_s3.csv`, `work_dirs/sanity/rsar_corrupt_switch/val_corrupt-interf_jamB_s3.csv`

- [x] P0028: RSAR jamB_s3 鲁棒训练实验矩阵（baseline / UT / UT+CGA；interf-only & mix）
  - Summary: 在 `interf_jamB_s3` 上做两种训练数据策略：A) 仅用 interfered（interf-only）；B) clean+interf 混训（mix）。模型覆盖 baseline / UT / UT+CGA(SARCLIP)。
  - Rationale: `E0027` 显示 jamB severity 对性能影响显著；用代表性 s3 做训练策略对比，验证“混训/课程/打分”是否提升鲁棒性曲线。
  - Scope: `scripts/exp_rsar_baseline.sh`, `scripts/eval_rsar_severity_curve.sh`, `scripts/eval_rsar_severity_curve_baseline.sh`, `sfod/utils/patches.py`, `docs/experiment.md`
  - Acceptance: 每个实验 smoke 跑通（训练+测试产出 `eval_*.json` 且非 NaN）；full 跑通并记录 mAP；对每个 ckpt 至少产出一份 jamB severity 曲线 CSV（clean + s1..s5）。
  - Verification: 见 `docs/experiment.md` 中 E0028+（smoke/full cmd）。
  - Outputs: `work_dirs/exp_rsar_*_interf_jamB_s3*/` + `work_dirs/exp_rsar_severity/<tag>/interf_jamB/severity_summary.csv`
  - Dependencies: GPU0（避免被 `vram_fill` 占用）；磁盘（训练 ckpt + show-dir）
  - Evidence: `docs/experiment.md` E0028–E0033；eval json: `work_dirs/exp_rsar_baseline_interf_jamB_s3/eval_20260125_132251.json`, `work_dirs/exp_rsar_baseline_mix_interf_jamB_s3/eval_20260125_125952.json`, `work_dirs/exp_rsar_ut_nocga_interf_jamB_s3/eval_20260125_134639.json`, `work_dirs/exp_rsar_ut_nocga_mix_interf_jamB_s3/eval_20260125_141022.json`, `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/eval_20260125_143501.json`, `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/eval_20260125_145932.json`；severity csv: `work_dirs/exp_rsar_severity/exp_rsar_baseline_interf_jamB_s3/interf_jamB/severity_summary.csv`, `work_dirs/exp_rsar_severity/exp_rsar_baseline_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`, `work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_interf_jamB_s3/interf_jamB/severity_summary.csv`, `work_dirs/exp_rsar_severity/exp_rsar_ut_nocga_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`, `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_interf_jamB_s3/interf_jamB/severity_summary.csv`, `work_dirs/exp_rsar_severity/exp_rsar_ut_cga_sarclip_tinit_t2_mix_interf_jamB_s3/interf_jamB/severity_summary.csv`

## Conclusions
- [x] C0001: DIOR 与 RSAR 均能跑通 smoke 训练/测试闭环（含可视化与 mAP）
  - Evidence required: `work_dirs/exp_smoke_dior/` 与 `work_dirs/exp_smoke_rsar/` 中日志包含 mAP 且非 NaN；`work_dirs/vis_rsar/` 有输出图。
  - Experiments: E0001,E0002
  - Artifacts: `docs/experiment.md` 记录的 Logs/Artifacts
  - Evidence: `work_dirs/exp_smoke_dior/20260121_000801.log`, `work_dirs/exp_smoke_rsar/20260121_003231.log`, `work_dirs/vis_rsar/`

- [x] C0002: RSAR 的 ann->image 解析对 jpg/png/bmp 扩展名无关且对齐检查 missing=0/conflict=0
  - Evidence required: `work_dirs/sanity/rsar_alignment_train.csv` 中 status 全为 ok；统计 missing=0/conflict=0。
  - Experiments: E0003
  - Artifacts: `work_dirs/sanity/rsar_alignment_train.csv`
  - Evidence: `work_dirs/sanity/rsar_alignment_train.csv`

- [x] C0003: scorer=CLIP/SARCLIP 可通过配置切换，且 cache 显著减少重复打分
  - Evidence required: benchmark 日志记录 cache hit；两种 scorer smoke 均可运行并输出分数。
  - Experiments: E0004
  - Artifacts: `work_dirs/sanity/` 下的 smoke/benchmark 日志与 cache 文件
  - Evidence: `work_dirs/sanity/cache_benchmark.json`, `work_dirs/sanity/cache_benchmark_sarclip.json`, `work_dirs/sanity/sarclip_smoke.log`

- [x] C0004: RSAR-Interference 可仅通过新增目录与 `corrupt` 配置项无痛接入
  - Evidence required: `tools/verify_rsar_corrupt_switch.py` 对 clean/interf 均通过；smoke train/test 可运行。
  - Experiments: E0005
  - Artifacts: 校验报告 + 对应 smoke 运行日志
  - Evidence: `work_dirs/sanity/rsar_corrupt_switch/`, `work_dirs/exp_smoke_rsar_interf_jamA/`, `work_dirs/vis_rsar_interf_jamA/`

- [x] C0005: DIOR 支持 clean/cloudy/brightness/contrast 的 corruption 评估与训练入口
  - Evidence required: `dataset/DIOR/Corruption/JPEGImages-*/` 存在且文件覆盖 val/test；对每个 corrupt 执行一次 `test.py --eval mAP --work-dir ...` 成功并生成 `eval_*.json`。
  - Experiments: E0006,E0007
  - Artifacts: `dataset/DIOR/Corruption/JPEGImages-*/`, `work_dirs/exp_dior_*/*eval_*.json`
  - Evidence: `dataset/DIOR/Corruption/JPEGImages-cloudy/`, `dataset/DIOR/Corruption/JPEGImages-brightness/`, `dataset/DIOR/Corruption/JPEGImages-contrast/`, `work_dirs/exp_dior_baseline_eval/eval_clean/eval_20260121_041854.json`, `work_dirs/exp_dior_baseline_eval/eval_cloudy/eval_20260121_042417.json`, `work_dirs/exp_dior_baseline_eval/eval_brightness/eval_20260121_042944.json`, `work_dirs/exp_dior_baseline_eval/eval_contrast/eval_20260121_043406.json`, `work_dirs/exp_dior_ut/latest.pth`, `work_dirs/exp_dior_ut_cga_clip/latest.pth`

- [x] C0006: DIOR（baseline/UT/UT+CGA(CLIP)）在 clean 与多 corruption 上的 mAP 对比可复现并汇总成表
  - Evidence required: `work_dirs/results/metrics.csv` 包含 DIOR 相关行（method+corrupt+mAP）；对应 work_dir 下存在 `eval_*.json`。
  - Experiments: E0006,E0007,E0008
  - Artifacts: `work_dirs/results/metrics.csv`
  - Evidence: `work_dirs/results/metrics.csv`, `work_dirs/results/ablation_table.md`

- [x] C0007: RSAR（baseline/UT/UT+CGA）在 clean 上的 mAP 对比可复现并汇总成表
  - Evidence required: `work_dirs/results/metrics.csv` 包含 RSAR 相关行；对应 work_dir 下存在 `eval_*.json`。
  - Experiments: E0009,E0010,E0011,E0015,E0016,E0017
  - Artifacts: `work_dirs/results/metrics.csv`
  - Evidence: `work_dirs/results/metrics.csv`, `work_dirs/exp_rsar_baseline/eval_20260121_041717.json`, `work_dirs/exp_rsar_ut_nocga_tinit/eval_20260121_065058.json`, `work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_074641.json`

- [x] C0008: SARCLIP 在 torch 1.7.1 下不再因 batch_first 崩溃
  - Evidence required: torch 1.7.1 环境下执行 smoke 脚本成功，日志包含 `torch=1.7.1` 与 `OK`。
  - Experiments: E0018
  - Artifacts: `work_dirs/sanity/sarclip_smoke_torch171.log`
  - Evidence: `work_dirs/sanity/sarclip_smoke_torch171.log`

- [x] C0009: RSAR 上 UT+CGA(SARCLIP) 可跑通，且 prompt 模板对 mAP 有影响
  - Evidence required: UT+CGA(SARCLIP) 的 smoke/full 产出 `eval_*.json`；模板2与模板1可对比；no-cache 重跑 eval 与原 mAP 一致并记录耗时。
  - Experiments: E0016,E0019,E0020,E0021
  - Artifacts: `docs/experiment.md`（E0019–E0021），各 work_dir 的 `eval_*.json`
  - Evidence: `work_dirs/exp_rsar_ut_cga_clip_tinit/eval_20260121_071054.json`（CLIP, mAP=0.3094）, `work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_193952.json`（SARCLIP tmpl1, mAP=0.3088）, `work_dirs/exp_rsar_ut_cga_sarclip_tinit_t2/eval_20260122_194541.json`（SARCLIP tmpl2, mAP=0.3119）, `work_dirs/exp_rsar_ut_cga_sarclip_tinit/eval_20260122_194635.json`（no-cache eval, mAP=0.3088）, `.rd_queue/logs/J20260122-113351-bff0__e0021-full.log`（no-cache eval real=62.63s）
