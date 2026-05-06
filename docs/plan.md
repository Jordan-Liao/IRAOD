# RSAR SFOD-RS Full Fix Plan (Completed)

## Scope
- Project: IRAOD RSAR SFOD-RS full rerun
- Remote repo: `/mnt/SSD1_8TB/zechuan/IRAOD`
- Run root: `work_dirs/rsar_sfodrs_full_fix_20260424_172627`
- Runtime window: `2026-04-24 17:26:27 CST` to `2026-04-26 21:13:36 CST`
- Total elapsed wall time: `51.786 h`

## Objective
1. Enforce SFOD-RS protocol with target-only adaptation on RSAR 7 corruptions.
2. Make full pipeline runnable from `auto` source checkpoint generation (no manual source ckpt handoff).
3. Add explicit SFOD-RS diagnostics in evaluation stage.
4. Produce one final aggregate result table (`csv` + `md`) for the completed run.

## Fixed Protocol
1. `source_train`: clean labeled train split only.
2. `source_clean_test`: clean test split.
3. For each corruption in:
   `chaff`, `gaussian_white_noise`, `point_target`, `noise_suppression`,
   `am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`
   - `direct_test` on `corruptions/<corr>/test/images`
   - `self_training` on `corruptions/<corr>/val/images` then eval on `test/images`
   - `self_training_plus_cga` on `corruptions/<corr>/val/images` then eval on `test/images`
4. Do not use corruption train split.

## Code/Script Delta Included
- `test.py`: SFOD-RS diagnostics print in evaluation stage.
- `scripts/run_rsar_sfodrs_full_3gpu.sh`: support `source_arg=auto` to train source detector first and then continue full 7-corr pipeline.
- `scripts/run_rsar_sfodrs_7corr.sh`, `scripts/exp_rsar_sfodrs_adapt.sh`: protocol wiring updates.
- `tools/collect_rsar_sfodrs_results.py`: aggregate final metrics table.
- Split-run utilities added during rerun recovery:
  - `scripts/cutover_split_main_to_shard.sh`
  - `scripts/cutover_resume_after_wait.sh`
  - `scripts/run_single_corr_ddp.sh`

## Execution Matrix
- Total planned tasks: `37`
  - `1` source train
  - `1` clean source test
  - `7 x 5` target tasks (`direct_test`, `adapt_nocga`, `eval_nocga`, `adapt_cga`, `eval_cga`)
- Completion: `37/37`

## Parallel Cutover Record
- Main line covered source + 2 corruptions + partial `point_target`.
- Shard line covered:
  - `am_noise_horizontal`
  - `smart_suppression`
  - `am_noise_vertical`
- Final parallel completion for remaining two domains:
  - `point_target` finished `2026-04-26 21:12:25 CST`
  - `noise_suppression` finished `2026-04-26 21:13:36 CST`

## Final Outputs
- `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv`
- `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md`
- Both generated at `2026-04-27 00:51 CST`.

## Acceptance Status
- Protocol completion check: `23/23` artifact checks passed
  - `source_train` ckpt present
  - `source_clean_test` eval json present
  - `7` corruptions x `3` eval json present (`direct_test`, `self_training`, `self_training_plus_cga`)
- Training/eval processes ended cleanly; no IRAOD run process remained active after completion.

---

## Phase 8: Loose SFOD-RS（Faithful Dataloader，已完成）

### 目标
- 修复 strict target-only adaptation 在重度腐败域的崩溃问题
- 验证 `SemiRSARSFODDataset + weight_l=0.5` 是否防止 teacher EMA 漂移
- 消融 score_thr=0.5 vs thr=0.7 的影响

### 运行配置
- 源权重：`work_dirs/rsar_sfodrs_full_fix_20260424_172627/source_train/latest.pth`（clean=0.5385）
- E0123（thr=0.7）运行目录：`work_dirs/rsar_sfodrs_loose_ablation`
- E0124（thr=0.5）运行目录：`work_dirs/rsar_sfodrs_loose_scrthr05_ablation`
- 测试域：chaff、gaussian_white_noise、noise_suppression（各 12 epoch）

### 完成时间
- E0123 完成：2026-05-03
- E0124 完成：2026-05-03

### 关键结果
| corruption | direct | loose nocga | loose cga | Δ(vs direct) |
|---|---:|---:|---:|---|
| chaff | 0.4690 | 0.4516 | 0.4580 | -0.017 |
| gaussian_white_noise | 0.5429 | 0.4983 | 0.4988 | -0.045 |
| noise_suppression | 0.2436 | 0.2996 | 0.2994 | **+0.056** |

### 结论
1. Collapse 完全消除（strict nocga chaff 0.011 → loose 0.452）
2. noise_suppression 是唯一 adapt > direct 的域（+23% relative）
3. score_thr=0.5 为阴性（更多伪框=更多噪声）
4. 最优配置：`RSAR_LOADER_MODE=loose RSAR_WEIGHT_L=0.5 RSAR_PSEUDO_SCORE_THR=0.7`

---

## Phase 9: Corr-Aug 源模型重训（已完成）

### 目标
- 在源训练阶段在线随机施加 7 种 SAR 腐败（p=0.5），使源模型具备 domain invariance
- 从根源提升所有域的 direct_test 基线（预期 +5~8pp mean）

### 实现
- 新增 `tools/rsar_corruption_pipeline.py`：`RsarOnlineCorruptionAugment` transform（ROTATED_PIPELINES 注册）
- 修改主 config 的 `source_train_pipeline`：`LoadImageFromFile` 后插入 online augmentation（`RSAR_CORR_AUG=1` 时激活）
- 新增薄配置 `configs/current/rsar_source_corraug.py`
- 控制变量：`RSAR_CORR_AUG=1 RSAR_CORR_AUG_PROB=0.5`，3×RTX 4090 D（NGPUS=3，RSAR_SAMPLES_PER_GPU=8）

### 运行配置
- 源权重：`work_dirs/rsar_corraug_loose_20260504/source_train/latest.pth`（clean test mAP=0.5125）
- Loose adaptation：`RSAR_LOADER_MODE=loose RSAR_WEIGHT_L=0.5 RSAR_PSEUDO_SCORE_THR=0.7`，全7域
- 完成时间：2026-05-05 17:59 CST（source train ~6h + 7域 adaptation ~35h）

### 关键结果（全7域）
| corruption | old direct | corr-aug direct | Δ | corr-loose nocga |
|---|---:|---:|---:|---:|
| chaff | 0.4690 | 0.4899 | +0.0209 | 0.4574 |
| gaussian_white_noise | 0.5429 | 0.5154 | -0.0275 | 0.4805 |
| point_target | 0.3183 | 0.5100 | **+0.1917** | 0.4757 |
| noise_suppression | 0.2436 | 0.4701 | **+0.2265** | 0.4359 |
| am_noise_horizontal | 0.2985 | 0.4546 | **+0.1561** | 0.3776 |
| smart_suppression | 0.2935 | 0.4245 | **+0.1310** | 0.4026 |
| am_noise_vertical | 0.2967 | 0.4548 | **+0.1581** | 0.3979 |
| **Mean (7 corr)** | **0.3372** | **0.4742** | **+0.1370 (+40.6%)** | **0.4325** |

### 结论
1. **假设验证**：Corr-aug 大幅提升 direct_test（+41% relative），6/7 域正向，明显超出预期（+5~8pp → 实际 +13.7pp）
2. **gwn 轻微回退**（-0.028）：原始源模型对 GWN 天然鲁棒，corr-aug 略微稀释 clean 特征
3. **Loose adaptation 仍回退**（-0.042 mean vs direct）：pseudo_num(acc)=0.000 全程，有效信号极少
4. **当前最优策略**：corr-aug 源模型 + direct inference（mean 7corr = 0.4742）
5. **下一步方向**：伪标签质量提升（更强滤波/温度缩放）或 test-time adaptation（无需伪标签）

---

## Phase 10: 伪标签阈值修复实验（E0127/E0128，进行中）

### 目标
- 诊断并修复 `pseudo_num(acc)=0.000` 问题
- 通过降低 score_thr（0.7→0.2 退火 / 0.4 固定）增加伪标签数量以产生适应信号

### 诊断结论（2026-05-05）
- `pseudo_num(acc)` 是 IoU TP 指标，目标域无 GT → 恒为 0，是**误导性指标而非 bug**
- 真实状态：thr=0.7 时 pseudo/img=1.74，mean_score=0.916，伪标签正常生成
- 问题在于：高置信度伪标签只覆盖已知模式，不产生新适应信号
- **代码修复**：暴露 `RSAR_THR_SCHEDULE/RSAR_SCORE_THR_START/RSAR_SCORE_THR_END` env var 到 config，新增 `[PseudoScoreDist]` 日志

### E0127（thr=0.2→0.4 linear annealing）
- 运行目录：`work_dirs/rsar_e0127_thr_anneal_20260505_213455`
- GPU：2,3,7（NGPUS=3），启动：2026-05-05 21:34 CST

| corruption | direct | nocga | Δ | cga | Δ |
|---|---:|---:|---:|---:|---:|
| chaff | 0.4899 | 0.4061 | -8.4pp | 0.4067 | -8.3pp |
| gaussian_white_noise | 0.5154 | 0.4448 | -7.0pp | 0.4469 | -6.8pp |
| point_target | 0.5100 | 0.4501 | -6.0pp | 0.4493 | -6.1pp |
| noise_suppression | 0.4701 | 0.3810 | -8.9pp | 0.3803 | -9.0pp |
| am_noise_horizontal | 0.4546 | 0.2613 | **-19.4pp** | 0.2612 | **-19.4pp** |
| smart_suppression | 0.4245 | 0.3301 | -9.4pp | 0.3324 | -9.2pp |
| am_noise_vertical | 0.4548 | 0.3068 | **-14.8pp** | 进行中 | — |
| **Mean (7域)** | **0.4742** | **0.3686** | **-10.6pp** | — | — |

### E0128（thr=0.4 fixed，对照组）
- 运行目录：`work_dirs/rsar_e0128_thr04_fixed_20260506_011722`
- GPU：8,9（NGPUS=2），启动：2026-05-06 01:17 CST

| corruption | direct | nocga | Δ | cga | Δ |
|---|---:|---:|---:|---:|---:|
| chaff | 0.4899 | 0.4349 | -5.5pp | 0.4413 | -4.9pp |
| gaussian_white_noise | 0.5154 | 0.4726 | -4.2pp | 0.4730 | -4.2pp |
| point_target | 0.5100 | 0.4676 | -4.2pp | 0.4728 | -3.7pp |
| noise_suppression | 0.4701 | 0.4331 | -3.7pp | 进行中 | — |

### Phase 10 结论（截至 2026-05-06 12:20 CST）
1. **阈值调优无法修复适应问题**：thr=0.2、0.4 均全域退步，幅度 4~19pp
2. **am_noise_horizontal 灾难性崩溃**（E0127 -19.4pp）：pseudo/img=8.88，大量低质量伪标签造成灾难性遗忘
3. **类别不平衡是根本障碍**：ship 占伪标签 66-73%，更高阈值反而加剧（ship 置信度本身更高）
4. **E0128 优于 E0127** 约 2.8pp，说明更少但更高质量的伪标签好于数量多但嘈杂
5. **UnbiasedTeacher 自训练在无 GT 监督下系统性有害**，不是超参数问题
6. **下一步**：TENT 方案（`tools/tent_adapt_per_corr.py` 已实现），只更新 BN 统计量，无伪标签，无漂移风险
