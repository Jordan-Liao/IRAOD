# RSAR Semi-Supervised Experiments (SFOD-RS, 2026-04/05)

## Scope
- Dataset: RSAR 6 classes (`ship`, `aircraft`, `car`, `tank`, `bridge`, `harbor`)
- Corruptions: 7 domains (`chaff`, `gaussian_white_noise`, `point_target`, `noise_suppression`, `am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`)
- Protocol:
  - source model trained on clean labeled source split
  - adaptation uses target-domain `val/images` only
  - evaluation uses target-domain `test/images`
  - no corruption train split used

---

## Experiment A: Strict SFOD-RS（Target-Only，Phase 5-7）

**配置**：`RSAR_LOADER_MODE=strict`，`weight_l=0.0`，`use_labeled=False`  
源权重：`work_dirs/rsar_sfodrs_full_fix_20260424_172627/source_train/latest.pth`（clean=0.5385）

### Core Metrics (mAP) — 7 corruption 全集
| Setting | Mean (clean + 7 domains) | Mean (7 corrupted domains only) |
| --- | ---: | ---: |
| direct_test | 0.3624 | 0.3372 |
| self_training strict (no CGA) | 0.1025 | 0.0402 |
| self_training_plus_cga strict | 0.1402 | 0.0833 |

### Per-Domain Breakdown (strict self_training vs +CGA)
| corruption | direct | nocga | cga | Δ(cga-nocga) |
|---|---:|---:|---:|---:|
| chaff | 0.4690 | 0.0106 | 0.0833 | +0.0727 |
| gaussian_white_noise | 0.5429 | 0.0080 | 0.0816 | +0.0736 |
| point_target | 0.3183 | 0.0448 | 0.1018 | +0.0570 |
| noise_suppression | 0.2436 | 0.0861 | 0.0952 | +0.0091 |
| am_noise_horizontal | 0.2985 | 0.0292 | 0.0587 | +0.0295 |
| smart_suppression | 0.2935 | 0.0538 | 0.0720 | +0.0182 |
| am_noise_vertical | 0.2967 | 0.0301 | 0.0714 | +0.0413 |

### Observations
1. Strict target-only 在重度腐败域（chaff/gwn）出现完全崩溃（mAP<0.01）
2. CGA 显著缓解崩溃（+0.043 mean），但仍远低于 direct_test
3. 评分规则确认：`0.7*teacher + 0.3*clip_prob_orig`，`keep_label=True`

### Traceability
- Run root: `work_dirs/rsar_sfodrs_full_fix_20260424_172627`
- Results: `rsar_sfodrs_results.csv`, `rsar_sfodrs_results.md`

---

## Experiment B: Loose SFOD-RS（Faithful Dataloader，weight_l=0.5，Phase 8）

**配置**：`RSAR_LOADER_MODE=loose`，`weight_l=0.5`，`use_labeled=True`  
数据集：`SemiRSARSFODDataset`（labeled clean RSAR train + unlabeled corrupt val）  
源权重：同上（clean=0.5385）；`RSAR_PSEUDO_SCORE_THR=0.7`

### Per-Domain Results（3 corruption 验证集，E0123）
| corruption | direct | loose nocga | loose cga | Δ(loose nocga - direct) |
|---|---:|---:|---:|---:|
| chaff | 0.4690 | 0.4516 | 0.4580 | -0.0174 |
| gaussian_white_noise | 0.5429 | 0.4983 | 0.4988 | -0.0446 |
| noise_suppression | 0.2436 | **0.2996** | **0.2994** | **+0.0560** |

### 与 Strict 直接对比（已测3域）
| corruption | strict nocga | loose nocga | 改善 |
|---|---:|---:|---|
| chaff | 0.0106 | 0.4516 | 崩溃完全消除 (+4200%) |
| gaussian_white_noise | 0.0080 | 0.4983 | 崩溃完全消除 |
| noise_suppression | 0.0861 | 0.2996 | +0.2135 (+248%) |

### score_thr=0.5 消融（E0124，负结果）
| corruption | thr=0.7 nocga | thr=0.5 nocga | Δ |
|---|---:|---:|---:|
| chaff | 0.4516 | 0.4370 | -0.0146 |
| gaussian_white_noise | 0.4983 | 0.5011 | +0.0028 |
| noise_suppression | 0.2996 | 0.2994 | -0.0002 |

**结论**：thr=0.5 持平或更差；额外伪框为纯噪声（pseudo_num(acc)=0.000 全程）

### Observations
1. **Loose mode 根治 collapse**：weight_l=0.5 clean 梯度锚点防止 teacher EMA 漂移
2. **noise_suppression 真实增益**：+0.056 mAP (+23% relative），本项目唯一 adapt > direct 的域
3. **重度腐败保持接近 direct 水平**（-0.01~-0.04），无崩溃
4. **CGA 贡献边际化**：loose mode 下 CGA 增益 <0.007（strict 下 +0.043 mean）
5. **最优配置**：`RSAR_LOADER_MODE=loose RSAR_WEIGHT_L=0.5 RSAR_PSEUDO_SCORE_THR=0.7`

### Traceability
- Run root (E0123, thr=0.7): `work_dirs/rsar_sfodrs_loose_ablation`
- Run root (E0124, thr=0.5): `work_dirs/rsar_sfodrs_loose_scrthr05_ablation`
- 启动日志: `work_dirs/rsar_sfodrs_loose_ablation/launch.log`

---

---

## Experiment C: Corr-Aug Source + Loose SFOD-RS（Phase 9，全7域）

**配置**：`RSAR_CORR_AUG=1 RSAR_CORR_AUG_PROB=0.5`，`RSAR_LOADER_MODE=loose`，`weight_l=0.5`  
源权重：`work_dirs/rsar_corraug_loose_20260504/source_train/latest.pth`（clean=0.5125，新源模型）  
关键差异：源训练时对 clean 图随机施加 7 种 SAR 腐败（p=0.5），提升 domain invariance

### 源模型对比（clean test mAP）
| 源模型 | clean test |
|---|---:|
| 原始源模型（Phase 5-8） | 0.5385 |
| Corr-aug 源模型（Phase 9） | 0.5125 |

### Per-Domain Results（全7域，E0125 direct + E0126 loose）
| corruption | direct | loose nocga | loose cga | Δ(direct-old) | Δ(nocga vs direct) |
|---|---:|---:|---:|---:|---:|
| chaff | 0.4899 | 0.4574 | 0.4509 | +0.0209 | -0.0325 |
| gaussian_white_noise | 0.5154 | 0.4805 | 0.4770 | -0.0275 | -0.0349 |
| point_target | 0.5100 | 0.4757 | 0.4766 | +0.1917 | -0.0343 |
| noise_suppression | 0.4701 | 0.4359 | 0.4384 | +0.2265 | -0.0342 |
| am_noise_horizontal | 0.4546 | 0.3776 | 0.3826 | +0.1561 | -0.0770 |
| smart_suppression | 0.4245 | 0.4026 | 0.3978 | +0.1310 | -0.0219 |
| am_noise_vertical | 0.4548 | 0.3979 | 0.3979 | +0.1581 | -0.0569 |
| **Mean (7 corr)** | **0.4742** | **0.4325** | **0.4316** | **+0.1370** | **-0.0417** |

### Corr-Aug Direct vs 原始 Direct（全7域）
| corruption | old direct (Phase5-8) | new direct (corr-aug) | Δ |
|---|---:|---:|---:|
| chaff | 0.4690 | 0.4899 | **+0.0209** |
| gaussian_white_noise | 0.5429 | 0.5154 | -0.0275 |
| point_target | 0.3183 | 0.5100 | **+0.1917** |
| noise_suppression | 0.2436 | 0.4701 | **+0.2265** |
| am_noise_horizontal | 0.2985 | 0.4546 | **+0.1561** |
| smart_suppression | 0.2935 | 0.4245 | **+0.1310** |
| am_noise_vertical | 0.2967 | 0.4548 | **+0.1581** |
| **Mean** | **0.3372** | **0.4742** | **+0.1370 (+40.6%)** |

### Observations
1. **Corr-aug 全面提升 direct_test 基线**：mean 0.3372 → 0.4742（+40.6% relative），6/7 域有改善
2. **gwn 小幅下降**：原始源模型对 GWN 天然鲁棒（源模型 clean 本身接近 GWN 频谱），corr-aug 略微干扰
3. **loose adaptation 仍导致 mAP 回退**（-0.04 mean）：pseudo_num(acc)=0.000 贯穿全程，适应无有效信号
4. **CGA 作用边际**：loose corr-aug 下 CGA vs nocga 差异 <0.005 mean
5. **最优策略确认**：corr-aug 源模型 + direct_test 优于任何适应方法

### Traceability
- Source train: `work_dirs/rsar_corraug_loose_20260504/source_train`
- Run root: `work_dirs/rsar_corraug_loose_20260504`
- 完成时间：2026-05-05 17:59 CST

---

## Experiment D: 阈值退火修复伪标签（E0127，thr=0.2→0.4 linear，进行中）

**背景**：E0125/E0126 中 `pseudo_num(acc)=0.000` 被误判为"伪标签全部被过滤"。诊断后发现：
- `pseudo_kept ≈ 1.74/img`（伪标签确实存在），`mean_score=0.916`（高置信度）
- `pseudo_num(acc)` 是 IoU-matched TP，目标域无 GT → 该指标恒为 0，是误导性指标
- 真实问题：thr=0.7 只选 18% NMS 输出，仅覆盖已知模式，无新适应信号
- **修复**：将 score_thr 从 0.7 降至 0.2 并线性退火至 0.4，同时暴露 `RSAR_THR_SCHEDULE` env var 到 config

**配置**：`RSAR_PSEUDO_SCORE_THR=0.2 RSAR_THR_SCHEDULE=linear RSAR_SCORE_THR_START=0.2 RSAR_SCORE_THR_END=0.4`  
源权重：`work_dirs/rsar_corraug_loose_20260504/source_train/latest.pth`（clean=0.5125）  
GPU：2,3,7（NGPUS=3），RSAR_SAMPLES_PER_GPU=8，MASTER_PORT=29513  
Run root：`work_dirs/rsar_e0127_thr_anneal_20260505_213455`  
启动时间：2026-05-05 21:34 CST

### Pseudo-Label 统计（chaff，前12 epoch）
| epoch | pseudo_kept | pseudo/img | mean_score | ship占比 |
|---:|---:|---:|---:|---:|
| 0 | 32,041 | 3.78 | 0.623 | 57.5% |
| 6 | 57,405 | 6.78 | 0.556 | 66.4% |
| 11 | 54,388 | 6.42 | 0.567 | 66.1% |

vs 旧 thr=0.7：pseudo/img=1.74，mean_score=0.916。新阈值产生 3~7 倍更多伪标签但质量更低。

### Per-Domain Results（已完成5域，2域进行中）
| corruption | direct | E0127 nocga | Δ | E0127 cga | Δ |
|---|---:|---:|---:|---:|---:|
| chaff | 0.4899 | 0.4061 | -8.4pp | 0.4067 | -8.3pp |
| gaussian_white_noise | 0.5154 | 0.4448 | -7.0pp | 0.4469 | -6.8pp |
| point_target | 0.5100 | 0.4501 | -6.0pp | 0.4493 | -6.1pp |
| noise_suppression | 0.4701 | 0.3810 | -8.9pp | 0.3803 | -9.0pp |
| am_noise_horizontal | 0.4546 | 0.2613 | **-19.4pp** | 0.2612 | **-19.4pp** |
| smart_suppression | 0.4245 | 0.3301 | -9.4pp | 0.3324 | -9.2pp |
| am_noise_vertical | 0.4548 | 0.3068 | **-14.8pp** | 进行中 | — |
| **Mean (7域)** | **0.4742** | **0.3686** | **-10.6pp** | — | — |

**关键发现**：
- am_noise_horizontal（-19.4pp）和 am_noise_vertical（-14.8pp）灾难性崩溃，两域 pseudo/img 分别为 8.88 和 10.0
- 7域 nocga 均值 0.3686，远低于旧 E0125/E0126（0.4325），降低阈值反而大幅加剧退步
- CGA 对崩溃域无任何帮助（am_noise_horizontal cga=0.2612 ≈ nocga=0.2613）

---

## Experiment E: 固定阈值 thr=0.4（E0128，进行中）

**目标**：对照 E0127，验证更高固定阈值是否减少伪标签噪声  
**配置**：`RSAR_PSEUDO_SCORE_THR=0.4`（无退火，无 RSAR_THR_SCHEDULE）  
GPU：8,9（NGPUS=2），RSAR_SAMPLES_PER_GPU=8，MASTER_PORT=29517  
Run root：`work_dirs/rsar_e0128_thr04_fixed_20260506_011722`  
启动时间：2026-05-06 01:17 CST

### Pseudo-Label 统计（chaff，前4 epoch）
| epoch | pseudo/img | mean_score | ship占比 |
|---:|---:|---:|---:|
| 0 | 2.46 | 0.800 | 66.8% |
| 3 | 3.33 | 0.763 | 72.6% |

**关键发现**：更高阈值选出的伪标签 score 更高（0.76-0.80 vs 0.57-0.62），但 ship 占比反而**更高**（72% vs 65%），证明 ship 类置信度本身更高，类别不平衡是固有问题。

### Per-Domain Results（已完成3域，4域进行中）
| corruption | direct | E0128 nocga | Δ | E0128 cga | Δ |
|---|---:|---:|---:|---:|---:|
| chaff | 0.4899 | 0.4349 | -5.5pp | 0.4413 | -4.9pp |
| gaussian_white_noise | 0.5154 | 0.4726 | -4.2pp | 0.4730 | -4.2pp |
| point_target | 0.5100 | 0.4676 | -4.2pp | 0.4728 | -3.7pp |
| noise_suppression | 0.4701 | 0.4331 | -3.7pp | 进行中 | — |

**对比结论**：thr=0.4 比 thr=0.2 少退步 ~2.8pp，但仍全部为负，适应仍然有害。

---

## 综合对比：所有方法（全7域）

### Mean mAP（7 corruption domains 均值）
| 方法 | 源模型 | mean direct | mean adapt | 状态 |
|---|---|---:|---:|---|
| strict nocga | 原始 | 0.3372 | 0.0402 | ❌ 完全崩溃 |
| strict +cga | 原始 | 0.3372 | 0.0833 | ⚠️ 部分恢复 |
| loose nocga | 原始 | 0.3372 | ~0.42* | ✅ 无崩溃（3域） |
| loose +cga | 原始 | 0.3372 | ~0.42* | ✅ 无崩溃（3域） |
| **corr-aug direct** | **corr-aug** | **0.4742** | — | **✅ 当前最优** |
| corr-aug loose nocga | corr-aug | 0.4742 | 0.4325 | ✅ 无崩溃（全7域） |
| corr-aug loose cga | corr-aug | 0.4742 | 0.4316 | ✅ 无崩溃（全7域） |
| E0127 thr=0.2→0.4 nocga | corr-aug | 0.4742 | 0.3686‡| ❌ 全域退步，am/sm崩溃 |
| E0128 thr=0.4 fixed nocga | corr-aug | 0.4742 | ~0.454‡| ❌ 全域退步（2域） |

*\*3域（chaff/gwn/noise_sup）均值估算*  
‡E0127 7域 nocga 均值（am_noise_vertical cga 进行中）：(0.4061+0.4448+0.4501+0.3810+0.2613+0.3301+0.3068)/7=0.3686  
‡2域已完成 nocga 均值：(0.4349+0.4726)/2=0.4538（尚不完整）

### Per-Domain 全方法对比
| corruption | strict-nocga | strict-cga | loose-nocga(orig) | corr-direct | corr-loose-nocga | E0127-nocga | E0128-nocga |
|---|---:|---:|---:|---:|---:|---:|---:|
| chaff | 0.0106 | 0.0833 | 0.4516 | **0.4899** | 0.4574 | 0.4061 | 0.4349 |
| gwn | 0.0080 | 0.0816 | 0.4983 | **0.5154** | 0.4805 | 0.4448 | 0.4726 |
| point_target | 0.0448 | 0.1018 | — | **0.5100** | 0.4757 | 0.4501 | 0.4676 |
| noise_suppression | 0.0861 | 0.0952 | 0.2996 | **0.4701** | 0.4359 | 0.3810 | 进行中 |
| am_noise_horizontal | 0.0292 | 0.0587 | — | **0.4546** | 0.3776 | 0.2613 | 进行中 |
| smart_suppression | 0.0538 | 0.0720 | — | **0.4245** | 0.4026 | 进行中 | 进行中 |
| am_noise_vertical | 0.0301 | 0.0714 | — | **0.4548** | 0.3979 | 进行中 | 进行中 |

### 结论
- **Corr-aug 源模型 + direct_test 是当前最优策略**（mean 0.4742）
- **伪标签阈值调优无法修复适应问题**：thr=0.2（+247%伪标签）和 thr=0.4 均系统性退步，根因是类别不平衡（ship 占 66-73%）
- **am_noise_horizontal 为极端失败案例**：pseudo/img=8.88 导致 E0127 nocga -19.4pp 灾难性遗忘
- **E0128 优于 E0127** 约 2.8pp（更高阈值减少噪声），但仍无法超越 direct
- **根本障碍**：UnbiasedTeacher 在无 GT 监督下向 ship 主导崩溃，伪标签质量/类平衡无法通过阈值解决
- **下一步方向**：TENT（只更新 BN 统计量，零伪标签，无优化漂移风险）
