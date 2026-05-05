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

## 综合对比：所有方法（全7域）

### Mean mAP（7 corruption domains 均值）
| 方法 | 源模型 | mean direct | mean adapt | 状态 |
|---|---|---:|---:|---|
| strict nocga | 原始 | 0.3372 | 0.0402 | ❌ 完全崩溃 |
| strict +cga | 原始 | 0.3372 | 0.0833 | ⚠️ 部分恢复 |
| loose nocga | 原始 | 0.3372 | ~0.42* | ✅ 无崩溃（3域） |
| loose +cga | 原始 | 0.3372 | ~0.42* | ✅ 无崩溃（3域） |
| **corr-aug direct** | **corr-aug** | **0.4742** | — | **✅ 最优基线** |
| corr-aug loose nocga | corr-aug | 0.4742 | 0.4325 | ✅ 无崩溃（全7域） |
| corr-aug loose cga | corr-aug | 0.4742 | 0.4316 | ✅ 无崩溃（全7域） |

*\*3域（chaff/gwn/noise_sup）均值估算*

### Per-Domain 全方法对比（已完成域）
| corruption | strict-nocga | strict-cga | loose-nocga(orig) | corr-direct | corr-loose-nocga |
|---|---:|---:|---:|---:|---:|
| chaff | 0.0106 | 0.0833 | 0.4516 | **0.4899** | 0.4574 |
| gwn | 0.0080 | 0.0816 | 0.4983 | 0.5154 | 0.4805 |
| point_target | 0.0448 | 0.1018 | — | **0.5100** | 0.4757 |
| noise_suppression | 0.0861 | 0.0952 | 0.2996 | **0.4701** | 0.4359 |
| am_noise_horizontal | 0.0292 | 0.0587 | — | **0.4546** | 0.3776 |
| smart_suppression | 0.0538 | 0.0720 | — | **0.4245** | 0.4026 |
| am_noise_vertical | 0.0301 | 0.0714 | — | **0.4548** | 0.3979 |

### 结论
- Corr-aug 源模型 + direct_test 是当前最优策略（mean 0.4742）
- Loose adaptation 在 corr-aug 源模型基础上仍回退（-0.04），根因是 pseudo_num(acc)=0.000
- 提升 adaptation 效果的关键障碍：伪标签质量极低，需探索更强的伪标签滤波或在线腐败增强结合
