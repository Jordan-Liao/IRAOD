# RSAR 半监督目标检测实验记录

## 概述

本文档记录了在 RSAR 遥感数据集（6类：ship, aircraft, car, tank, bridge, harbor）上进行的半监督目标检测系列实验。基于 Unbiased Teacher + Oriented RCNN 框架，通过多轮迭代优化，从 baseline mAP 0.6599 提升至 **0.6701**（+1.02%）。

**框架**：Unbiased Teacher（EMA teacher-student 半监督框架）+ Oriented RCNN（旋转框检测器）
**数据集**：RSAR，6 类遥感目标：ship(10700), aircraft(494), car(1231), tank(886), bridge(3272), harbor(399)
**标注设置**：有标注数据 + 无标注数据（伪标签半监督学习）
**评估指标**：mAP（mean Average Precision）

---

## Baseline

**配置**: `configs/unbiased_teacher/sfod/ut_oriented_rcnn_r50_rsar_corrected.py`
**Checkpoint**: `work_dirs/rsar_r50_corrected/iter_31537.pth`
**mAP**: 0.6599 (epoch 8)

| 类别 | GT数 | 检测数 | Recall | AP |
|------|------|--------|--------|----|
| ship | 10700 | 106016 | 0.884 | 0.779 |
| aircraft | 494 | 656 | 0.870 | 0.769 |
| car | 1231 | 1843 | 0.966 | 0.895 |
| tank | 886 | 1869 | 0.442 | 0.334 |
| bridge | 3272 | 8189 | 0.674 | 0.563 |
| harbor | 399 | 1335 | 0.749 | 0.620 |
| **mAP** | | | | **0.660** |

**关键问题**：
- ship 虚检严重（106K检测 vs 10.7K GT）
- tank AP 最低（0.334），假阳性过多
- bridge 过度检测（8189 vs 3272 GT）

---

## Round 1：单因素实验（Exp A-E）

每次只改变一个变量，探索各改进方向的效果。所有实验均从 baseline checkpoint 启动。

### Exp A: 高阈值 + 高权重 (weight_u=2.0, score_thr=0.7)
- **配置**: `exp_a_high_thr_high_wu.py`
- **假设**: 提高伪标签阈值到0.7过滤噪声，同时加大无监督权重到2.0加速学习
- **核心参数**: `score_thr=0.7, weight_u=2.0`
- **结果**: mAP 0.6564 — **失败**
- **分析**: weight_u=2.0 放大了伪标签噪声，噪声放大效应超过了高阈值的过滤能力

### Exp B: 阈值退火 (thr_schedule=linear, 0.7→0.5)
- **配置**: `exp_b_thr_anneal.py`
- **假设**: 训练初期用高阈值（0.7）过滤噪声，后期降至0.5增加伪标签数量
- **核心参数**: `thr_schedule=linear, score_thr_start=0.7, score_thr_end=0.5`
- **结果**: mAP 0.6608 (epoch 8)
- **分析**: pseudo_num(acc) 最高81%，退火策略本身有效，但提升有限

### Exp C: 扩大无标注池 (train set 78.8K图)
- **配置**: `exp_c_more_unlabeled.py`
- **假设**: 用全部训练集（78.8K图）作为无标注数据池（而非默认的少量），增加伪标签多样性
- **核心参数**: `ann_file_u=train_ann, img_prefix_u=train_img`（约78.8K无标注图）
- **结果**: mAP 0.6619 (epoch 9)
- **分析**: 最稳定的实验，mAP曲线波动最小，但绝对提升不大

### Exp D: 强增强 DTRandCrop
- **配置**: `exp_d_strong_aug.py`
- **假设**: 在强增强 pipeline 中加入 DTRandCrop 随机裁剪，强迫模型学习局部特征
- **核心参数**: unsup_pipeline_strong 中添加 `DTRandCrop`
- **结果**: mAP 0.6625 (epoch 5)
- **分析**: 收敛最快，epoch 5即达最优；DTRandCrop 提供了有效的正则化

### Exp E: 类自适应阈值
- **配置**: `exp_e_class_adaptive.py`
- **假设**: 不同类别伪标签质量差异大，为每个类单独设置阈值。难检类(tank)用高阈值过滤，易检类(ship)用低阈值
- **核心参数**: `score_thr=[0.5, 0.5, 0.5, 0.8, 0.7, 0.6]`（ship/aircraft/car/tank/bridge/harbor）
- **结果**: mAP 0.6626 (epoch 8)
- **分析**: tank AP 从 0.334 提升到 0.383（+4.9%），验证了类自适应阈值对低AP类的有效性

### Round 1 小结

| 排名 | 实验 | 最佳 mAP | Best Epoch | 核心改进 |
|------|------|---------|------------|---------|
| 1 | Exp E（类自适应阈值） | 0.6626 | 8 | tank AP +4.9% |
| 2 | Exp D（强增强 DTRandCrop） | 0.6625 | 5 | 收敛最快 |
| 3 | Exp C（大无标注池） | 0.6619 | 9 | 最稳定 |
| 4 | Exp B（阈值退火） | 0.6608 | 8 | pseudo_num(acc) 81% |
| 5 | Exp A（高 weight_u） | 0.6564 | - | 失败 |

**关键结论**：
- per-class 阈值 > 全局阈值（Exp E > B）
- 强增强加速收敛（Exp D）
- weight_u=2.0 过大，1.0 可能仍偏大
- 后期 mAP 下降是共性问题：LR step 后伪标签噪声自我强化

---

## Round 2：组合实验（Exp F-J）

组合 Round 1 最优策略，同时引入新策略：低weight_u、长burn-in、激进类阈值。

### Exp F: D+E 组合（强增强 + 类自适应）
- **配置**: `exp_f_de_combo.py`
- **核心参数**: DTRandCrop + score_thr=[0.5,0.5,0.5,0.8,0.7,0.6], weight_u=1.0
- **结果**: mAP 0.6634 (epoch 2)
- **分析**: 组合未产生明显叠加增益

### Exp G: D+E+C 全组合
- **配置**: `exp_g_dec_combo.py`
- **核心参数**: DTRandCrop + per-class阈值 + 78.8K大无标注池
- **结果**: mAP 0.6629 (epoch 8)
- **分析**: 三者组合反而互相干扰

### Exp H: 保守策略（E + 低权重 + 长burn-in）★
- **配置**: `exp_h_conservative.py`
- **核心参数**: score_thr=[0.5,0.75,0.5,0.85,0.75,0.6], **weight_u=0.5**, **burn_in=4 epoch**, total_epoch=16
- **结果**: **mAP 0.6679** (epoch 12)
- **分析**: 低weight_u+长burn-in 显著延缓了伪标签噪声积累，mAP在更晚的epoch才达峰

### Exp I: 激进类阈值（tank禁用伪标签）
- **配置**: `exp_i_aggressive_cls.py`
- **核心参数**: score_thr=[0.45,0.45,0.45,**1.0**,**0.85**,0.55]（tank阈值1.0=完全禁用伪标签）
- **结果**: mAP 0.6651 (epoch 2)
- **分析**: 禁用tank伪标签有效果但不如预期

### Exp J: 全组合 + 阈值退火
- **配置**: `exp_j_full_anneal.py`
- **核心参数**: DTRandCrop + 大无标注池 + per-class高阈值 + linear退火
- **结果**: mAP 0.6585 (epoch 1)
- **分析**: 过多技巧叠加导致训练不稳定，是最差的一组

### Round 2 小结

| 排名 | 实验 | 最佳 mAP | Best Epoch |
|------|------|---------|------------|
| 1 | Exp H（保守策略） | **0.6679** | 12 |
| 2 | Exp I（激进类阈值） | 0.6651 | 2 |
| 3 | Exp F（D+E组合） | 0.6634 | 2 |
| 4 | Exp G（D+E+C全组合） | 0.6629 | 8 |
| 5 | Exp J（全组合+退火） | 0.6585 | 1 |

**关键结论**：
- **降低 weight_u + 延长 burn-in 是最有效策略**（Exp H 碾压所有组合）
- 简单组合 Round 1 最优策略并不叠加（F/G 不如单独的 E/D）
- 过多技巧叠加会互相干扰（Exp J 最差）
- weight_u 从 1.0 降至 0.5 是关键因素

---

## Round 3：精细化实验（Exp K-O）

基于 Exp H 的保守策略（wu=0.5, burn_in=4），引入新机制。

### 源码改进
在 `sfod/rotated_unbiased_teacher.py` 中新增：
- `weight_u_schedule`: 按epoch动态调整无监督权重，如 `{4: 0.0, 5: 0.25, 7: 0.5, 12: 0.25}`
- `weight_u_bbox`: 分离 bbox 回归的无监督权重（默认跟随 weight_u）
- `class_score_thr_start` / `class_score_thr_end`: per-class 阈值退火的起止点

在 `sfod/dense_teacher_rand_aug.py` 中新增：
- `ConditionalDTRandCrop`: 根据全局 epoch 决定是否启用强裁剪
- `_GLOBAL_EPOCH` / `set_global_epoch()`: 全局 epoch 追踪

### Exp K: 类自适应阈值 v2 + 大无标注池 + wu=0.5
- **配置**: `exp_k_cls_adaptive_v2.py`
- **核心参数**: score_thr=[0.5,0.75,0.5,0.85,0.75,0.6], weight_u=0.5, 大无标注池, burn_in=4
- **结果**: mAP 0.6635 (epoch 11)

### Exp L: Per-class 阈值退火
- **配置**: `exp_l_perclass_anneal.py`
- **核心参数**: score_thr从[0.55,0.90,0.55,1.0,0.85,0.65]退火到[0.50,0.75,0.50,0.85,0.75,0.60]
- **结果**: mAP 0.6667 (epoch 7)
- **分析**: 退火策略适度有效，但不如直接调 weight_u schedule

### Exp M: weight_u epoch-based schedule ★★★ 最佳
- **配置**: `exp_m_wu_schedule.py`
- **核心参数**:
  - `score_thr=[0.5, 0.75, 0.5, 0.85, 0.75, 0.6]`
  - `weight_u_schedule={4: 0.0, 5: 0.25, 7: 0.5, 12: 0.25}`
  - 含义：epoch 1-4 burn-in(wu=0), epoch 5 wu=0.25, epoch 7 wu=0.5(峰值), epoch 12 wu=0.25(衰减)
  - LR step at epoch [12, 15]
- **结果**: **mAP 0.6701** (epoch 13) — 全局最佳
- **Per-class AP**:

| 类别 | GT | Dets | Recall | AP |
|------|-----|------|--------|------|
| ship | 10700 | 14558 | 0.892 | 0.790 |
| aircraft | 494 | 595 | 0.826 | 0.726 |
| car | 1231 | 1518 | 0.968 | 0.898 |
| tank | 886 | 1507 | 0.503 | 0.414 |
| bridge | 3272 | 5731 | 0.662 | 0.551 |
| harbor | 399 | 934 | 0.744 | 0.641 |
| **mAP** | | | | **0.670** |

- **分析**:
  - **核心创新**：weight_u 的 warmup→peak→decay 调度，与 LR step 在 epoch 12 对齐
  - ship 虚检从106K降至14.6K（-86%）
  - tank AP 从 0.334 提升至 0.414（+8.0%）
  - harbor AP 从 0.620 提升至 0.641（+2.1%）
  - **代价**：aircraft AP 从 0.769 降至 0.726（-4.3%），是伪标签训练的副作用

### Exp N: ConditionalDTRandCrop (epoch>=5 启用强裁剪)
- **配置**: `exp_n_cond_crop.py`
- **核心参数**: ConditionalDTRandCrop 在 epoch >= 5 时才启用 DTRandCrop 强增强
- **结果**: mAP 0.6662 (epoch 6)

### Exp O: 分离 bbox/cls 无监督权重
- **配置**: `exp_o_bbox_split.py`
- **核心参数**: weight_u=0.5（cls部分）, weight_u_bbox=0.2（bbox回归）
- **结果**: mAP 0.6685 (epoch 13)
- **分析**: bbox 回归更易被伪标签噪声干扰，降低其权重有效

### Round 3 小结

| 排名 | 实验 | 最佳 mAP | Best Epoch |
|------|------|---------|------------|
| 1 | **Exp M（wu schedule）** | **0.6701** | 13 |
| 2 | Exp O（bbox split） | 0.6685 | 13 |
| 3 | Exp L（per-class退火） | 0.6667 | 7 |
| 4 | Exp N（CondCrop） | 0.6662 | 6 |
| 5 | Exp K（cls adaptive v2） | 0.6635 | 11 |

---

## Round 3b：基于 Exp M 的变体（Exp P-R）

### Exp P: M + aircraft 保护（阈值0.95）
- **配置**: `exp_p_m_aircraft_protect.py`
- **核心参数**: M 基础上 aircraft 阈值提高到 0.95（接近禁用 aircraft 伪标签）
- **结果**: mAP 0.6652 (epoch 4)
- **分析**: 禁用aircraft伪标签反而使其AP下降更多

### Exp Q: M + bbox split (wu_bbox=0.2)
- **配置**: `exp_q_m_bbox_split.py`
- **核心参数**: M 的 wu_schedule + Exp O 的 weight_u_bbox=0.2
- **结果**: mAP 0.6664 (epoch 5)
- **分析**: 两种好策略组合并未叠加

### Exp R: M + 长训练 (burn_in=6, 20 epochs)
- **配置**: `exp_r_m_long.py`
- **核心参数**: M 基础 + burn_in=6 + total_epoch=20 + lr_step=[16,19]
- **结果**: mAP 0.6652 (epoch 11)
- **分析**: 更长训练不能提升上限，反而后期 mAP 回落

---

## Round 4：后处理方法（Exp S1, S4, SWA, Interpolation）— 进行中

### 方法1 - Exp S1: 监督微调
- **配置**: `exp_s1_sup_finetune.py`
- **核心参数**: 从 Exp M ep13 加载权重, weight_u=0（纯监督）, lr=1e-4, 仅2个epoch
- **假设**: 用小LR纯监督训练消除伪标签引入的噪声（尤其是 aircraft 的 AP 下降）
- **状态**: 训练中

### 方法2 - SWA（随机权重平均）
- **操作**: 将 Exp M 的 epoch 10/11/12/13 四个 checkpoint 的 state_dict 取平均
- **假设**: 权重平均可以平滑不同 epoch 的偏差，提高泛化
- **Checkpoint**: `work_dirs/exp_m_wu_schedule/swa_ep10_13.pth`
- **状态**: 评估中

### 方法3 - 模型插值（Exp M × baseline）
- **操作**: `alpha * ExpM_ep13 + (1-alpha) * baseline`，alpha = 0.9, 0.8, 0.7, 0.5
- **假设**: baseline 的 aircraft AP 更高（0.769），插值可以恢复部分 aircraft 性能同时保留 Exp M 的其他增益
- **Checkpoints**: `work_dirs/exp_m_wu_schedule/interp_a{0.9,0.8,0.7,0.5}.pth`
- **状态**: 评估中

### 方法4 - Exp S4: aircraft 禁用 + 缓和wu_schedule
- **配置**: `exp_s4_no_aircraft.py`
- **核心参数**: aircraft阈值=1.0(完全禁用), wu_schedule={4:0, 5:0.15, 7:0.35, 12:0.2}（更缓和）
- **假设**: 完全禁用 aircraft 伪标签 + 降低 wu 峰值，保护 aircraft AP
- **状态**: 训练中

---

## 全局排名

| 排名 | 实验 | 最佳 mAP | Best Epoch | 轮次 |
|------|------|---------|------------|------|
| **1** | **Exp M（wu schedule）** | **0.6701** | 13 | Round 3 |
| 2 | Exp O（bbox split） | 0.6685 | 13 | Round 3 |
| 3 | Exp H（保守策略） | 0.6679 | 12 | Round 2 |
| 4 | Exp L（per-class退火） | 0.6667 | 7 | Round 3 |
| 5 | Exp Q（M+bbox） | 0.6664 | 5 | Round 3b |
| 6 | Exp N（CondCrop） | 0.6662 | 6 | Round 3 |
| 7 | Exp P（aircraft=0.95） | 0.6652 | 4 | Round 3b |
| 8 | Exp R（20ep长训练） | 0.6652 | 11 | Round 3b |
| 9 | Exp I（激进类阈值） | 0.6651 | 2 | Round 2 |
| 10 | Exp K（cls adaptive v2） | 0.6635 | 11 | Round 3 |
| 11 | Exp F（D+E组合） | 0.6634 | 2 | Round 2 |
| 12 | Exp C（大无标注池） | 0.6619 | 9 | Round 1 |
| 13 | Exp G（D+E+C全组合） | 0.6629 | 8 | Round 2 |
| 14 | Exp E（类自适应） | 0.6626 | 8 | Round 1 |
| 15 | Exp D（强增强） | 0.6625 | 5 | Round 1 |
| 16 | Exp B（阈值退火） | 0.6608 | 8 | Round 1 |
| - | **Baseline** | **0.6599** | 8 | - |
| 17 | Exp J（全组合+退火） | 0.6585 | 1 | Round 2 |
| 18 | Exp A（高weight_u） | 0.6564 | - | Round 1 |

---

## Exp M vs Baseline 对比

| 类别 | Baseline AP | Exp M AP | 变化 |
|------|------------|---------|------|
| ship | 0.779 | 0.790 | **+1.1%** |
| aircraft | 0.769 | 0.726 | **-4.3%** |
| car | 0.895 | 0.898 | +0.3% |
| tank | 0.334 | 0.414 | **+8.0%** |
| bridge | 0.563 | 0.551 | -1.2% |
| harbor | 0.620 | 0.641 | **+2.1%** |
| **mAP** | **0.660** | **0.670** | **+1.0%** |

**核心发现**：
1. **tank 提升最大**（+8.0%），得益于 per-class 高阈值过滤噪声 + weight_u schedule 控制噪声积累
2. **aircraft 下降最大**（-4.3%），是半监督训练的主要副作用（aircraft 样本少、伪标签质量低）
3. **ship 虚检大幅减少**（106K→14.6K），检测质量显著改善
4. 如能恢复 aircraft AP，mAP 预计可达 0.677+

---

## 关键经验总结

1. **weight_u schedule 是最有效的策略**：warmup→peak→decay 让模型在不同阶段接受不同程度的伪标签信号
2. **简单组合不等于叠加**：Round 1 各项最优的简单组合（F/G/J）反而不如单项改进
3. **降低 weight_u 比提高阈值更有效**：Exp H(wu=0.5) > Exp E(类阈值)
4. **burn-in 期至关重要**：4 epoch burn-in 让模型在引入伪标签前充分稳定
5. **伪标签后期退化是核心瓶颈**：所有实验在一定 epoch 后 mAP 下降，weight_u 衰减是最好的缓解手段
6. **少样本类（aircraft）最易受损**：伪标签训练系统性损害少样本类，需要专门保护策略


---

## Phase 2: 方法创新实验

### 创新点 1: OrthoNet + OCA-FPN

将正交通道注意力网络 ([OrthoNets](https://github.com/hady1011/OrthoNets)) 融入检测框架:

- **OrthoNet backbone**: 替换 ResNet50，注册为 `OrthoNet`
- **OCA-FPN neck**: FPN 每一层输出叠加 OrthoChannelAttention
- **Config**: `unbiased_teacher_oriented_rcnn_selftraining_cga_o_rsar.py`（RSAR）, `unbiased_teacher_oriented_rcnn_selftraining_cga_o.py`（DIOR）
- **frontier-026**: OCA-FPN + 24ep schedule，Val mAP=0.6511 (Epoch 11)，训练中断于 Epoch 13/24

### 创新点 2: SAR-CLIP LoRA 微调

用 SARDet100k 裁剪目标 patch + 干扰增强 → LoRA 微调 SAR-CLIP (ViT-L-14) → 提升 CGA 打分精度:

**核心逻辑**:
1. SARDet100k 训练集标注精确裁剪目标 patch（水平框）+ 叠加 7 种干扰
2. 构造图像-文本对: 图像 = 裁剪 patch，文本 = `"a SAR image of a {cls}"`
3. LoRA 微调 vision encoder（4 linear layers, 122,880 params, 0.12%）
4. 替换 SFOD-RS CGA scorer → 打分更准 → 伪标签质量更高

**代码**: `lora_finetune/lora_sarclip_train.py`, `lora_finetune/crop_sardet100k.py`

**结果**:
- CGA 零样本精度: 0.6021 → **0.6513** (+4.9%)
- 端到端检测 mAP: 0.6842 → **0.6943** (+1.0%)
- 全 SFOD 迭代链稳定 +1% 增益

详细实验记录见 `docs/experiment.md` E0042-E0096 和 `docs/plan.md` §6.4-6.5。
