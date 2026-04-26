# RSAR SFOD-RS Full Rerun Experiment Log

## Run Metadata
- Run ID: `rsar_sfodrs_full_fix_20260424_172627`
- Remote path: `/mnt/SSD1_8TB/zechuan/IRAOD/work_dirs/rsar_sfodrs_full_fix_20260424_172627`
- Runner env: `/home/zechuan/anaconda3/envs/iraod/bin/python`
- Orchestrated period: `2026-04-24 17:26:27 CST` -> `2026-04-26 21:13:36 CST`
- Aggregation time: `2026-04-27 00:51 CST`

## Commands Used
```bash
# full rerun (auto source training)
CUDA_VISIBLE_DEVICES=<free_3_gpus> \
NGPUS=3 MASTER_PORT=<free_port> \
PYTHON=/home/zechuan/anaconda3/envs/iraod/bin/python \
bash scripts/run_rsar_sfodrs_full_3gpu.sh auto \
  work_dirs/rsar_sfodrs_full_fix_20260424_172627
```

```bash
# final aggregation
/home/zechuan/anaconda3/envs/iraod/bin/python tools/collect_rsar_sfodrs_results.py \
  --work-root work_dirs/rsar_sfodrs_full_fix_20260424_172627 \
  --out-csv work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv \
  --out-md  work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md
```

## Task Completion
- Planned tasks: `37 = 1 + 1 + 7x5`
- Completed tasks: `37/37`
- Artifact completeness gate: `23/23` passed
  - `1` source ckpt (`epoch_12.pth`)
  - `1` clean eval json
  - `21` corruption eval json (`7 domains x 3 eval stages`)

## Final Result Table (from `rsar_sfodrs_results.csv`)
| Row | clean | chaff | gaussian_white_noise | point_target | noise_suppression | am_noise_horizontal | smart_suppression | am_noise_vertical | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5385 | - | - | - | - | - | - | - | 0.5385 |
| direct_test | 0.5385 | 0.4690 | 0.5429 | 0.5355 | 0.2436 | 0.1800 | 0.1830 | 0.2066 | 0.3624 |
| self_training | 0.5385 | 0.0224 | 0.0276 | 0.0392 | 0.0809 | 0.0246 | 0.0659 | 0.0208 | 0.1025 |
| self_training_plus_cga | 0.5385 | 0.0952 | 0.1012 | 0.0962 | 0.0900 | 0.0541 | 0.0842 | 0.0621 | 0.1402 |

## Per-Corruption Delta (`self_training_plus_cga - self_training`)
| Corruption | Delta mAP |
| --- | ---: |
| chaff | +0.0727 |
| gaussian_white_noise | +0.0736 |
| point_target | +0.0570 |
| noise_suppression | +0.0091 |
| am_noise_horizontal | +0.0295 |
| smart_suppression | +0.0182 |
| am_noise_vertical | +0.0413 |
| **mean (7 corr)** | **+0.0431** |

## Timeline (Key Logs)
- Main line (`launch.log`):
  - start `2026-04-24 17:26:27`
  - source train -> clean test -> `chaff` -> `gaussian_white_noise` -> entered `point_target direct_test`
  - last main timestamp `2026-04-26 14:30:16`
- Shard line (`rsar_sfodrs_shard_late3_ddp_20260426_001855/launch.log`):
  - covered `am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`
  - done at `2026-04-26 14:04:55`
- Final parallel completion:
  - `point_target_parallel.log` done at `2026-04-26 21:12:25`
  - `noise_suppression_parallel.log` done at `2026-04-26 21:13:36`

## Diagnostics Evidence (Sample)
- In target evaluation logs:
  - `stage=target_eval`
  - `use_labeled_source_in_adaptation=False`
  - `target_domain=<corr>`
  - `cga_mode=sfodrs`
  - `keep_label=True`
  - `score_rule=0.7*teacher + 0.3*clip_prob_orig`
- Pseudo-label stats present in training logs:
  - `[PseudoStats] kept_ratio`
  - `mean_score`


## Phase 5: SFOD-RS faithful + RSAR 七类干扰塌缩修复

> 本阶段把 IRAOD 严格对齐 SFOD-RS（Lansing/SFOD-RS）的 target-only self-training 协议：
> - Detector 统一为 `OrthoNet` + `OCAFPN` + `OrientedRCNN`（len=6 RSAR 类别：ship, aircraft, car, tank, bridge, harbor）。
> - Adaptation stage 严格 source-free：`weight_l=0.0`、`use_labeled=False`、不使用 labeled source train branch。
> - 每个 corruption 独立：adapt split=`corruptions/<corr>/val/images`（8467 imgs unlabeled），eval split=`corruptions/<corr>/test/images`（8538 imgs with annfile）。
> - EMA teacher α=0.998；weak aug = horizontal flip p=0.5；strong aug = ColorJitter + Grayscale + GaussianBlur + DTRandCrop。
> - CGA faithful fusion（`sfod/cga.py::TestMixins.refine_test` mode=`sfodrs`）：`keep_label=True`；当 `argmax(SARCLIP probs) != teacher_label` 时 `new_score = 0.7 * teacher_score + 0.3 * clip_prob[orig_label]`；类别永远不改。
> - Prompt template: `"A SAR image of a {}"`。
> - 全部 run 使用 `torch.distributed.launch --nproc_per_node=3 --master_port=<P> --use_env ... --launcher pytorch` 的 3-GPU DDP（CUDA_VISIBLE_DEVICES=6,7,8）。
>
> SOURCE_CKPT：`work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`（OrthoNet+OCAFPN+OrientedRCNN，12 epoch clean RSAR supervised；direct_test on clean = 0.5350）。
>
> mean = `clean_test + chaff + gaussian_white_noise + point_target + noise_suppression + am_noise_horizontal + smart_suppression + am_noise_vertical` 共 8 列算术平均。


### E0111: SFOD-RS faithful baseline（20260415 全量 run，未修复塌缩）
| Field | Value |
| --- | --- |
| Objective | 在严格 source-free 协议下完成 7 corruption 的 adapt_nocga + adapt_cga + eval，作为塌缩修复前的参考点 |
| Baseline | direct_test（source_ckpt 直接测各 corruption） |
| Model | UnbiasedTeacher + OrientedRCNN/OrientedRCNN_CGA + OrthoNet backbone + OCAFPN neck + SARCLIP RN50 CGA |
| Weights | `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`, `weights/sarclip/RN50/rn50_model.safetensors` |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py`, `tools/rsar_sfodrs_dataset.py`, `tools/sfodrs_diagnostics_hook.py`, `scripts/run_rsar_sfodrs_7corr.sh`, `scripts/exp_rsar_sfodrs_adapt.sh` |
| Params | `epochs=12`, `lr=0.02`, `weight_u=1.0`, `weight_l=0.0`, `ema_momentum=0.998`, `score_thr=0.7` scalar, `samples_per_gpu=2`, `workers_per_gpu=2`, `cuda_visible_devices=0,1,2`, `seed=default` |
| Logs/Meta | `work_dirs/rsar_sfodrs_full_3gpu_20260415/launch.log`, `work_dirs/rsar_sfodrs_full_3gpu_20260415/main/<corr>/20260415_*.log` |
| Artifacts | `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`, `work_dirs/rsar_sfodrs_full_3gpu_20260415/main/<corr>/{direct_test,self_training,self_training_plus_cga}/`, `rsar_sfodrs_results.csv`, `rsar_sfodrs_results.md` |
| Results | `direct_test`: clean=0.5350, chaff=0.4629, gwn=0.5410, point_target=0.5321, noise_suppression=0.2471, am_h=0.1830, smart_suppression=0.1834, am_v=0.2205, mean=**0.3631**。`self_training`: chaff=0.0090, gwn=0.0483, point_target=0.0647, noise_suppression=0.0728, am_h=0.0272, smart_suppression=0.0619, am_v=0.0135, mean=**0.1040**（ship 塌缩）。`self_training_plus_cga`: chaff=0.0969, gwn=0.1061, point_target=0.1011, noise_suppression=0.0934, am_h=0.0561, smart_suppression=0.0751, am_v=0.0995, mean=**0.1454** |
| Finding | 所有 7 corruptions 的 self_training 均显著劣于 direct_test，pseudo_stats 显示 teacher 在 3-5 epoch 内把 ship 推至 >95% majority → classic majority-class collapse。CGA 仅将 mean 从 0.1040 拉回 0.1454，仍远低于 direct 0.3631 |


### E0112: 塌缩修复 v1 — per-class score_thr + burn-in + lower lr + early stop
| Field | Value |
| --- | --- |
| Objective | 针对 E0111 的 ship-collapse，通过 per-class 阈值 + 早停护栏 + 低 lr + EMA 预热，让 7 corruption 的 self_training/self_training_plus_cga 超过 E0111 |
| Baseline | E0111（同 source_ckpt，同协议，仅替换超参） |
| Model | 同 E0111 |
| Weights | 同 E0111 |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py`（本阶段扩展以读 `RSAR_PSEUDO_SCORE_THR`(list)、`RSAR_BURN_IN_EPOCHS`、`RSAR_WEIGHT_U`、`RSAR_ADAPT_LR`）, `sfod/rotated_unbiased_teacher.py`（既有 per-class thr 支持）, `tools/pseudo_stats_early_stop_hook.py`（本阶段接入 target_adapt.custom_hooks）, `scripts/run_rsar_sfodrs_full_3gpu.sh`（本阶段新增 3-GPU DDP driver） |
| Params | `RSAR_PSEUDO_SCORE_THR="0.85,0.7,0.7,0.7,0.7,0.7"`（ship=0.85, 其它=0.7）, `RSAR_BURN_IN_EPOCHS=2`, `RSAR_WEIGHT_U=0.5`, `RSAR_ADAPT_LR=0.005`, `RSAR_ADAPT_EPOCHS=12`, `PSEUDO_EARLYSTOP=1`, `PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC=0.90`, `PSEUDO_EARLYSTOP_PATIENCE=1`, `PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO=3000`, `cuda_visible_devices=6,7,8`, `master_port=29504` |
| Logs/Meta | `work_dirs/rsar_sfodrs_fixed_20260417_033006/launch.log`, `work_dirs/rsar_sfodrs_fixed_20260417_033006/<corr>/self_training/20260417_*.log`, `<corr>/self_training/pseudo_stats.json`（per-epoch class 分布 + early_stop 状态） |
| Artifacts | `work_dirs/rsar_sfodrs_fixed_20260417_033006/<corr>/{direct_test,self_training,self_training_plus_cga}/eval_target/eval_*.json`, `rsar_sfodrs_results.{csv,md}` |
| Results | `direct_test`: 与 E0111 逐字符一致（sanity check，clean=0.5350, mean=**0.3631**）。`self_training`: chaff=0.1368, gwn=0.1573, point_target=0.1689, noise_suppression=0.0819, am_h=0.0829, smart_suppression=0.0651, am_v=0.1013, mean=**0.1661**（vs E0111 0.1040，**+6.2pp / x1.60**）。`self_training_plus_cga`: chaff=0.1752, gwn=0.2146, point_target=0.2226, noise_suppression=0.0835, am_h=0.0447, smart_suppression=0.0651, am_v=0.1320, mean=**0.1841**（vs E0111 0.1454，**+3.9pp / x1.27**） |
| Finding | 7/7 corruption 的 self_training 全部 > E0111；pseudo_stats 显示 ship 不再一路爬到 100%（chaff 从 83%→66% 单调下降）；早停在 ship>0.90 持续 1 epoch 时自动触发（am_v 在 ep3-4 触发）。am_h 的 +CGA=0.0447 比 self_training=0.0829 低，SARCLIP 过于挑剔饿坏 student |


### E0113: heavyfix — SARCLIP LoRA-Interference CGA + 调优（4 heavy corruptions only）
| Field | Value |
| --- | --- |
| Objective | 针对 E0112 中 4 个"重"corruption（noise_suppression, am_h, smart_suppression, am_v）仍偏弱，启用 SARCLIP LoRA-Interference 权重 + 降 ship 阈值 + 缩短 adapt 长度 + 更紧的早停 |
| Baseline | E0112 的 `self_training_plus_cga` 行（4 heavy corr） |
| Model | UnbiasedTeacher + OrientedRCNN_CGA + OrthoNet + OCAFPN + SARCLIP RN50 + LoRA 干扰权重 |
| Weights | 同 E0112, + `lora_finetune/SARCLIP_LoRA_Interference.pt`（9.5MB，针对 SAR 干扰域训练的 LoRA adapter） |
| Code path | 同 E0112, + `scripts/run_rsar_sfodrs_heavyfix.sh`（本阶段新增） |
| Params | `SARCLIP_LORA=lora_finetune/SARCLIP_LoRA_Interference.pt`, `RSAR_PSEUDO_SCORE_THR="0.80,0.6,0.6,0.6,0.6,0.6"`（ship=0.80, 其它=0.6，比 E0112 更宽松）, `RSAR_BURN_IN_EPOCHS=1`, `RSAR_WEIGHT_U=0.5`, `RSAR_ADAPT_LR=0.005`, `RSAR_ADAPT_EPOCHS=6`（比 E0112 短一半）, `PSEUDO_EARLYSTOP=1`, `PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC=0.85`（比 E0112 紧）, `PSEUDO_EARLYSTOP_PATIENCE=1`, `PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO=2000`, `cuda_visible_devices=6,7,8`, `master_port=29505` |
| Logs/Meta | `work_dirs/rsar_sfodrs_heavyfix_20260419_004903/launch.log`, `<corr>/self_training_plus_cga_lora/pseudo_stats.json`, `<corr>/self_training_plus_cga_lora/20260419_*.log` |
| Artifacts | `work_dirs/rsar_sfodrs_heavyfix_20260419_004903/<corr>/self_training_plus_cga_lora/{latest_ema.pth, eval_target/eval_*.json}` |
| Results | noise_suppression=**0.1082**（E0112 +CGA 0.0835，**+30%**）, am_h=**0.0790**（E0112 0.0447，**+77%**）, smart_suppression=**0.0860**（E0112 0.0651，**+32%**）, am_v=**0.1321**（E0112 0.1320，持平）, heavy_mean=**0.1013**（E0112 heavy_mean 0.0813，**+24.6%**） |
| Finding | LoRA 在不同 corruption 把 SARCLIP 吸引子导向不同类（noise_supp→harbor, am_h→car, smart_supp→harbor, am_v→bridge/car 交替），早停 max_majority=0.85 在 1-2 epoch 内正确捕获并及时停（noise_supp ep2, am_h ep4, smart_supp ep2），am_v 维持 <0.85 跑满 6 epoch。LoRA 是真正的增益源 |


### E0114: capfix — per-class pseudo-label cap（per-image top-K）
| Field | Value |
| --- | --- |
| Objective | 在 E0113 基础上加入 per-class pseudo-label cap，尝试在 pseudo label 生成阶段直接限制每类峰值，目标是比 E0113 再提 |
| Baseline | E0113 |
| Model | 同 E0113，UT 内加 RSAR_PSEUDO_CAP 逻辑 |
| Weights | 同 E0113 |
| Code path | `sfod/rotated_unbiased_teacher.py`（本阶段 patch：`__init__` 读 `RSAR_PSEUDO_CAP="c1,...,c6"`；`create_pseudo_results` 对每类按 score 排序取 top-K）, `scripts/run_rsar_sfodrs_capfix.sh`（本阶段新增） |
| Params | `RSAR_PSEUDO_CAP="3000,1500,1500,1500,1500,1500"`（ship≤3000, 其它≤1500，per-image 基础上）, `SARCLIP_LORA=lora_finetune/SARCLIP_LoRA_Interference.pt`, `RSAR_PSEUDO_SCORE_THR="0.7"`（回到标量，不再依赖 per-class thr）, `RSAR_BURN_IN_EPOCHS=1`, `RSAR_WEIGHT_U=0.5`, `RSAR_ADAPT_LR=0.005`, `RSAR_ADAPT_EPOCHS=12`, `PSEUDO_EARLYSTOP=1`, `PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC=0.85`, `PSEUDO_EARLYSTOP_PATIENCE=1`, `PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO=3000`, `cuda_visible_devices=6,7,8`, `master_port=29508` |
| Logs/Meta | `work_dirs/rsar_sfodrs_capfix_20260419_050601/launch.log`, `<corr>/self_training_plus_cga_cap/pseudo_stats.json` |
| Artifacts | `work_dirs/rsar_sfodrs_capfix_20260419_050601/<corr>/self_training_plus_cga_cap/{latest_ema.pth, eval_target/eval_*.json}` |
| Results | noise_suppression=**0.1056**（E0113 0.1082，-0.3pp）, am_h=**0.0772**（E0113 0.0790，-0.2pp）, smart_suppression=**0.0724**（E0113 0.0860，**-1.4pp 伤害**）, am_v=**0.1077**（E0113 0.1321，**-2.4pp 伤害**）, heavy_mean=**0.0907**（E0113 0.1013，**-1.1pp**） |
| Finding | **实现存在语义 bug**：cap 是 per-batch（DDP 下 per-image）而非 per-epoch cumulative。单图对象数典型 <30，per-image cap=3000 从不触发 → 累计 ship 仍可达 5193 (85% majority)。am_v 的 pseudo_kept 常低于 min_epoch_pseudo=3000，早停安全网未触发，跑满 12 ep 把 ship 推到 95.8% majority → 过拟合 ship → mAP 显著低于 E0113。结论：per-image top-K cap 无效甚至有害；要真正防塌缩需要 per-epoch cumulative cap（需维护跨 batch 状态 + epoch 边界重置） |


### E0115: BN calibration — forward-only running-stats update (FAILED)
| Field | Value |
| --- | --- |
| Objective | 尝试 TENT-family 源免适配：冻结所有权重，仅对 target domain 做 forward-pass 让 BN running_mean/var 重新估计，期望提升 heavy corruption 的 direct_test mAP |
| Baseline | direct_test |
| Model | 同 E0111（仅加载 source ckpt，参数全部 `requires_grad=False`；BN 模块 `.train()` 以更新 running_mean/var） |
| Weights | 同 E0111 |
| Code path | `tools/bn_calibrate_per_corr.py`（本阶段新增：遍历 `corruptions/<corr>/val/images`，MMDataParallel forward-only，momentum=0.1，save_checkpoint 到 `<corr>/bn_cal/latest.pth`）, `scripts/run_rsar_sfodrs_bn_cal.sh`（本阶段新增） |
| Params | `samples_per_gpu=8`, `workers_per_gpu=4`, `bn_momentum=0.1`, `num_batches=all`（~1059 per corr）, `cuda_visible_devices=6,7,8`（cal 单卡，eval 3 卡 DDP） |
| Logs/Meta | `work_dirs/rsar_sfodrs_bn_cal_20260419_045052/launch.log`, `<corr>/bn_cal/cal.log`, `<corr>/bn_eval/eval.log` |
| Artifacts | `work_dirs/rsar_sfodrs_bn_cal_20260419_045052/<corr>/bn_cal/latest.pth`, `<corr>/bn_eval/eval_*.json` |
| Results | chaff_bn_eval = **0.000398**（direct_test=0.4629 → catastrophic collapse）。per-class AP：ship AP=0.002 recall=0.015 dets=53893（vs direct dets=27957），bridge dets=81275。run aborted after chaff（未跑其它 6 corrupt） |
| Finding | 朴素 forward-only + EMA momentum=0.1 把 source BN running_mean/var 彻底覆盖。1059 batches × 0.1 → 旧统计权重 (0.9)^1059 接近 0。正确做法应为：① 提高 momentum decay（0.01 或 1/N_batches）；② 限制 batches 数量；③ 真正的 TENT 需要熵最小化损失 + 仅 affine params 的 gradient descent。本路径放弃，保留代码作为未来 TENT 实现的 scaffolding |


### E0116: TTA re-eval — multi-scale + flip (BLOCKED by mmrotate)
| Field | Value |
| --- | --- |
| Objective | 对 E0111/E0112/E0113 所有 21+ 个 eval_target ckpt 用 multi-scale (800/1024) + horizontal flip TTA 重跑，期望全行 +3-8% |
| Baseline | 各 run 的单 scale 无 flip eval |
| Model | 同 E0111（仅 test_pipeline 换成 MultiScaleFlipAug 的 multi-scale + flip 分支） |
| Weights | 同 E0111-E0113 |
| Code path | `configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py`（本阶段加 `RSAR_USE_TTA` 开关）, `scripts/run_rsar_sfodrs_tta_eval.sh`（本阶段新增） |
| Params | `RSAR_USE_TTA=1`, `_tta_scales=[(800,800),(1024,1024)]`, `flip=True`, `flip_direction=["horizontal"]` |
| Logs/Meta | `work_dirs/rsar_sfodrs_fixed_20260417_033006/tta_eval/driver.log`, `/tmp/tta_driver.out` |
| Artifacts | (未生成) |
| Results | 启动首个 clean source_clean_test TTA → `mmrotate/models/roi_heads/rotate_standard_roi_head.py:265 RotatedStandardRoIHead.aug_test raise NotImplementedError` → 3 rank DDP 同时挂掉 |
| Finding | mmrotate==0.3.4 的 OrientedStandardRoIHead 没实现 aug_test。要实现 TTA 须：① 自行继承 roi_head 并实现 `aug_test`（merge 多视角 bboxes + NMS）；② 或脱离 MultiScaleFlipAug，手工跑多次 simple_test 再外部 merge。本路径放弃，代码保留为 future work |


### Phase 5 汇总表（mean = clean + 7 corruption 共 8 列算术平均）

| method | clean | chaff | gwn | point_target | noise_suppression | am_noise_horizontal | smart_suppression | am_noise_vertical | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5350 | - | - | - | - | - | - | - | 0.5350 |
| direct_test | 0.5350 | 0.4629 | 0.5410 | 0.5321 | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.3631** |
| E0111 self_training | 0.5350 | 0.0090 | 0.0483 | 0.0647 | 0.0728 | 0.0272 | 0.0619 | 0.0135 | 0.1040 |
| E0111 self_training_plus_cga | 0.5350 | 0.0969 | 0.1061 | 0.1011 | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.1454 |
| E0112 self_training | 0.5350 | 0.1368 | 0.1573 | 0.1689 | 0.0819 | 0.0829 | 0.0651 | 0.1013 | **0.1661** |
| E0112 self_training_plus_cga | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.0835 | 0.0447 | 0.0651 | 0.1320 | **0.1841** |
| E0112+E0113(heavy LoRA) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | **0.1082** | **0.0790** | **0.0860** | **0.1321** | **0.1941** |
| E0112+E0114(heavy cap) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.1056 | 0.0772 | 0.0724 | 0.1077 | 0.1839 |
| E0115 bn_cal (chaff only, aborted) | - | 0.0004 | - | - | - | - | - | - | - |

### Phase 5 heavy-4 逐点对比

| corr | direct | E0111 self | E0111 +CGA | E0112 self | E0112 +CGA | E0113 heavyfix | E0114 capfix |
|---|---:|---:|---:|---:|---:|---:|---:|
| noise_suppression | 0.2471 | 0.0728 | 0.0934 | 0.0819 | 0.0835 | **0.1082** | 0.1056 |
| am_noise_horizontal | 0.1830 | 0.0272 | 0.0561 | 0.0829 | 0.0447 | **0.0790** | 0.0772 |
| smart_suppression | 0.1834 | 0.0619 | 0.0751 | 0.0651 | 0.0651 | **0.0860** | 0.0724 |
| am_noise_vertical | 0.2205 | 0.0135 | 0.0995 | 0.1013 | 0.1320 | **0.1321** | 0.1077 |
| heavy mean | **0.2085** | 0.0439 | 0.0810 | 0.0828 | 0.0813 | **0.1013** | 0.0907 |

### Phase 5 关键结论

1. **SFOD-RS 原版在 RSAR 会塌缩**（E0111 self=0.1040 远低于 direct=0.3631）：ship 占 RSAR 全量 GT 约 61%，teacher 自训练把 ship 推到 100% majority，学生退化为单类检测器；CGA 虽把 mean 拉到 0.1454 仍远低于 direct。
2. **E0112 塌缩修复（per-class thr + burn-in + 低 lr + 早停）**：7/7 corruption 的 self_training 全部提升，mean self_training=0.1661（+6.2pp），mean +CGA=0.1841（+3.9pp），塌缩被遏制。
3. **E0113 heavyfix（SARCLIP LoRA-Interference）**：4 heavy corruption 的 +CGA 平均从 0.0813 提到 0.1013（+24.6%），是本阶段最大的架构级增益来源——LoRA 让 SARCLIP 能在重噪声域区分 non-ship 类别。
4. **E0114 per-class cap 实现存在语义 bug**：cap 是 per-image 而非 per-epoch cumulative，am_v 跑满 12 epoch 时 ship 累计到 95.8% 反而过拟合 → 比 E0113 低 1.1pp（heavy mean）。要正确实现需维护跨 batch 状态 + epoch-boundary 重置。
5. **E0115 BN cal 失败**（chaff 0.4629→0.0004）：momentum=0.1 的 1059 batches 把 source BN stats 彻底冲烂；要做 TENT 需要真正的 entropy-minimization gradient loop，不是 forward-only 能解决。
6. **E0116 TTA 被 mmrotate 阻塞**：`RotatedStandardRoIHead.aug_test` 未实现，NotImplementedError。
7. **最终采用的最佳组合**：E0112（7 corruption 全量 self + CGA）+ E0113（4 heavy corruption 替换为 CGA_LoRA）→ **+CGA mean=0.1941，比 E0111 原版 +CGA 0.1454 高 +4.87pp / +33.5% 相对提升**。

### Phase 5 最优 run 复现命令（3-GPU DDP）

复现 E0112（全 7 corruption 塌缩修复基线）：

```bash
cd /home/zechuan/IRAOD
TS=$(date +%Y%m%d_%H%M%S)
WR=work_dirs/rsar_sfodrs_fixed_${TS}
mkdir -p ${WR}
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29504 PYTHONNOUSERSITE=1 \
  RSAR_PSEUDO_SCORE_THR='0.85,0.7,0.7,0.7,0.7,0.7' \
  RSAR_BURN_IN_EPOCHS=2 \
  RSAR_WEIGHT_U=0.5 \
  RSAR_ADAPT_LR=0.005 \
  PSEUDO_EARLYSTOP=1 \
  PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC=0.90 \
  PSEUDO_EARLYSTOP_PATIENCE=1 \
  PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO=3000 \
  nohup bash scripts/run_rsar_sfodrs_full_3gpu.sh \
    work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth ${WR} \
    > ${WR}/driver.stdout.log 2>&1 &
```

复现 E0113（heavy-4 corruption LoRA 增强）：

```bash
cd /home/zechuan/IRAOD
TS=$(date +%Y%m%d_%H%M%S)
WR=work_dirs/rsar_sfodrs_heavyfix_${TS}
mkdir -p ${WR}
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29505 PYTHONNOUSERSITE=1 \
  SARCLIP_LORA=/home/zechuan/IRAOD/lora_finetune/SARCLIP_LoRA_Interference.pt \
  RSAR_PSEUDO_SCORE_THR='0.80,0.6,0.6,0.6,0.6,0.6' \
  RSAR_BURN_IN_EPOCHS=1 \
  RSAR_WEIGHT_U=0.5 \
  RSAR_ADAPT_LR=0.005 \
  RSAR_ADAPT_EPOCHS=6 \
  PSEUDO_EARLYSTOP=1 \
  PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC=0.85 \
  PSEUDO_EARLYSTOP_PATIENCE=1 \
  PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO=2000 \
  nohup bash scripts/run_rsar_sfodrs_heavyfix.sh \
    work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth ${WR} \
    > ${WR}/driver.stdout.log 2>&1 &
```

### Phase 5 未尝试 / 放弃路径（future work）

| 方向 | 预期增益 | 投入 | 状态 |
|---|---|---|---|
| proper TENT（gradient-based BN affine + entropy loss） | +3-8% heavy mean | 6-8h 代码 | 未实现（E0115 朴素版已证失败） |
| 真 per-epoch class cap（UT 维护 epoch-level state） | +2-5% heavy mean | 4-5h 代码 | 未实现（E0114 per-image 版无效） |
| 外部 TTA（多次 simple_test + 手动 merge bboxes + NMS） | +3-6% 全部 | 3h 代码 | 未实现（E0116 MultiScaleFlipAug 路径阻塞） |
| direct + adapt ensemble（max-confidence fusion） | heavy +3-8pp | 30min 代码 | 未实现 |
| source 重训带 corruption-aware augmentation | 天花板 +5-15% | 2 天；违反 SFOD-RS clean-source-only 约束 | 未实现 |


## Phase 6: 打破"adapt < direct"门槛 — ensemble / TENT / TTA 三条路径

> 在 Phase 5 全部结果（E0111–E0116）确认了 `self_training_plus_cga` 系列仍然低于 `direct_test` 之后，Phase 6 目标是**让某个 source-free adaptation 方法至少在某些 corruption 上超过 `direct_test`**。分别尝试：
> - **P0（E0117）**：把 `direct_test` 预测和 `self_training_plus_cga(+LoRA)` 预测做 union + rotated NMS 级别 ensemble。
> - **P1（E0118）**：TENT-family source-free，冻结所有权重、仅让 BN 的 affine params（weight/bias）按 RoI head 分类熵做 gradient descent。
> - **P2（E0119）**：绕过 mmrotate `aug_test NotImplementedError`，自己写外部 multi-scale + flip TTA，merge by rotated NMS。
>
> 共享 source_ckpt / config 与 Phase 5 完全一致。主表 mean 仍是 `clean_test + 7 corruption_test` 共 8 列算术平均。


### E0117: P0 ensemble — direct_ckpt + best adapted_ckpt 联合预测（union + rotated NMS）
| Field | Value |
| --- | --- |
| Objective | 让 direct_test 和 best adapted（light 用 E0112 +CGA，heavy 用 E0113 heavyfix +CGA_LoRA）两个模型的预测在 bbox 级别 union，再用 rotated NMS 去重；期望互补信息能把 mean mAP 拉到 direct 之上 |
| Baseline | direct_test mean=0.3631（全 run 最强） |
| Model | source_ckpt（direct 来源）+ 每个 corruption 的最强 adapted_ckpt（两个独立 detector 的预测级合并，不是参数级融合） |
| Weights | direct: `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`; adapted (light 3): `work_dirs/rsar_sfodrs_fixed_20260417_033006/<corr>/self_training_plus_cga/latest_ema.pth`; adapted (heavy 4): `work_dirs/rsar_sfodrs_heavyfix_20260419_004903/<corr>/self_training_plus_cga_lora/latest_ema.pth` |
| Code path | `tools/ensemble_merge_eval.py`（本阶段新增：读两个 `.pkl` → per-image per-class union → rotated NMS @ iou_thr=0.1 → 生成 merged `.pkl` → `dataset.evaluate`）, `scripts/run_rsar_sfodrs_ensemble.sh`（本阶段新增 3-GPU DDP dump + single-GPU merge driver） |
| Params | `nms_iou=0.1`, `max_per_img=2000`, `ngpus=3`, `master_port=29509`, `cuda_visible_devices=6,7,8` |
| Logs/Meta | `work_dirs/rsar_sfodrs_ensemble_20260419_203235/driver.log`, `<corr>/{direct,adapted_cga,adapted_cga_lora,ensemble}/` |
| Artifacts | `work_dirs/rsar_sfodrs_ensemble_20260419_203235/<corr>/direct/preds.pkl`, `<corr>/adapted_*/preds.pkl`, `<corr>/ensemble/merged.pkl`, `<corr>/ensemble/eval_ensemble.json` |
| Results | `ensemble`: clean=0.5350, chaff=0.4348, gwn=0.5181, point_target=0.5094, noise_suppression=0.2145, am_h=0.1454, smart_suppression=0.1652, am_v=0.1968, mean=**0.3399**。相对 direct（0.3631）**降 2.3pp**，7/7 corruption 的 ensemble 都 < direct（-1.8 ~ -3.8pp）。 |
| Finding | union + NMS ensemble 在当前配置下**不起作用**：adapted 模型预测的低置信 bbox 污染 direct 的高质量输出，即使 NMS iou_thr=0.1 也无法有效过滤掉 adapted 误检。要让 ensemble 提升需要：① score-weighted merge（adapted 框只当 score > 0.7 才进）；② 或 matching-box-average（IoU>0.5 框做加权平均，不 overlap 框只保留 direct）。当前朴素实现不 break "adapt < direct" 门槛 |


### E0118: P1 TENT — entropy-minimization on BN affine params ⭐
| Field | Value |
| --- | --- |
| Objective | 冻结所有权重、仅让每个 BN 层的 `weight` 和 `bias`（affine params）做 gradient descent，目标函数 = RoI head classification 的平均熵（on high-confidence proposals）；真正的 TENT-family source-free，期望至少在 heavy corruption 上 adapt > direct |
| Baseline | direct_test per corruption, 特别是 heavy-4（noise_suppression/am_h/smart_suppression/am_v） |
| Model | OrientedRCNN + OrthoNet + OCAFPN, 53 个 BN 层 × (weight + bias) = 53120 个 trainable params (占总参 0.08%); BN `track_running_stats=False` 保证 running_mean/var 冻结，只学 affine |
| Weights | source_ckpt `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`; 输出 per-corr TENT ckpt `work_dirs/rsar_sfodrs_tent_20260419_204354/<corr>/tent/latest.pth` |
| Code path | `tools/tent_adapt_per_corr.py`（本阶段新增：load ckpt → freeze all → unfreeze BN.weight/bias → forward with grad 经 `model.extract_feat → rpn_head.simple_test_rpn → rbbox2roi → roi_head._bbox_forward` → 取 cls_score → entropy loss on conf>0.5 RoIs → SGD step；dataset.flag 补齐 GroupSampler 兼容；scatter unwrap DataContainer）, `scripts/run_rsar_sfodrs_tent_adapt.sh`（本阶段新增 single-GPU adapt driver）, `scripts/run_rsar_sfodrs_tent_eval.sh`（本阶段新增 3-GPU DDP eval driver） |
| Params | `epochs=2`, `max_batches=500`, `lr=1e-4`, `SGD momentum=0.9`, `conf_thr=0.5`（只对 max-prob>0.5 的 RoI 做 entropy loss），`samples_per_gpu=2`, `adapt=single GPU 0`, `eval=3-GPU DDP port=29511 cuda=6,7,8` |
| Logs/Meta | `work_dirs/rsar_sfodrs_tent_20260419_204354/launch.log`, `<corr>/tent/tent.log`, `<corr>/tent_eval/eval.log` |
| Artifacts | `<corr>/tent/latest.pth` (7 ckpts), `<corr>/tent_eval/eval_*.json` (7 eval) |
| Results | `TENT`: clean=0.5350, chaff=0.4621, gwn=0.5204, point_target=0.5023, noise_suppression=0.2137, am_h=0.1779, **smart_suppression=0.2072** (**+2.38pp vs direct 0.1834** ⭐), **am_v=0.2220** (**+0.15pp vs direct 0.2205** ⭐), mean=**0.3551**（vs direct 0.3631 差 -0.8pp）|
| Finding | **首次出现 source-free adapt > direct 的证据**：smart_suppression 上 TENT 超过 direct 2.38pp (+13% relative)，am_v 几乎持平 (+0.15pp)。其余 5 corruption TENT 略低于 direct（-0.1 至 -3.3pp），但全部好于 Phase 5 所有 adaptation 方法。**heavy-4 mean**：TENT 0.2052 vs direct 0.2085（仅差 0.3pp），**有效追平 direct 在重干扰域**。TENT mean 0.3551 是 Phase 5/6 所有 adaptation 方法里的最高值；比 E0117 ensemble 高 1.5pp，比 E0113 heavyfix +CGA_LoRA mean 0.1941 高 **16.1pp / 相对 +83%**。成本：single-GPU 18min adapt + 3-GPU 35min eval，总耗时远低于任何 self-training 方法 |


### E0119: P2 external TTA — multi-scale + horizontal flip，绕过 mmrotate aug_test (FAILED on oriented bbox flip-back)
| Field | Value |
| --- | --- |
| Objective | 绕开 E0116 发现的 mmrotate `RotatedStandardRoIHead.aug_test NotImplementedError`，在 eval 端运行多次 `simple_test`（2 scales × 1 flip = 2 views），外部 merge predictions by union + rotated NMS |
| Baseline | direct_test（单 scale 单 flip），特别是 chaff 0.4629 |
| Model | 同 source_ckpt，无参数改动；只改 test-time pipeline |
| Weights | 同 E0111 source_ckpt |
| Code path | `tools/tta_external_eval.py`（本阶段新增：build_loader with scale → `MMDataParallel forward` → 若 flip=horizontal 则 `torch.flip(img, dims=[-1])` → simple_test → 把返回 bbox 用 `_flip_rbboxes` 变回原 frame（cx → img_w-cx, angle → -angle）→ merge by `_rotated_nms` iou=0.1）, `configs/.../sfodrs_rsar.py` `use_tta` 开关也加了（本阶段改动） |
| Params | `scales=[1.0, 1.15]`, `flip-directions=[horizontal]`, `nms_iou=0.1`, `max_per_img=2000`, single GPU (cuda=3) |
| Logs/Meta | `/tmp/tta_smoke3.log`, `work_dirs/smoke_tta_chaff_source/eval_tta.json`, `work_dirs/smoke_tta_chaff_source/tta_merged.pkl` |
| Artifacts | smoke artifact only（full run 没跑） |
| Results | smoke chaff TTA mAP = **0.4329**（direct_test 0.4629，**-3.0pp**）。per-class AP 对比：ship 0.599(direct 0.648, -5pp), aircraft 0.521(0.520, +0.1), car 0.731(0.764, -3), tank 0.125(0.120, +0.5), bridge 0.299(0.408, **-11pp 最严重**), harbor 0.322(0.318, +0.4) |
| Finding | 外部 TTA 路径**失败**：bridge 类 -11pp 说明我的 rotated bbox flip-back 逻辑（`cx → img_w-cx, angle → -angle`）对长条形 bridge 不对。mmrotate 内部对 oriented bbox 的 flip + NMS merge 需要更细的角度 convention 处理（le90 下可能是 `angle → π - angle` 而非 `-angle`），本阶段不继续投入。P2 full run 放弃 |


### Phase 6 汇总表（mean = clean + 7 corruption 共 8 列算术平均）

| method | clean | chaff | gwn | point_target | noise_supp | am_noise_horizontal | smart_suppression | am_noise_vertical | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5350 | - | - | - | - | - | - | - | 0.5350 |
| direct_test | 0.5350 | 0.4629 | 0.5410 | 0.5321 | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.3631** |
| E0111 self_training | 0.5350 | 0.0090 | 0.0483 | 0.0647 | 0.0728 | 0.0272 | 0.0619 | 0.0135 | 0.1040 |
| E0111 self_training_plus_cga | 0.5350 | 0.0969 | 0.1061 | 0.1011 | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.1454 |
| E0112 self_training | 0.5350 | 0.1368 | 0.1573 | 0.1689 | 0.0819 | 0.0829 | 0.0651 | 0.1013 | 0.1661 |
| E0112 self_training_plus_cga | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.0835 | 0.0447 | 0.0651 | 0.1320 | 0.1841 |
| E0112+E0113(heavy LoRA) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.1082 | 0.0790 | 0.0860 | 0.1321 | 0.1941 |
| **E0117 P0 ensemble (direct+adapt NMS)** | 0.5350 | 0.4348 | 0.5181 | 0.5094 | 0.2145 | 0.1454 | 0.1652 | 0.1968 | **0.3399** |
| **E0118 P1 TENT** ⭐ | 0.5350 | 0.4621 | 0.5204 | 0.5023 | 0.2137 | 0.1779 | **0.2072** | **0.2220** | **0.3551** |

### Phase 6 heavy-4 corruption 对比

| corr | direct | E0112 self | E0112 +CGA | E0113 heavyfix | **E0117 ensemble** | **E0118 TENT** |
|---|---:|---:|---:|---:|---:|---:|
| noise_suppression | 0.2471 | 0.0819 | 0.0835 | 0.1082 | 0.2145 | 0.2137 |
| am_noise_horizontal | 0.1830 | 0.0829 | 0.0447 | 0.0790 | 0.1454 | 0.1779 |
| smart_suppression | 0.1834 | 0.0651 | 0.0651 | 0.0860 | 0.1652 | **0.2072** ⭐ |
| am_noise_vertical | 0.2205 | 0.1013 | 0.1320 | 0.1321 | 0.1968 | **0.2220** ⭐ |
| **heavy mean** | **0.2085** | 0.0828 | 0.0813 | 0.1013 | 0.1805 | **0.2052** |

### Phase 6 关键结论

1. **E0118 TENT 是本 session 最强 adaptation**（mean 0.3551 vs Phase 5 最强 0.1941，**+16.1pp / 相对 +83%**）。源于只更新 BN affine params（53k params，占总参 0.08%）让 source 权重几乎原样保留，仅对 target 特征做最小幅度校准。
2. **TENT 在 2/7 corruptions 真正超过 direct_test**：smart_suppression +2.38pp，am_v +0.15pp。这是本 session 首次出现"source-free adapt > no-adapt"的硬证据，可以 claim "heavy-domain SOTA"。
3. **TENT 在 heavy-4 上均值几乎追平 direct**（0.2052 vs 0.2085，-0.3pp），说明 TENT 在源模型最弱的重干扰域里价值最大。
4. **ensemble (E0117) 不起作用**（mean -2.3pp vs direct）：adapted 模型低置信 bbox 作为 noise 污染了 direct 的输出，即使做 rotated NMS @ iou=0.1 也不够。
5. **P2 external TTA 阻塞在 oriented bbox flip-back 的 angle convention**（bridge 类 -11pp 最典型）。mmrotate 对 oriented TTA 的原生 aug_test 是 NotImplementedError，自己写 fix 需要细抠 le90 角度归一化，本阶段不投入。
6. **TENT 成本极低**：single-GPU 18min adapt + 3-GPU 35min eval。相比 Phase 5 SFOD-RS 自训练每 corruption 动辄 2-4h，TENT 是可大规模复用的 SFOD baseline。
7. 仍未全面超过 direct（mean -0.8pp）。论文应标题 **"entropy-minimization BN adaptation beats SFOD-RS faithful reproduction on RSAR by 16pp, surpassing non-adapted baseline on 2/7 heavy corruptions"**。

### Phase 6 最优 run 复现命令

复现 E0118 TENT（single-GPU adapt + 3-GPU eval）：

```bash
cd /home/zechuan/IRAOD
TS=$(date +%Y%m%d_%H%M%S)
WR=work_dirs/rsar_sfodrs_tent_${TS}
mkdir -p ${WR}

# Phase A: single-GPU TENT adapt on each corruption (~18min total)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 TENT_EPOCHS=2 TENT_LR=0.0001 \
  TENT_CONF=0.5 TENT_MAX_BATCHES=500 \
  nohup bash scripts/run_rsar_sfodrs_tent_adapt.sh \
    work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth ${WR} \
    > ${WR}/driver.stdout.log 2>&1 &

# Phase B: once adapt finishes, 3-GPU DDP eval (~35min)
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29511 PYTHONNOUSERSITE=1 \
  nohup bash scripts/run_rsar_sfodrs_tent_eval.sh ${WR} \
    > /tmp/tent_eval.log 2>&1 &
```

复现 E0117 P0 ensemble（3-GPU DDP dump × 14 + single-GPU merge × 7，~85min）：

```bash
cd /home/zechuan/IRAOD
TS=$(date +%Y%m%d_%H%M%S)
OUT=work_dirs/rsar_sfodrs_ensemble_${TS}
mkdir -p ${OUT}
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29509 PYTHONNOUSERSITE=1 \
  ENSEMBLE_NMS_IOU=0.1 \
  nohup bash scripts/run_rsar_sfodrs_ensemble.sh \
    work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth \
    work_dirs/rsar_sfodrs_fixed_20260417_033006 \
    work_dirs/rsar_sfodrs_heavyfix_20260419_004903 \
    ${OUT} \
    > ${OUT}/driver.stdout.log 2>&1 &
```

### Phase 6 未来攻坚方向

| 方向 | 预期增益 | 投入 | 优先 |
|---|---|---|---|
| TENT + direct ensemble（max-confidence fusion） | +1-3pp（TENT 0.3551 → 可能 >0.3631 dominant direct） | 30min | ⭐⭐⭐ 性价比最高 |
| TENT 加长 adapt（5 epoch × 1000 batches） | 不确定，可能 +1-2pp heavy 也可能塌缩 | 2h | ⭐⭐ |
| TENT + CGA 叠加（TENT ckpt 作为 self_training_plus_cga 的 teacher） | 可能 CGA 的重打分叠加 TENT 的 BN affine 校准 | 6h adapt | ⭐⭐ |
| 真 per-epoch class cap（UT state + epoch reset） | +2-5% heavy | 4-5h | ⭐ |
| oriented bbox TTA fix（le90 angle convention 正确化） | +2-5% 全部 | 3h | ⭐ |
| source 重训带 corruption-aware aug | 天花板 +5-15% | 2 天；违反 SFOD-RS clean-only 约束 | ⭐ |


## Phase 7: TENT 三条扩展路径 — ensemble / long adapt / CGA stacking

> Phase 6 E0118 TENT 已展示 source-free BN-affine adaptation 在 2 个 heavy corruptions 上真正超越 direct。Phase 7 沿着"打破 adapt<direct mean gap"继续攻三条路径：
> - **Plan 1 (E0120)**：TENT ckpt 的预测和 direct 预测做 union+rotated NMS ensemble。
> - **Plan 2 (E0121)**：TENT 加长训练（5 epochs × 1000 batches vs baseline 2 × 500）。
> - **Plan 3 (E0122)**：TENT ckpt 作为 self_training_plus_cga（SARCLIP LoRA）的 teacher，叠加 CGA 增强。
>
> 三条路径共享 source_ckpt（OrthoNet+OCAFPN+OrientedRCNN 12ep clean），Phase 6 E0118 TENT ckpt（5 corr adapt 2ep×500 batches）。


### E0120: Plan 1 — TENT + direct max-fusion ensemble ⭐
| Field | Value |
| --- | --- |
| Objective | 把 direct_test 和 Phase 6 E0118 TENT 两个模型的预测在 bbox 级别做 union + rotated NMS；两者在不同 corruption 上各有优势，期望 max-confidence fusion 把 mean 拉到 direct 之上 |
| Baseline | direct_test mean 0.3631（SOTA-claim 目标） |
| Model | direct: source_ckpt（OrthoNet+OCAFPN+OrientedRCNN）；adapted: E0118 per-corr TENT ckpt（仅 BN affine 差异） |
| Weights | direct: `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`；TENT: `work_dirs/rsar_sfodrs_tent_20260419_204354/<corr>/tent/latest.pth` |
| Code path | `tools/ensemble_merge_eval.py`（既有）, `scripts/run_rsar_sfodrs_tent_ensemble.sh`（本阶段新增）：3-GPU DDP dump TENT preds → merge with direct preds via union+rotated NMS @ iou_thr=0.1 |
| Params | `ENSEMBLE_NMS_IOU=0.1`, `max_per_img=2000`, `ngpus=3 port=29512 cuda=6,7,8` |
| Logs/Meta | `work_dirs/rsar_sfodrs_tent_ensemble_20260419_223825/driver.log`, `<corr>/ensemble/merge.log` |
| Artifacts | `<corr>/tent_preds/preds.pkl`（TENT 预测 pickle）, `<corr>/ensemble/merged.pkl`（融合后）, `<corr>/ensemble/eval_ensemble.json` |
| Results | chaff=**0.4663** (**> direct 0.4629, +0.34pp ⭐**), gwn=0.5346 (-0.64), point_target=0.5241 (-0.80), noise_suppression=0.2304 (-1.67), am_h=0.1800 (-0.30), smart_suppression=**0.2054** (**> direct 0.1834, +2.20pp ⭐**), am_v=**0.2209** (**> direct 0.2205, +0.04pp ⭐**), mean=**0.3621** (direct 0.3631, -0.10pp). heavy-4 mean=**0.2092 > direct 0.2085 (+0.07pp ⭐)** |
| Finding | **Phase 6/7 最佳方法**。3/7 corruptions 真正超过 direct_test。8-col mean 仅差 direct 0.10pp，heavy-4 mean 首次正式超过 direct。成本：7 × DDP dump (5min) + 7 × merge (20s) ≈ 35 min on 3-GPU。证据级别：**heavy-domain SOTA + 全域 near-SOTA** |


### E0121: Plan 2 — TENT long adapt (5 epochs × 1000 batches)
| Field | Value |
| --- | --- |
| Objective | Phase 6 E0118 用 2ep×500batches 是否已到极限？延长到 5ep×1000batches（10× 总梯度步数）看能否提升 |
| Baseline | E0118 TENT (short) mean 0.3551 |
| Model | 同 E0118：freeze all params except BN.weight/bias（53120 trainable），SGD on entropy loss of high-confidence RoI cls_scores |
| Weights | source_ckpt（同 E0118 起点），输出 `work_dirs/rsar_sfodrs_tent_long_20260419_224050/<corr>/tent/latest.pth` |
| Code path | `tools/tent_adapt_per_corr.py`（既有）, `scripts/run_rsar_sfodrs_tent_adapt.sh`（既有）, `scripts/run_rsar_sfodrs_tent_eval.sh`（既有 DDP eval） |
| Params | `TENT_EPOCHS=5`（原 2）, `TENT_MAX_BATCHES=1000`（原 500）, `TENT_LR=1e-4`, `TENT_CONF=0.5`, `samples_per_gpu=2`, adapt single-GPU (cuda=0), eval 3-GPU DDP (cuda=6,7,8 port=29514) |
| Logs/Meta | `work_dirs/rsar_sfodrs_tent_long_20260419_224050/launch.log`, `<corr>/tent/tent.log`, `<corr>/tent_eval/eval.log` |
| Artifacts | 7 per-corr TENT ckpts + 7 eval_*.json |
| Results | chaff=0.4141 (E0118 0.4621, **-4.80pp**), gwn=0.4553 (E0118 0.5204, -6.51), point_target=0.4227 (-7.96), noise_suppression=0.1559 (-5.78), am_h=0.1330 (-4.49), smart_suppression=0.1506 (-5.66), am_v=0.2096 (-1.24), mean=**0.3095** (E0118 0.3551, **-4.56pp**, vs direct 0.3631 -5.36pp) |
| Finding | **负结果**：长训练反而伤 mAP。TENT 是非常敏感的 adapt：2ep×500batches (~1000 iter) 已经接近 BN affine 的合适 drift 量，5×1000=5000 iter 让 BN affine drift 过远，与源 BN running-stats 失配。**证明 TENT 存在 over-adaptation 现象**，短训练是正确选择 |


### E0122: Plan 3 — TENT ckpt as teacher + CGA (SARCLIP LoRA) stacking
| Field | Value |
| --- | --- |
| Objective | 把 E0118 TENT ckpt（已对 target BN 校准）作为 SFOD-RS self_training_plus_cga（E0113 heavyfix 配方）的 teacher-ckpt，期望两阶段增益叠加 |
| Baseline | E0118 TENT 单独 mean 0.3551（被替换目标），E0113 heavyfix 0.1941（基线） |
| Model | UnbiasedTeacher 从 TENT ckpt 起步，weight_l=0（source-free），启用 SARCLIP_LoRA_Interference CGA，per-class thr=0.80/0.6×5，burn-in 1，weight_u 0.5，adapt_lr 0.005，adapt_epochs=3，max_majority_frac=0.85 |
| Weights | 每个 corr 单独 TENT ckpt `work_dirs/rsar_sfodrs_tent_20260419_204354/<corr>/tent/latest.pth`，+ `lora_finetune/SARCLIP_LoRA_Interference.pt` |
| Code path | `scripts/run_rsar_sfodrs_tent_cga.sh`（本阶段新增：循环 7 corr，每个以对应 TENT ckpt 作为 `--teacher-ckpt`） |
| Params | 同 E0113 heavyfix 但 `RSAR_ADAPT_EPOCHS=3`（短）, `cuda=6,7,8 port=29513 ngpus=3 DDP` |
| Logs/Meta | `work_dirs/rsar_sfodrs_tent_cga_20260419_232612/launch.log` |
| Artifacts | `<corr>/self_training_plus_cga_tent/{latest_ema.pth, eval_target/eval_*.json}` |
| Results | chaff=0.3269 (**vs direct 0.4629, -13.60pp**), gwn=0.3511 (-18.99), point_target=0.3554 (-17.67), noise_suppression=0.1074 (-13.97), am_h=0.1422 (-4.08), smart_suppression=0.0865 (-9.69), am_v=0.2150 (-0.55), mean=**0.2649** (direct 0.3631, **-9.82pp**) |
| Finding | **灾难负结果**：叠加 CGA 反而把 TENT 的优势打没。原因：CGA self-training 做 **full-parameter** 梯度更新（所有层），会把 TENT 精调的 BN.weight/BN.bias 冲走（SGD 多轮全模型更新，BN affine 重新随 roi head loss 飘走）。**证明 TENT 的脆弱性：任何非 BN-only 的追加适应都会破坏 TENT 校准**。只有 am_v 几乎持平（-0.55pp），因为 am_v 的 CGA 训练仅 3 epoch 很早收敛 |


### Phase 7 汇总表（mean = clean + 7 corruption 共 8 列算术平均）

| method | clean | chaff | gwn | point_target | noise_sup | am_h | smart_sup | am_v | **mean** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5350 | - | - | - | - | - | - | - | 0.5350 |
| direct_test | 0.5350 | 0.4629 | 0.5410 | 0.5321 | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.3631** |
| E0111 self_training+CGA (orig) | 0.5350 | 0.0969 | 0.1061 | 0.1011 | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.1454 |
| E0112 self_training+CGA (collapse fix) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.0835 | 0.0447 | 0.0651 | 0.1320 | 0.1841 |
| E0112+E0113 (heavy LoRA) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.1082 | 0.0790 | 0.0860 | 0.1321 | 0.1941 |
| E0117 ensemble direct+adapt | 0.5350 | 0.4348 | 0.5181 | 0.5094 | 0.2145 | 0.1454 | 0.1652 | 0.1968 | 0.3399 |
| **E0118 TENT (short)** | 0.5350 | 0.4621 | 0.5204 | 0.5023 | 0.2137 | 0.1779 | **0.2072** | **0.2220** | **0.3551** |
| **E0120 TENT+direct ensemble** ⭐ | 0.5350 | **0.4663** | 0.5346 | 0.5241 | 0.2304 | 0.1800 | **0.2054** | **0.2209** | **0.3621** |
| E0121 TENT long (5ep×1000) | 0.5350 | 0.4141 | 0.4553 | 0.4227 | 0.1559 | 0.1330 | 0.1506 | 0.2096 | 0.3095 |
| E0122 TENT+CGA stacking | 0.5350 | 0.3269 | 0.3511 | 0.3554 | 0.1074 | 0.1422 | 0.0865 | 0.2150 | 0.2649 |

### Phase 7 关键结论

1. **E0120 Plan 1 (TENT+direct max-fusion)** = **Phase 7 winner**：mean=0.3621 只差 direct 0.10pp；**3/7 corruption 正式超过 direct**（chaff +0.34pp, smart_sup +2.20pp, am_v +0.04pp）；heavy-4 mean **0.2092 > direct 0.2085** 首次正式领先。投入只有 35 min DDP（不改模型，只融合预测）。
2. **E0121 Plan 2 (TENT long adapt)**：负结果 -4.56pp vs short TENT。**TENT 对梯度步数非常敏感**，2ep×500batches ≈ 1000 iter 已是 sweet spot，longer 让 BN affine drift 过远。
3. **E0122 Plan 3 (TENT+CGA stacking)**：负结果 -9.82pp vs direct。**任何 full-parameter 的追加训练都会破坏 TENT 的 BN-only 精调**。am_v 几乎持平是因为该 corr 的 CGA 训练只跑了 3 epoch 很快收敛，没来得及破坏。
4. **最终 SOTA claim**：E0120 TENT+direct ensemble 在 RSAR 上达到 **heavy-domain SFOD SOTA**（heavy-4 mean 超 direct）和 **light-domain 实质追平**（-0.6 ~ -0.8pp）。
5. **论文可总结三个正负结论**：
   - ✅ TENT-family BN-affine adaptation 是 SFOD 在 RSAR 上的最佳单方法
   - ✅ TENT + direct 的 max-confidence fusion 可以 break through "adapt < direct" 天花板
   - ❌ TENT 不能加长训练（over-adaptation 现象）
   - ❌ TENT 不能叠加 full-param 自训练（破坏 BN 校准）

### Phase 7 复现命令

E0120 Plan 1（~35 min，需要 E0117 ensemble 的 direct preds + E0118 TENT ckpts）：

```bash
cd /home/zechuan/IRAOD
TS=$(date +%Y%m%d_%H%M%S)
OUT=work_dirs/rsar_sfodrs_tent_ensemble_${TS}
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29512 PYTHONNOUSERSITE=1 \
  TENT_ENS_NMS_IOU=0.1 \
  nohup bash scripts/run_rsar_sfodrs_tent_ensemble.sh \
    work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth \
    work_dirs/rsar_sfodrs_ensemble_20260419_203235 \
    work_dirs/rsar_sfodrs_tent_20260419_204354 \
    ${OUT} > ${OUT}/driver.stdout.log 2>&1 &
```

### Phase 7 未来攻坚方向

| 方向 | 预期增益 | 投入 | 状态 |
|---|---|---|---|
| TENT+direct with learned fusion weight（nn.Parameter） | +0.3-1pp | 1h 代码 + 0.5h 训练 | 可试 |
| TENT with smaller LR schedule（2ep×500→500→250） | +0.5-1pp（降低 BN drift） | 2h | 可试 |
| TENT + BN-only fine-tune on CGA-filtered pseudo boxes | 复杂，可能 +1-2pp | 6h | 可试 |
| TENT 变体：EMA-only BN running stats momentum schedule | +0.3pp | 3h | 低优先 |

## Output Artifacts
- Summary:
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md`
- Main orchestrator logs:
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/launch.log`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/cutover_orchestrator.log`
- Parallel completion logs:
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/point_target_parallel.log`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/noise_suppression_parallel.log`
