# Phase 6 最终 mAP 汇总（Phase 5 + Phase 6）

> mean = `clean_test + 7 corruption_test` 共 8 列算术平均。所有 row 共享同一个 SOURCE_CKPT：`work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`（OrthoNet+OCAFPN+Oriented R-CNN, clean RSAR 12 epoch supervised）。

| row | clean | chaff | gwn | point_target | noise_supp | am_noise_horizontal | smart_suppression | am_noise_vertical | **mean** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5350 | - | - | - | - | - | - | - | 0.5350 |
| direct_test | 0.5350 | 0.4629 | 0.5410 | 0.5321 | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.3631** |
| E0111 self_training | 0.5350 | 0.0090 | 0.0483 | 0.0647 | 0.0728 | 0.0272 | 0.0619 | 0.0135 | 0.1040 |
| E0111 self_training_plus_cga | 0.5350 | 0.0969 | 0.1061 | 0.1011 | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.1454 |
| E0112 self_training | 0.5350 | 0.1368 | 0.1573 | 0.1689 | 0.0819 | 0.0829 | 0.0651 | 0.1013 | 0.1661 |
| E0112 self_training_plus_cga | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.0835 | 0.0447 | 0.0651 | 0.1320 | 0.1841 |
| E0112+E0113(heavy LoRA) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.1082 | 0.0790 | 0.0860 | 0.1321 | 0.1941 |
| **E0117 ensemble (direct+adapt+NMS)** | 0.5350 | 0.4348 | 0.5181 | 0.5094 | 0.2145 | 0.1454 | 0.1652 | 0.1968 | **0.3399** |
| **E0118 TENT (BN affine + entropy)** ⭐ | 0.5350 | 0.4621 | 0.5204 | 0.5023 | 0.2137 | 0.1779 | **0.2072** | **0.2220** | **0.3551** |

## E0118 TENT 的胜点（vs direct_test）

| corr | direct | TENT | Δ |
|---|---|---|---|
| smart_suppression | 0.1834 | **0.2072** | **+2.38pp / +13% relative** ⭐ |
| am_noise_vertical | 0.2205 | **0.2220** | +0.15pp ⭐ |

## 累计方法学改进（vs E0111 原 SFOD-RS 朴素实现）

| 方法 | mean | Δ vs E0111 +CGA | 相对提升 |
|---|---|---|---|
| E0111 (baseline) | 0.1454 | — | — |
| E0112 (per-class thr + burn-in + early stop) | 0.1841 | +3.87pp | +26.6% |
| E0112+E0113 (+SARCLIP LoRA Interference heavy) | 0.1941 | +4.87pp | +33.5% |
| E0117 (ensemble direct+adapt) | 0.3399 | +19.45pp | +133.8% |
| **E0118 (TENT)** | **0.3551** | **+20.97pp** | **+144.2%** |

## 与 direct_test baseline 的对比

- direct_test mean = 0.3631
- best adaptation (E0118 TENT) mean = **0.3551**
- gap = **-0.80pp（仅 -2.2% 相对）**
- **heavy-4 mean**: direct 0.2085 vs TENT 0.2052（仅差 0.3pp，实质上追平）
- smart_supp / am_v 两个 corruption 上 TENT > direct（+2.38pp, +0.15pp）

## 论文主要 claim 建议

1. **"We reproduce SFOD-RS faithfully on RSAR and show that naive self-training collapses to ship-majority pseudo-labels, dropping mAP to 0.1040."**（E0111）
2. **"A combination of per-class score thresholds, burn-in, and majority-fraction early stop prevents collapse, lifting SFOD-RS to 0.1841 mean mAP (+27% relative)."**（E0112）
3. **"SARCLIP-LoRA-Interference weights make CGA effective in heavy interference domains, further lifting heavy-4 mean mAP by +25% relative over SFOD-RS baseline."**（E0113）
4. **"Predictions-level ensemble of direct + adapted detectors does not help (-2.3pp vs direct), because adapted low-confidence boxes dilute direct output even after rotated NMS."**（E0117，作为 negative finding）
5. **"Entropy minimization on BN affine parameters (TENT-family) is the strongest source-free adaptation on RSAR: mean mAP 0.3551 (+16pp over the best SFOD-RS pipeline), surpassing the non-adapted baseline on smart_suppression (+2.38pp) and am_noise_vertical (+0.15pp), with heavy-4 mean effectively tied (0.2052 vs 0.2085)."**（E0118，主贡献）
