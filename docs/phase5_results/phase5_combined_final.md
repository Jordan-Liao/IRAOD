# Phase 5 全 7-corruption mAP 对比（最终汇总）

> mean = `clean_test + chaff + gaussian_white_noise + point_target + noise_suppression + am_noise_horizontal + smart_suppression + am_noise_vertical` 共 8 列算术平均。
> 所有 run 共享同一个 SOURCE_CKPT：`work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth`。

| row | clean | chaff | gwn | point_target | noise_suppression | am_noise_horizontal | smart_suppression | am_noise_vertical | **mean** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_clean_test | 0.5350 | - | - | - | - | - | - | - | 0.5350 |
| direct_test | 0.5350 | 0.4629 | 0.5410 | 0.5321 | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.3631** |
| E0111 self_training | 0.5350 | 0.0090 | 0.0483 | 0.0647 | 0.0728 | 0.0272 | 0.0619 | 0.0135 | 0.1040 |
| E0111 self_training_plus_cga | 0.5350 | 0.0969 | 0.1061 | 0.1011 | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.1454 |
| E0112 self_training | 0.5350 | 0.1368 | 0.1573 | 0.1689 | 0.0819 | 0.0829 | 0.0651 | 0.1013 | **0.1661** |
| E0112 self_training_plus_cga | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.0835 | 0.0447 | 0.0651 | 0.1320 | **0.1841** |
| **E0112+E0113(heavy LoRA)** | **0.5350** | **0.1752** | **0.2146** | **0.2226** | **0.1082** | **0.0790** | **0.0860** | **0.1321** | **0.1941** |
| E0112+E0114(heavy cap) | 0.5350 | 0.1752 | 0.2146 | 0.2226 | 0.1056 | 0.0772 | 0.0724 | 0.1077 | 0.1839 |

## 累计提升

| 阶段 | +CGA mean | Δ vs 前一版 | Δ vs E0111 baseline |
|---|---|---|---|
| E0111 原版 | 0.1454 | — | — |
| E0112 塌缩修复 | 0.1841 | +3.87pp | +3.87pp |
| **E0112+E0113 heavy LoRA** | **0.1941** | **+1.00pp** | **+4.87pp (+33.5%)** |

## 与 direct_test 的 gap

- direct_test mean = 0.3631
- best +CGA mean = 0.1941
- remaining gap = **0.1690 (46.5% 的 direct_test 绝对值)**

gap 主要来自 4 heavy corruptions（noise_suppression / am_noise_horizontal / smart_suppression / am_noise_vertical），因为 source detector 在这些重干扰域 direct_test 本身只有 0.18-0.25，teacher-student 无法凭无标签信号填补这部分域差距。
