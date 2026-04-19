# Phase 5 heavy-4 corruption 对比表

Source ckpt: `work_dirs/rsar_sfodrs_full_3gpu_20260415/source_train/latest.pth` (OrthoNet + OCAFPN + Oriented R-CNN，clean RSAR 12 epoch supervised)

| row | noise_suppression | am_noise_horizontal | smart_suppression | am_noise_vertical | heavy_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_test | 0.2471 | 0.1830 | 0.1834 | 0.2205 | **0.2085** |
| E0111 orig_self_training | 0.0728 | 0.0272 | 0.0619 | 0.0135 | 0.0439 |
| E0111 orig_self_training_plus_cga | 0.0934 | 0.0561 | 0.0751 | 0.0995 | 0.0810 |
| E0112 fixed_self_training | 0.0819 | 0.0829 | 0.0651 | 0.1013 | 0.0828 |
| E0112 fixed_self_training_plus_cga | 0.0835 | 0.0447 | 0.0651 | 0.1320 | 0.0813 |
| **E0113 heavyfix_self_training_plus_cga_lora** | **0.1082** | **0.0790** | **0.0860** | **0.1321** | **0.1013** |
| E0114 capfix_self_training_plus_cga_cap | 0.1056 | 0.0772 | 0.0724 | 0.1077 | 0.0907 |

## 绝对提升（heavy_mean）

- E0111 orig +CGA: 0.0810 (baseline)
- E0112 fixed +CGA: 0.0813（+0.03pp）
- **E0113 heavyfix +CGA_LoRA: 0.1013（+2.03pp / 相对 +25.1%）← 最佳**
- E0114 capfix +CGA_cap: 0.0907（+0.97pp）

## 实验配置快速参考

| experiment | score_thr | burn_in | weight_u | adapt_lr | adapt_epochs | early_stop max_majority | SARCLIP LoRA | pseudo cap |
|---|---|---|---|---|---|---|---|---|
| E0111 | 0.7 scalar | 0 | 1.0 | 0.02 | 12 | 0.995 | off | off |
| E0112 | 0.85,0.7×5 | 2 | 0.5 | 0.005 | 12 | 0.90 | off | off |
| E0113 heavyfix | 0.80,0.6×5 | 1 | 0.5 | 0.005 | 6 | 0.85 | **on (Interference LoRA)** | off |
| E0114 capfix | 0.7 scalar | 1 | 0.5 | 0.005 | 12 | 0.85 | on | 3000,1500×5 per-image |
