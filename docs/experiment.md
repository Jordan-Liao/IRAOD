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
