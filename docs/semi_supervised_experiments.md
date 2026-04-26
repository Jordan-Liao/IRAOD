# RSAR Semi-Supervised Experiments (SFOD-RS Target-Only, 2026-04)

## Scope
- Dataset: RSAR 6 classes (`ship`, `aircraft`, `car`, `tank`, `bridge`, `harbor`)
- Corruptions: 7 domains (`chaff`, `gaussian_white_noise`, `point_target`, `noise_suppression`, `am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`)
- Protocol:
  - source model trained on clean labeled source split
  - adaptation uses target-domain `val/images` only
  - evaluation uses target-domain `test/images`
  - no corruption train split used

## Core Metrics (mAP)
| Setting | Mean (clean + 7 domains) | Mean (7 corrupted domains only) |
| --- | ---: | ---: |
| direct_test | 0.3624 | 0.3372 |
| self_training (no CGA) | 0.1025 | 0.0402 |
| self_training_plus_cga | 0.1402 | 0.0833 |

## CGA Contribution
- `self_training_plus_cga` vs `self_training` on 7 corrupted domains:
  - absolute gain: `+0.0431 mAP` (mean)
  - gains are positive on all 7 domains
- Domain-level gains:
  - `chaff +0.0727`
  - `gaussian_white_noise +0.0736`
  - `point_target +0.0570`
  - `noise_suppression +0.0091`
  - `am_noise_horizontal +0.0295`
  - `smart_suppression +0.0182`
  - `am_noise_vertical +0.0413`

## Observation
1. Target-only self-training under this configuration is substantially below direct test baseline.
2. CGA mitigates the drop consistently, but does not recover to direct-test level yet.
3. Logs confirm SFOD-RS scoring rule in eval path:
   `0.7*teacher + 0.3*clip_prob_orig` with `keep_label=True`.

## Traceability
- Run root: `work_dirs/rsar_sfodrs_full_fix_20260424_172627`
- Aggregated outputs:
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md`
- Completion logs:
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/launch.log`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/point_target_parallel.log`
  - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/noise_suppression_parallel.log`
