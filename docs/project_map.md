# Project Map

## Active Entry Points

- `scripts/common/iraod.sh`: shared shell helpers for repo root discovery,
  Python/env defaults, DDP launch construction, logging, and `DRY_RUN=1`.
- `scripts/run/rsar_sfodrs_full.sh`: recommended full RSAR SFOD-RS runner.
  Supports `SOURCE_CKPT=auto` to train `WORK_ROOT/source_train/latest.pth`.
- `scripts/run/rsar_sfodrs_domain.sh`: one-corruption runner for direct,
  adapt, eval, and CGA variants.
- `scripts/run/rsar_tent.sh`: unified TENT adapt/eval/ensemble runner.
- `scripts/run/collect_results.sh`: result table collection wrapper.

## Compatibility And Archive

Old public command names remain as thin wrappers:

- `scripts/run_rsar_sfodrs_full_3gpu.sh`
- `scripts/run_rsar_sfodrs_7corr.sh`
- `scripts/exp_rsar_sfodrs_adapt.sh`
- `scripts/run_rsar_sfodrs_tent.sh`
- `scripts/run_rsar_sfodrs_tent_adapt.sh`
- `scripts/run_rsar_sfodrs_tent_eval.sh`
- `scripts/run_rsar_sfodrs_tent_ensemble.sh`

One-off recovery and historical 7-corruption scripts live in `scripts/archive/`.
Same-name files in `scripts/` forward there and print a deprecation notice.

## Configs

- `configs/current/`: stable short aliases for current work.
- `configs/baseline/`: baseline and EMA detector definitions.
- `configs/experiments/`: DIOR/RSAR baseline and frontier experiment configs.
- `configs/unbiased_teacher/sfod/`: source-free adaptation configs and older
  phase variants kept for reproducibility.

Prefer `configs/current/rsar_sfodrs.py` for new RSAR commands unless reproducing
a specific historical result.

## Runtime Code

- `train.py` and `test.py`: public CLIs remain compatible with the original
  OpenMMLab-style scripts.
- `sfod/runtime.py`: shared CLI runtime behavior for CGA env mapping, RSAR
  data-root rewrites, dataloader overrides, max-epoch overrides, and teacher
  checkpoint injection.
- `sfod/`: detector, dataset, CGA, and compatibility code.
- `tools/`: data preparation, smoke checks, evaluation helpers, and result
  aggregation.
- `tools/rsar_corruption_pipeline.py`: `RsarOnlineCorruptionAugment` transform
  (ROTATED_PIPELINES registered). Randomly applies one of 7 SAR corruptions
  during source training when `RSAR_CORR_AUG=1`. Improved direct_test mean by
  +40.6% relative across 7 corruption domains (Phase 9).

## Key Tools

- `tools/tent_adapt_per_corr.py`: TENT adaptation script (BN affine entropy
  minimization, no pseudo-labels). Used by `scripts/run/rsar_tent.sh`.
- `tools/rsar_semi_sfodrs_dataset.py`: `SemiRSARSFODDataset` — faithful SFOD-RS
  dataloader (labeled source + unlabeled target). Activated by `RSAR_LOADER_MODE=loose`.
- `tools/sfodrs_diagnostics_hook.py`: hook logging CGA mode, keep_label, score rule.
- `tools/pseudo_stats_early_stop_hook.py`: pseudo-label per-epoch statistics.
- `tools/collect_rsar_sfodrs_results.py`: aggregate eval JSONs into CSV/MD table.

## Experiment Records

- `docs/plan.md`: phased experiment history (Phase 5–11) with results and conclusions.
- `docs/semi_supervised_experiments.md`: full method comparison table (all Exp A–F).
- `docs/mohu.md`: open ambiguities and resolved decisions.
- `docs/phase5_results/`: frozen CSV/Markdown result snapshots.
- `work_dirs/`: local generated training/eval artifacts, not an authority source
  for Git history.

## Current Best Results (2026-05-07)

| Strategy | Source Model | Mean mAP (7 corr) | Notes |
|---|---|---:|---|
| direct_test | corr-aug | **0.4742** | Best overall |
| TENT (E0129) | corr-aug | 0.4656 | Best adaptation method (-0.86pp) |
| corr-aug loose nocga | corr-aug | 0.4325 | Pseudo-label with faithful loader |
| E0128 thr=0.4 nocga | corr-aug | 0.4034 | Best pseudo-label with threshold tuning |
| strict nocga | original | 0.0402 | Complete collapse |
