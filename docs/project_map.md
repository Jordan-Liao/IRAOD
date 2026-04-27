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

## Experiment Records

- `docs/experiment.md`: complete ledger with commands, artifacts, and metrics.
- `docs/phase5_results/`: frozen CSV/Markdown result snapshots.
- `work_dirs/`: local generated training/eval artifacts, not an authority source
  for Git history.
