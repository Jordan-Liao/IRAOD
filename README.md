# IRAOD

IRAOD is an oriented object detection workspace for DIOR/RSAR source-free
adaptation and corruption robustness experiments.

## Quick Start

```bash
bash scripts/setup_env_iraod.sh
conda activate iraod

python tools/verify_dataset_layout.py --dior dataset/DIOR --rsar dataset/RSAR
DRY_RUN=1 SOURCE_CKPT=auto WORK_ROOT=work_dirs/smoke_refactor \
  CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 bash scripts/run/rsar_sfodrs_full.sh
```

Use `DRY_RUN=1` first on any launcher to verify paths, env, and generated
commands without starting training or evaluation.

## Recommended RSAR Entries

```bash
# Full SFOD-RS protocol. SOURCE_CKPT=auto trains the clean source detector first.
SOURCE_CKPT=auto WORK_ROOT=work_dirs/<run> CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 \
  bash scripts/run/rsar_sfodrs_full.sh

# One corruption domain.
CORR=chaff SOURCE_CKPT=work_dirs/<source>/latest.pth WORK_ROOT=work_dirs/<run> \
  bash scripts/run/rsar_sfodrs_domain.sh

# TENT adapt/eval/fusion.
TENT_MODE=adapt SOURCE_CKPT=work_dirs/<source>/latest.pth WORK_ROOT=work_dirs/<tent> \
  bash scripts/run/rsar_tent.sh

# Collect SFOD-RS result tables.
WORK_ROOT=work_dirs/<run> bash scripts/run/collect_results.sh
```

Compatibility wrappers remain for older commands such as:

```bash
bash scripts/run_rsar_sfodrs_full_3gpu.sh auto work_dirs/<run>
bash scripts/run_rsar_sfodrs_7corr.sh work_dirs/<source>/latest.pth
bash scripts/exp_rsar_sfodrs_adapt.sh chaff work_dirs/<source>/latest.pth
```

## Datasets

Default local layout:

```text
dataset/DIOR/
dataset/RSAR/
  train/{images,annfiles}
  val/{images,annfiles}
  test/{images,annfiles}
  corruptions/<corruption>/{val,test}/images
```

Override RSAR location with either `RSAR_DATA_ROOT=/path/to/RSAR` or
`DATA_ROOT=/path/to/RSAR`.

Prepare the RSAR corruption layout when needed:

```bash
python tools/prepare_rsar_corruption.py --data-root dataset/RSAR --workers 8
```

## Documentation Map

- `docs/commands.md`: current smoke/full/eval/collect command cookbook.
- `docs/project_map.md`: directory responsibilities and active vs archived entrypoints.
- `docs/experiment.md`: full experiment ledger and recorded metrics.
- `configs/README.md`: current config aliases and historical config locations.
- `MODEL_ZOO.md`: checkpoint and external weight notes.
