# Commands

All commands assume the repo root as the working directory. Set `PYTHON` only
when you need a non-default interpreter.

## Smoke The Launchers

```bash
DRY_RUN=1 SOURCE_CKPT=auto WORK_ROOT=work_dirs/smoke_refactor \
  CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 bash scripts/run/rsar_sfodrs_full.sh

DRY_RUN=1 CORR=chaff SOURCE_CKPT=work_dirs/source/latest.pth \
  WORK_ROOT=work_dirs/smoke_refactor bash scripts/run/rsar_sfodrs_domain.sh

DRY_RUN=1 TENT_MODE=adapt SOURCE_CKPT=work_dirs/source/latest.pth \
  WORK_ROOT=work_dirs/tent_smoke bash scripts/run/rsar_tent.sh
```

## Full RSAR SFOD-RS

```bash
SOURCE_CKPT=auto \
WORK_ROOT=work_dirs/rsar_sfodrs_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29501 \
bash scripts/run/rsar_sfodrs_full.sh
```

Use an existing source checkpoint:

```bash
SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
WORK_ROOT=work_dirs/rsar_sfodrs_existing_source \
CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29501 \
bash scripts/run/rsar_sfodrs_full.sh
```

Remote example:

```bash
PYTHON=/home/zechuan/anaconda3/envs/iraod/bin/python \
RSAR_DATA_ROOT=/mnt/SSD1_8TB/zechuan/IRAOD/dataset/RSAR \
SOURCE_CKPT=auto WORK_ROOT=work_dirs/rsar_sfodrs_remote \
CUDA_VISIBLE_DEVICES=6,7,8 NGPUS=3 MASTER_PORT=29504 \
bash scripts/run/rsar_sfodrs_full.sh
```

## One Corruption

```bash
CORR=chaff \
SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
WORK_ROOT=work_dirs/rsar_sfodrs_chaff \
CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29502 \
bash scripts/run/rsar_sfodrs_domain.sh
```

Run one stage only:

```bash
RSAR_DOMAIN_MODE=direct CORR=chaff SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
  WORK_ROOT=work_dirs/rsar_sfodrs_chaff bash scripts/run/rsar_sfodrs_domain.sh

RSAR_DOMAIN_MODE=adapt CORR=chaff SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
  WORK_ROOT=work_dirs/rsar_sfodrs_chaff bash scripts/run/rsar_sfodrs_domain.sh

RSAR_DOMAIN_MODE=eval CORR=chaff SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
  WORK_ROOT=work_dirs/rsar_sfodrs_chaff bash scripts/run/rsar_sfodrs_domain.sh
```

## TENT

```bash
TENT_MODE=adapt SOURCE_CKPT=work_dirs/rsar_source/latest.pth \
  WORK_ROOT=work_dirs/rsar_tent bash scripts/run/rsar_tent.sh

TENT_MODE=eval WORK_ROOT=work_dirs/rsar_tent \
  CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 MASTER_PORT=29511 \
  bash scripts/run/rsar_tent.sh

TENT_MODE=ensemble WORK_ROOT=work_dirs/rsar_tent \
  DIRECT_WORK_ROOT=work_dirs/rsar_direct_preds OUT_ROOT=work_dirs/rsar_tent_ensemble \
  bash scripts/run/rsar_tent.sh
```

## Collect Results

```bash
WORK_ROOT=work_dirs/rsar_sfodrs_remote bash scripts/run/collect_results.sh
```

The output files default to:

- `WORK_ROOT/rsar_sfodrs_results.csv`
- `WORK_ROOT/rsar_sfodrs_results.md`
