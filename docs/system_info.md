# System Info

## Snapshot
- Timestamp: `2026-04-27 01:04 CST`
- Host: `222.200.185.183`
- Repo path: `/mnt/SSD1_8TB/zechuan/IRAOD`
- Remote HEAD at snapshot: `46423ad03e78e4da040df9c9687922b9c9c5603f`
- Remote working tree dirty files count at snapshot: `5`

## Python / Framework
- Python: `3.10.19`
- Executable: `/home/zechuan/anaconda3/envs/iraod/bin/python`
- mmcv: `1.7.2`
- mmdet: `2.28.2`
- mmrotate: `0.3.4`
- torch: `2.0.1+cu118`
- torch CUDA build: `11.8`

## GPU Hardware
- GPU count: `10`
- Model: `NVIDIA GeForce RTX 4090 D`
- VRAM per card: `49140 MiB`
- Driver: `570.133.20`

## Run-Time Notes
1. For this project, `PYTHONNOUSERSITE=1` should be used in launch/resume scripts to avoid `~/.local` package pollution.
2. During final completion check (`2026-04-27 00:51 CST`), IRAOD training processes were no longer active.
3. At that same check, GPUs `0/4/8/9` were occupied by other jobs; `1/2/3/6/7` were idle and `5` had memory occupancy with `0%` utilization.
