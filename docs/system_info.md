# System Info
- Timestamp: 2026-04-05T22:00:00+08:00
- Python: `3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]`
- Executable: `/home/zechuan/anaconda3/envs/iraod/bin/python`
- CONDA_DEFAULT_ENV: `iraod`
- Git: `2453ef8` (Phase 3 complete)

## Key Versions
- mmcv: `1.7.2`
- mmdet: `2.28.2`
- mmrotate: `0.3.4`
- torch: `2.0.1+cu118`
- torch.cuda.current_device: `0`
- torch.cuda.device_count: `10`
- torch.cuda.get_device_name: `NVIDIA GeForce RTX 4090 D`
- torch.cuda.is_available: `True`
- torch.version.cuda: `11.8`

## 训练环境 (Phase 3)
- Conda env: `iraod`
- GPU 使用: 5× RTX 4090 D (GPU 1-5), 每卡 ~24 GB VRAM
- Batch size: 8 per GPU, 总 BS=40
- 分布式: `torch.distributed.launch`, NCCL backend
- SARCLIP: ViT-L-14 + LoRA (122,880 params)
- Precomputed features: `work_dirs/sarclip_features/`

## nvidia-smi
```text
10× NVIDIA GeForce RTX 4090 D, 49140 MiB each
Driver: 570.133.20, CUDA: 12.8
```
