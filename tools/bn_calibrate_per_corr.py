"""Per-corruption BN-only source-free calibration (TENT-style).

Usage:
  python tools/bn_calibrate_per_corr.py \
    --source-cfg configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py \
    --source-ckpt work_dirs/.../source_train/latest.pth \
    --corruption chaff \
    --target-img dataset/RSAR/corruptions/chaff/val/images \
    --out work_dirs/bn_cal/chaff/latest.pth

Freezes all params, sets BN modules to train() (norm_eval disabled),
loops over target images with torch.no_grad(), then saves state_dict.
"""
from __future__ import annotations

import sys
import os as _os
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import os
import os.path as osp
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--source-cfg', required=True)
    p.add_argument('--source-ckpt', required=True)
    p.add_argument('--corruption', required=True, help='corruption name for logging only')
    p.add_argument('--target-img', required=True, help='target image directory (flat list of .png/.jpg)')
    p.add_argument('--ref-ann', required=True, help='reference annfile dir to pair filenames (used only for building dataset)')
    p.add_argument('--out', required=True, help='output calibrated ckpt path')
    p.add_argument('--samples-per-gpu', type=int, default=8)
    p.add_argument('--workers-per-gpu', type=int, default=4)
    p.add_argument('--max-batches', type=int, default=0, help='0 = all')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('RSAR_STAGE', 'direct_test')
    os.environ.setdefault('RSAR_TARGET_DOMAIN', args.corruption)
    os.environ.setdefault('RSAR_USE_CGA', '0')
    os.environ.setdefault('RSAR_USE_TTA', '0')

    cfg = Config.fromfile(args.source_cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None

    # Override test dataset to point at the corruption/val/images
    test_cfg = dict(cfg.data.test)
    test_cfg['ann_file'] = args.ref_ann
    test_cfg['img_prefix'] = args.target_img
    test_cfg['test_mode'] = True

    dataset = build_dataset(test_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.source_ckpt, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = model.to(device).eval()

    # Freeze all params
    for p in model.parameters():
        p.requires_grad_(False)

    # Activate BN train mode (running_mean/var will track target-domain batches)
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            m.train()
            m.momentum = 0.1  # default; could lower for smoother cal
            bn_count += 1
    print(f'[BN-cal] {args.corruption}: activated {bn_count} BN layers in train mode', flush=True)

    # Forward loop
    dp = MMDataParallel(model, device_ids=[0])
    batch_count = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            dp(return_loss=False, rescale=False, **data)
            batch_count += 1
            if (i + 1) % 50 == 0:
                print(f'[BN-cal] {args.corruption}: batch {i+1}', flush=True)

    print(f'[BN-cal] {args.corruption}: total batches={batch_count}', flush=True)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        str(out_path),
        meta=dict(
            method='bn_calibrate_per_corr',
            corruption=args.corruption,
            target_img=args.target_img,
            source_ckpt=args.source_ckpt,
            num_batches=batch_count,
        ),
    )
    print(f'[BN-cal] saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
