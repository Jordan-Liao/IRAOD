"""Proper TENT: freeze everything except BN affine params (weight/bias),
minimize entropy of RoI classification head on target val images.

Usage:
    python tools/tent_adapt_per_corr.py \
      --source-cfg configs/.../sfodrs_rsar.py \
      --source-ckpt work_dirs/.../source_train/latest.pth \
      --corruption chaff \
      --target-img dataset/RSAR/corruptions/chaff/val/images \
      --ref-ann dataset/RSAR/val/annfiles \
      --out work_dirs/tent/chaff/latest.pth \
      --epochs 2 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmcv
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--source-cfg', required=True)
    p.add_argument('--source-ckpt', required=True)
    p.add_argument('--corruption', required=True)
    p.add_argument('--target-img', required=True)
    p.add_argument('--ref-ann', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--samples-per-gpu', type=int, default=2)
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--conf-thr', type=float, default=0.5,
                   help='only high-confidence RoIs contribute to entropy loss')
    p.add_argument('--max-batches', type=int, default=0,
                   help='0=all batches per epoch')
    return p.parse_args()


def setup_tent_model(cfg_path: str, ckpt_path: str, device: torch.device):
    """Load detector, freeze all params except BN affine."""
    cfg = Config.fromfile(cfg_path)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model = model.to(device)
    model.eval()  # IMPORTANT: keep BN in eval mode to use running stats

    # Freeze everything, then unfreeze BN affine params
    bn_params: list[torch.nn.Parameter] = []
    n_bn = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            n_bn += 1
            if m.weight is not None:
                m.weight.requires_grad_(True)
                bn_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                bn_params.append(m.bias)
            # Keep running_mean/var frozen (stay in eval mode)
            m.track_running_stats = False

    for p_ in model.parameters():
        if not any(p_ is bp for bp in bn_params):
            p_.requires_grad_(False)

    print(f'[TENT] BN modules={n_bn}; trainable params={sum(p.numel() for p in bn_params)} '
          f'(total_frozen={sum(p.numel() for p in model.parameters() if not p.requires_grad)})',
          flush=True)
    return cfg, model, bn_params


def _forward_with_grad(model_raw, img, img_metas):
    """Forward with gradient, return RoI head cls_score.

    OrientedRCNN's rpn_head produces oriented proposals (cx, cy, w, h, angle, score).
    roi_head uses rbbox2roi + RoIAlignRotated."""
    x = model_raw.extract_feat(img)
    with torch.no_grad():
        proposal_list = model_raw.rpn_head.simple_test_rpn(x, img_metas)
    from mmrotate.core import rbbox2roi
    # proposal_list: List[B] of (N_i, 6) -> strip score column
    proposals_noscore = [p[:, :5] for p in proposal_list]
    rois = rbbox2roi(proposals_noscore)
    if rois.shape[0] == 0:
        return None
    roi_head = model_raw.roi_head
    bbox_results = roi_head._bbox_forward(x, rois)
    return bbox_results['cls_score']


def tent_loss(cls_score: torch.Tensor, conf_thr: float) -> torch.Tensor:
    """Entropy loss on high-confidence RoIs."""
    probs = F.softmax(cls_score, dim=1)
    log_probs = F.log_softmax(cls_score, dim=1)
    max_probs, _ = probs.max(dim=1)
    mask = max_probs > conf_thr
    if mask.sum() == 0:
        # Fallback: use all RoIs
        entropy = -(probs * log_probs).sum(dim=1).mean()
    else:
        entropy = -(probs[mask] * log_probs[mask]).sum(dim=1).mean()
    return entropy


def main() -> None:
    args = parse_args()
    os.environ.setdefault('RSAR_STAGE', 'direct_test')
    os.environ.setdefault('RSAR_TARGET_DOMAIN', args.corruption)
    os.environ.setdefault('RSAR_USE_CGA', '0')
    os.environ.setdefault('RSAR_USE_TTA', '0')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg, model, bn_params = setup_tent_model(args.source_cfg, args.source_ckpt, device)

    # Override data.test to point at target val
    test_cfg = dict(cfg.data.test)
    test_cfg['ann_file'] = args.ref_ann
    test_cfg['img_prefix'] = args.target_img
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)
    import numpy as _np
    if not hasattr(dataset, "flag"):
        dataset.flag = _np.zeros(len(dataset), dtype=_np.uint8)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=True,
    )

    optimizer = torch.optim.SGD(bn_params, lr=args.lr, momentum=0.9)

    dp = MMDataParallel(model, device_ids=[0])
    model_raw = dp.module if hasattr(dp, 'module') else model

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_batches = 0
        total_rois = 0
        for i, data in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            img = data['img'][0].data[0] if isinstance(data['img'], list) else data['img']
            img_metas = data['img_metas'][0].data[0] if isinstance(data['img_metas'], list) else data['img_metas']
            if not isinstance(img, torch.Tensor):
                # MultiScaleFlipAug wraps in a list
                img = img[0] if isinstance(img, list) else img
            img = img.to(device)
            if not isinstance(img_metas, list):
                img_metas = [img_metas]
            # If img_metas is list-of-DataContainers or list-of-dict-lists, flatten
            if isinstance(img_metas[0], list):
                img_metas = img_metas[0]

            optimizer.zero_grad()
            try:
                cls_score = _forward_with_grad(model_raw, img, img_metas)
            except Exception as e:
                print(f'[TENT] epoch={epoch} batch={i} forward skipped: {e}', flush=True)
                continue
            if cls_score is None or cls_score.numel() == 0:
                continue
            loss = tent_loss(cls_score, args.conf_thr)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bn_params, max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1
            total_rois += int(cls_score.shape[0])
            if (i + 1) % 50 == 0:
                print(f'[TENT] {args.corruption} ep={epoch} batch={i+1} '
                      f'loss_avg={total_loss/max(1,total_batches):.4f} rois={total_rois}',
                      flush=True)
        print(f'[TENT] {args.corruption} EPOCH {epoch} DONE loss_avg={total_loss/max(1,total_batches):.4f} '
              f'batches={total_batches} rois={total_rois}', flush=True)

    # Save calibrated ckpt
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model_raw,
        str(out_path),
        meta=dict(
            method='tent_per_corr',
            corruption=args.corruption,
            source_ckpt=args.source_ckpt,
            target_img=args.target_img,
            epochs=args.epochs,
            lr=args.lr,
            conf_thr=args.conf_thr,
        ),
    )
    print(f'[TENT] saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
