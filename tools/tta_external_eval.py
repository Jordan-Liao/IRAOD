"""External multi-scale + flip TTA that bypasses mmrotate's missing aug_test.
Runs simple_test multiple times with different augmentations, then merges per-image
per-class predictions via union + rotated NMS.

Usage:
    python tools/tta_external_eval.py \
      --source-cfg configs/.../sfodrs_rsar.py \
      --source-ckpt <ckpt> \
      --corruption chaff \
      --ref-ann dataset/RSAR/val/annfiles \
      --target-img dataset/RSAR/corruptions/chaff/test/images \
      --work-dir out_dir \
      --scales 0.8 1.0 1.2 --flip-directions horizontal
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmcv
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes

from tools.ensemble_merge_eval import _rotated_nms, merge_per_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--source-cfg', required=True)
    p.add_argument('--source-ckpt', required=True)
    p.add_argument('--corruption', required=True)
    p.add_argument('--target-img', required=True)
    p.add_argument('--ref-ann', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, 1.15])
    p.add_argument('--flip-directions', nargs='+', default=['horizontal'], choices=['horizontal', 'vertical', 'none'])
    p.add_argument('--nms-iou', type=float, default=0.1)
    p.add_argument('--max-per-img', type=int, default=2000)
    return p.parse_args()


def _build_loader(cfg: Config, ann_dir: str, img_dir: str, scale_factor: float):
    """Build a test dataloader with scale + (no flip here, we'll flip boxes after inference)."""
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_dir
    test_cfg['img_prefix'] = img_dir
    test_cfg['test_mode'] = True

    # Locate MultiScaleFlipAug and override img_scale
    pipeline = test_cfg['pipeline']
    for step in pipeline:
        if step.get('type') == 'MultiScaleFlipAug':
            base_size = step.get('img_scale', (800, 800))
            if isinstance(base_size, list):
                base_size = base_size[0]
            new_size = (int(base_size[0] * scale_factor), int(base_size[1] * scale_factor))
            step['img_scale'] = new_size
            step['flip'] = False
            break

    dataset = build_dataset(test_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )
    return dataset, loader


def _flip_rbboxes(results: list[np.ndarray], img_w: float, direction: str) -> list[np.ndarray]:
    """Flip rotated bbox predictions back to original coord frame.
    results: list[n_classes] of (N, 6) [cx, cy, w, h, angle, score]"""
    out = []
    for r in results:
        if r.size == 0:
            out.append(r)
            continue
        r = r.copy()
        if direction == 'horizontal':
            r[:, 0] = img_w - r[:, 0]
            r[:, 4] = -r[:, 4]
        elif direction == 'vertical':
            r[:, 1] = img_w - r[:, 1]  # we use img_h assumed==img_w for simplicity
            r[:, 4] = -r[:, 4]
        out.append(r)
    return out


def _run_once(
    model, loader, device, flip_direction: str | None,
):
    """Run simple_test over the loader, optionally flipping input then un-flipping boxes."""
    results = []
    for data in mmcv.track_iter_progress(loader):
        # scatter() unwraps DataContainer -> Tensor, moves to device
        batch = scatter(data, [device.index])[0]
        # batch['img'] is a list of tensors [B,C,H,W] (one per scale view, single here)
        if flip_direction and flip_direction != 'none':
            img_list = batch['img']
            flipped = []
            for t in img_list:
                if flip_direction == 'horizontal':
                    flipped.append(torch.flip(t, dims=[-1]))
                elif flip_direction == 'vertical':
                    flipped.append(torch.flip(t, dims=[-2]))
                else:
                    flipped.append(t)
            batch['img'] = flipped
        with torch.no_grad():
            res = model(return_loss=False, rescale=True, **batch)
        if flip_direction and flip_direction != 'none':
            # Un-flip predicted boxes back to original coord frame
            for i, one_img in enumerate(res):
                img_meta = batch['img_metas'][0][i]
                ori_h, ori_w = img_meta['ori_shape'][:2]
                if flip_direction == 'horizontal':
                    unflipped = _flip_rbboxes(one_img, ori_w, 'horizontal')
                elif flip_direction == 'vertical':
                    unflipped = _flip_rbboxes(one_img, ori_h, 'vertical')
                else:
                    unflipped = one_img
                res[i] = unflipped
        results.extend(res)
    return results


def main() -> None:
    args = parse_args()
    os.environ.setdefault('RSAR_STAGE', 'target_eval')
    os.environ.setdefault('RSAR_TARGET_DOMAIN', args.corruption)
    os.environ.setdefault('RSAR_USE_CGA', '0')
    os.environ.setdefault('RSAR_USE_TTA', '0')  # IMPORTANT: disable the broken in-pipeline TTA

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = Config.fromfile(args.source_cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.source_ckpt, map_location='cpu')
    dp = MMDataParallel(model.to(device), device_ids=[device.index])
    dp.eval()
    if hasattr(dp, 'module'):
        dp.module.CLASSES = None  # will be set by first loader

    # Run all (scale, flip) combos
    all_views: list[list[list[np.ndarray]]] = []  # [n_views][n_imgs][n_classes]
    first_loader_dataset = None
    for scale in args.scales:
        for flip_dir in args.flip_directions:
            tag = f'scale{scale:.2f}_flip{flip_dir}'
            dataset, loader = _build_loader(cfg, args.ref_ann, args.target_img, scale)
            if first_loader_dataset is None:
                first_loader_dataset = dataset
                if hasattr(dp, 'module'):
                    dp.module.CLASSES = dataset.CLASSES
            print(f'[TTA] running {tag} on {len(dataset)} imgs', flush=True)
            res = _run_once(dp, loader, device, flip_direction=flip_dir)
            all_views.append(res)
            print(f'[TTA] {tag} done: {len(res)} results', flush=True)

    # Merge views per image
    assert first_loader_dataset is not None
    n_imgs = len(all_views[0])
    print(f'[TTA] merging {len(all_views)} views over {n_imgs} imgs', flush=True)
    merged: list[list[np.ndarray]] = []
    for i in range(n_imgs):
        acc = all_views[0][i]
        for v in range(1, len(all_views)):
            acc = merge_per_image(acc, all_views[v][i], iou_thr=args.nms_iou, max_per_img=args.max_per_img)
        merged.append(acc)

    # Save + evaluate
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_pkl = out_dir / 'tta_merged.pkl'
    mmcv.dump(merged, str(merged_pkl))

    eval_res = first_loader_dataset.evaluate(merged, metric='mAP')
    print(f'[TTA] mAP={eval_res}', flush=True)

    import json
    out_json = out_dir / 'eval_tta.json'
    payload = {
        'config': args.source_cfg,
        'ckpt': args.source_ckpt,
        'corruption': args.corruption,
        'scales': list(args.scales),
        'flip_directions': list(args.flip_directions),
        'metric': {k: float(v) for k, v in eval_res.items()},
        'merged_pkl': str(merged_pkl),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'[TTA] saved -> {out_json}', flush=True)


if __name__ == '__main__':
    main()
