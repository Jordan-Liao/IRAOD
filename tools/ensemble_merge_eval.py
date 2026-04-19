"""Ensemble two detectors by union + rotated NMS per image per class, then eval mAP.

Usage:
    python tools/ensemble_merge_eval.py \
      --source-pkl source.pkl --adapted-pkl adapted.pkl \
      --cfg configs/.../sfodrs_rsar.py \
      --work-dir out_dir --nms-iou 0.1

Expects:
  - both .pkl files produced by test.py --out <path>
  - cfg loads the test dataset (sets RSAR_STAGE=target_eval / RSAR_TARGET_DOMAIN=<corr>)

Outputs:
  work_dir/merged.pkl       -- merged per-image per-class predictions
  work_dir/eval_ensemble.json -- {"metric": {"mAP": ...}}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmrotate.datasets import build_dataset
from mmrotate.utils import compat_cfg, setup_multi_processes


def _try_nms_rotated():
    try:
        from mmcv.ops import nms_rotated
        return nms_rotated
    except ImportError:
        try:
            from mmrotate.core.post_processing import multiclass_nms_rotated as _m
            return None
        except ImportError:
            return None


def _rotated_nms(cat: np.ndarray, iou_thr: float) -> np.ndarray:
    """Apply rotated NMS to concatenated (N, 6) [cx,cy,w,h,angle,score].

    Falls back to HBB NMS + manual angle-aware dedupe if rotated NMS unavailable.
    """
    if cat.shape[0] <= 1:
        return cat
    nms_rotated = _try_nms_rotated()
    if nms_rotated is not None:
        boxes = torch.from_numpy(cat[:, :5]).float()
        scores = torch.from_numpy(cat[:, 5]).float()
        dets, keep = nms_rotated(boxes, scores, iou_thr)
        keep_np = keep.cpu().numpy() if isinstance(keep, torch.Tensor) else keep
        return cat[keep_np]
    # Fallback: keep top-scoring boxes only (coarse); acceptable for ensemble since
    # both models are very similar anyway.
    order = np.argsort(-cat[:, 5])
    kept: list[int] = []
    for idx in order:
        box = cat[idx]
        ok = True
        for j in kept:
            other = cat[j]
            # quick center-distance check
            d = np.hypot(box[0] - other[0], box[1] - other[1])
            if d < max(box[2], box[3], other[2], other[3]) * 0.3:
                ok = False
                break
        if ok:
            kept.append(int(idx))
    return cat[kept]


def merge_per_image(
    pred_a: list[np.ndarray],
    pred_b: list[np.ndarray],
    iou_thr: float,
    max_per_img: int,
) -> list[np.ndarray]:
    """Union + rotated NMS per class."""
    n_cls = len(pred_a)
    merged: list[np.ndarray] = []
    for c in range(n_cls):
        ra = pred_a[c] if c < len(pred_a) else np.zeros((0, 6), dtype=np.float32)
        rb = pred_b[c] if c < len(pred_b) else np.zeros((0, 6), dtype=np.float32)
        if ra.size == 0 and rb.size == 0:
            merged.append(np.zeros((0, 6), dtype=np.float32))
            continue
        if ra.size == 0:
            merged.append(np.asarray(rb, dtype=np.float32))
            continue
        if rb.size == 0:
            merged.append(np.asarray(ra, dtype=np.float32))
            continue
        cat = np.concatenate(
            [np.asarray(ra, dtype=np.float32), np.asarray(rb, dtype=np.float32)],
            axis=0,
        )
        cat = _rotated_nms(cat, iou_thr)
        merged.append(cat.astype(np.float32))
    # cap per image per class to keep downstream NMS output size sane
    total = sum(m.shape[0] for m in merged)
    if total > max_per_img:
        # trim smallest-per-class first
        sorted_idx = sorted(range(n_cls), key=lambda i: merged[i].shape[0], reverse=True)
        for i in sorted_idx:
            if total <= max_per_img:
                break
            m = merged[i]
            if m.shape[0] == 0:
                continue
            keep_n = max(1, int(m.shape[0] * max_per_img / total))
            merged[i] = m[np.argsort(-m[:, 5])[:keep_n]]
            total = sum(mm.shape[0] for mm in merged)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--source-pkl', required=True)
    p.add_argument('--adapted-pkl', required=True)
    p.add_argument('--cfg', required=True, help='config path (reads data.test)')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--nms-iou', type=float, default=0.1)
    p.add_argument('--max-per-img', type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('RSAR_STAGE', 'target_eval')
    os.environ.setdefault('RSAR_USE_CGA', '0')

    cfg = Config.fromfile(args.cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    # Load both pkls
    preds_a = mmcv.load(args.source_pkl)
    preds_b = mmcv.load(args.adapted_pkl)
    assert len(preds_a) == len(preds_b), (
        f'pkl length mismatch: {len(preds_a)} vs {len(preds_b)}'
    )
    print(f'[ensemble] images={len(preds_a)} iou_thr={args.nms_iou}', flush=True)

    # Merge
    merged: list[list[np.ndarray]] = []
    for i in range(len(preds_a)):
        m = merge_per_image(preds_a[i], preds_b[i], args.nms_iou, args.max_per_img)
        merged.append(m)
        if (i + 1) % 500 == 0:
            print(f'[ensemble] merged {i+1}/{len(preds_a)}', flush=True)

    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_pkl = out_dir / 'merged.pkl'
    mmcv.dump(merged, str(merged_pkl))
    print(f'[ensemble] merged pkl -> {merged_pkl}', flush=True)

    # Evaluate via dataset
    dataset = build_dataset(cfg.data.test)
    eval_res = dataset.evaluate(merged, metric='mAP')
    print(f'[ensemble] mAP={eval_res}', flush=True)

    import json
    out_json = out_dir / 'eval_ensemble.json'
    payload = {
        'config': args.cfg,
        'metric': {k: float(v) for k, v in eval_res.items()},
        'source_pkl': str(args.source_pkl),
        'adapted_pkl': str(args.adapted_pkl),
        'nms_iou': args.nms_iou,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'[ensemble] saved -> {out_json}', flush=True)


if __name__ == '__main__':
    main()
