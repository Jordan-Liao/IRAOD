#!/usr/bin/env python3
"""P0032: Crop object patches from RSAR (or SARDet-100K) for LoRA fine-tuning.

Given DOTA-format annotations, crop patches from images and organize them
into per-class directories for contrastive training.

Usage:
    python -m lora_finetune.crop_sardet100k \
        --img-dir dataset/RSAR/train/images \
        --ann-dir dataset/RSAR/train/annfiles \
        --output-dir work_dirs/sarclip_lora/patches \
        --classes ship aircraft car tank bridge harbor \
        --expand-ratio 0.2 --min-size 16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


def crop_patches(args):
    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.output_dir)
    classes = args.classes
    expand = args.expand_ratio
    min_size = args.min_size

    stats = {c: 0 for c in classes}
    cls_set = set(classes)

    for cls_name in classes:
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob('*.txt'))
    print(f'[crop] Found {len(ann_files)} annotation files')

    for ai, ann_file in enumerate(ann_files):
        if (ai + 1) % 500 == 0:
            print(f'  [{ai+1}/{len(ann_files)}]')

        # Find matching image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
            cand = img_dir / (ann_file.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        with open(ann_file) as f:
            lines = f.readlines()

        img = None  # lazy load
        patch_idx = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
                cls_name = parts[8]
            except (ValueError, IndexError):
                continue
            if cls_name not in cls_set:
                continue

            # Compute AABB from OBB
            xs = coords[0::2]
            ys = coords[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            w, h = x2 - x1, y2 - y1
            if w < min_size or h < min_size:
                continue

            # Expand
            x1 = max(0, x1 - w * expand)
            y1 = max(0, y1 - h * expand)
            x2 = x2 + w * expand
            y2 = y2 + h * expand

            if img is None:
                img = Image.open(img_path).convert('RGB')

            patch = img.crop((int(x1), int(y1), int(x2), int(y2)))
            patch_name = f'{ann_file.stem}_{patch_idx}.jpg'
            patch.save(out_dir / cls_name / patch_name, quality=95)
            stats[cls_name] += 1
            patch_idx += 1

    print('[crop] Done. Per-class counts:')
    total = 0
    for c in classes:
        print(f'  {c}: {stats[c]}')
        total += stats[c]
    print(f'  Total: {total}')
    return 0


def main():
    parser = argparse.ArgumentParser(description='Crop patches for LoRA training')
    parser.add_argument('--img-dir', required=True, help='Image directory')
    parser.add_argument('--ann-dir', required=True, help='Annotation directory (DOTA format)')
    parser.add_argument('--output-dir', required=True, help='Output directory for patches')
    parser.add_argument('--classes', nargs='+',
                        default=['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'])
    parser.add_argument('--expand-ratio', type=float, default=0.2)
    parser.add_argument('--min-size', type=int, default=16)
    args = parser.parse_args()
    return crop_patches(args)


if __name__ == '__main__':
    raise SystemExit(main())
