#!/usr/bin/env python3
"""Evaluate CGA test-time rescoring with different SARCLIP LoRA checkpoints.

Uses the UT-trained detector and runs test-time CGA with each LoRA variant.
Reports mAP for each configuration.

Usage:
    python eval_lora_cga.py
"""
import os
import sys
import gc
import time
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg

# Paths
CGA_CONFIG = "configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py"
DETECTOR_CKPT = "work_dirs/ut_rsar_corrected/latest_ema.pth"

LORA_CHECKPOINTS = [
    ("no_lora", ""),
    ("p0032_r8_10ep", "work_dirs/p0032_sarclip_lora/lora_final.pth"),
    ("p0033_r8_10ep_ent", "work_dirs/p0033_sarclip_lora_ent/lora_final.pth"),
    ("p0032v2_r16_30ep", "work_dirs/p0032v2_sarclip_lora_r16/lora_final.pth"),
    ("p0033v2_r16_30ep_ent", "work_dirs/p0033v2_sarclip_lora_ent_r16/lora_final.pth"),
]

GPU_ID = 0


def evaluate_with_lora(name, lora_path):
    """Run evaluation with a specific LoRA checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  LoRA: {lora_path or 'None'}")
    print(f"{'='*60}")

    # Always evaluate SARCLIP-backed CGA; LoRA is optional.
    os.environ["CGA_SCORER"] = "sarclip"
    os.environ["SARCLIP_MODEL"] = "RN50"
    if lora_path and os.path.isfile(lora_path):
        os.environ["SARCLIP_LORA"] = lora_path
        print(f"  SARCLIP_LORA = {lora_path}")
    else:
        os.environ.pop("SARCLIP_LORA", None)
        if lora_path:
            print(f"  WARNING: LoRA file not found: {lora_path}")
            return None

    # Reload CGA module to pick up new env vars
    for mod_name in list(sys.modules.keys()):
        if 'cga' in mod_name or 'sarclip_scorer' in mod_name:
            del sys.modules[mod_name]

    # Build config
    cfg = Config.fromfile(CGA_CONFIG)
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [GPU_ID]

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    # Build model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, DETECTOR_CKPT, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # Force test-time CGA for both the no-LoRA baseline and LoRA variants.
    orig_simple_test = model.simple_test

    def patched_simple_test(img, img_metas, with_cga=False, proposals=None, rescale=False):
        return orig_simple_test(
            img,
            img_metas,
            with_cga=True,
            proposals=proposals,
            rescale=rescale,
        )

    model.simple_test = patched_simple_test

    model = MMDataParallel(model, device_ids=[GPU_ID])
    model.eval()

    # Run evaluation
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    t0 = time.time()
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        prog_bar.update(len(result))

    elapsed = time.time() - t0
    print(f"\n  Inference time: {elapsed:.1f}s ({len(dataset)/elapsed:.1f} img/s)")

    # Evaluate
    eval_results = dataset.evaluate(results, metric='mAP')
    mAP = eval_results.get('mAP', 0)
    print(f"  mAP = {mAP:.4f}")

    # Cleanup
    del model, data_loader, dataset, results
    gc.collect()
    torch.cuda.empty_cache()

    return mAP


def main():
    print("="*60)
    print("  CGA + LoRA Evaluation")
    print(f"  Detector: {DETECTOR_CKPT}")
    print("="*60)

    # Check detector checkpoint exists
    if not os.path.isfile(DETECTOR_CKPT):
        print(f"ERROR: Detector checkpoint not found: {DETECTOR_CKPT}")
        return 1

    results = {}
    for name, lora_path in LORA_CHECKPOINTS:
        mAP = evaluate_with_lora(name, lora_path)
        if mAP is not None:
            results[name] = mAP

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY: CGA + LoRA Evaluation Results")
    print("="*60)
    print(f"  {'Config':<30s} {'mAP':>8s}")
    print(f"  {'-'*30} {'-'*8}")
    for name, mAP in results.items():
        print(f"  {name:<30s} {mAP:>8.4f}")
    print("="*60)

    # Save results
    import json
    out_path = "work_dirs/lora_cga_eval_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
