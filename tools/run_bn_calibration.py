from __future__ import annotations

import argparse
from pathlib import Path

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rsar_controls_common import (
    evaluate_suite,
    json_dump,
    build_adapt_loader,
    load_source_detector,
    resolve_source_defaults,
    save_model_checkpoint,
    scatter_data,
    set_random_seed,
    write_results_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BN-only source-free calibration on RSAR/train.")
    parser.add_argument("--source-config", default=None, help="Detector config for the source model.")
    parser.add_argument("--source-ckpt", default=None, help="SOURCE_CKPT path.")
    parser.add_argument("--data-root", required=True, help="RSAR dataset root.")
    parser.add_argument("--result-root", required=True, help="Root directory for control-baseline outputs.")
    parser.add_argument("--samples-per-gpu", type=int, default=16, help="Calibration batch size.")
    parser.add_argument("--workers-per-gpu", type=int, default=4, help="Calibration dataloader workers.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config, source_ckpt = resolve_source_defaults(args.source_config, args.source_ckpt)
    set_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_cfg, model = load_source_detector(
        source_config=source_config,
        source_ckpt=source_ckpt,
        device=device,
        with_cga=False,
    )

    for param in model.parameters():
        param.requires_grad = False
    model.train()

    loader = build_adapt_loader(
        source_cfg=source_cfg,
        data_root=args.data_root,
        split="train",
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
    )

    batch_count = 0
    with torch.no_grad():
        for data in loader:
            batch_count += 1
            batch = scatter_data(data, device)
            model(return_loss=False, rescale=False, **batch)

    method_dir = Path(args.result_root) / "bn"
    method_dir.mkdir(parents=True, exist_ok=True)
    calibrated_ckpt = method_dir / "latest.pth"
    save_model_checkpoint(
        model,
        calibrated_ckpt,
        meta={
            "method": "bn",
            "seed": args.seed,
            "source_config": source_config,
            "source_ckpt": source_ckpt,
            "adapt_split": "RSAR/train",
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "num_batches": batch_count,
        },
    )
    json_dump(
        method_dir / "run_meta.json",
        {
            "method": "bn",
            "source_config": source_config,
            "source_ckpt": source_ckpt,
            "calibrated_ckpt": str(calibrated_ckpt),
            "seed": args.seed,
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "num_batches": batch_count,
        },
    )

    payload = evaluate_suite(
        method="bn",
        config_path=source_config,
        checkpoint_path=str(calibrated_ckpt),
        data_root=args.data_root,
        result_root=args.result_root,
        seed=args.seed,
        env=None,
        extra_cfg_options=None,
        include_corruptions=True,
    )
    csv_path, md_path = write_results_tables(args.result_root)
    print(f"[run_bn_calibration] num_batches={batch_count}")
    print(f"[run_bn_calibration] mean={payload['metrics']['mean']:.4f}")
    print(f"[run_bn_calibration] wrote {csv_path}")
    print(f"[run_bn_calibration] wrote {md_path}")


if __name__ == "__main__":
    main()
