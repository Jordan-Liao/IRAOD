from __future__ import annotations

import argparse
from pathlib import Path

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rsar_controls_common import (
    build_adapt_loader,
    compute_roi_entropy_loss,
    configure_tent_trainable,
    evaluate_suite,
    json_dump,
    load_source_detector,
    resolve_source_defaults,
    save_model_checkpoint,
    scatter_data,
    set_random_seed,
    write_results_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tent-style entropy minimization on RSAR/train.")
    parser.add_argument("--source-config", default=None, help="Detector config for the source model.")
    parser.add_argument("--source-ckpt", default=None, help="SOURCE_CKPT path.")
    parser.add_argument("--data-root", required=True, help="RSAR dataset root.")
    parser.add_argument("--result-root", required=True, help="Root directory for control-baseline outputs.")
    parser.add_argument("--samples-per-gpu", type=int, default=4, help="Adaptation batch size.")
    parser.add_argument("--workers-per-gpu", type=int, default=2, help="Adaptation dataloader workers.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of adaptation epochs.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on optimizer steps (0 = full loader).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tent optimizer learning rate.")
    parser.add_argument("--topk", type=int, default=256, help="Max RoIs used for entropy minimization per batch.")
    parser.add_argument("--min-fg-conf", type=float, default=0.05, help="Minimum foreground confidence for selected RoIs.")
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
    trainable_params, trainable_names = configure_tent_trainable(model)
    if not trainable_params:
        raise RuntimeError("Tent found no trainable BN affine parameters.")

    model.train()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    loader = build_adapt_loader(
        source_cfg=source_cfg,
        data_root=args.data_root,
        split="train",
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
    )

    method_dir = Path(args.result_root) / "tent"
    method_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    last_stats = {"num_rois": 0.0, "num_selected": 0.0, "entropy": 0.0}
    for _epoch in range(args.epochs):
        for data in loader:
            batch = scatter_data(data, device)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = compute_roi_entropy_loss(
                model,
                batch,
                topk=args.topk,
                min_fg_conf=args.min_fg_conf,
            )
            last_stats = stats
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(
                    f"[run_tent_adapt] step={step} entropy={stats['entropy']:.4f} "
                    f"selected={int(stats['num_selected'])}/{int(stats['num_rois'])}"
                )
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    tent_ckpt = method_dir / "latest.pth"
    save_model_checkpoint(
        model,
        tent_ckpt,
        meta={
            "method": "tent",
            "seed": args.seed,
            "source_config": source_config,
            "source_ckpt": source_ckpt,
            "steps": step,
            "epochs": args.epochs,
            "lr": args.lr,
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "topk": args.topk,
            "min_fg_conf": args.min_fg_conf,
            "trainable_params": len(trainable_names),
        },
    )
    json_dump(
        method_dir / "run_meta.json",
        {
            "method": "tent",
            "source_config": source_config,
            "source_ckpt": source_ckpt,
            "tent_ckpt": str(tent_ckpt),
            "seed": args.seed,
            "epochs": args.epochs,
            "steps": step,
            "lr": args.lr,
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "topk": args.topk,
            "min_fg_conf": args.min_fg_conf,
            "last_stats": last_stats,
            "trainable_names": trainable_names,
        },
    )

    payload = evaluate_suite(
        method="tent",
        config_path=source_config,
        checkpoint_path=str(tent_ckpt),
        data_root=args.data_root,
        result_root=args.result_root,
        seed=args.seed,
        env=None,
        extra_cfg_options=None,
        include_corruptions=True,
    )
    csv_path, md_path = write_results_tables(args.result_root)
    print(f"[run_tent_adapt] steps={step}")
    print(f"[run_tent_adapt] mean={payload['metrics']['mean']:.4f}")
    print(f"[run_tent_adapt] wrote {csv_path}")
    print(f"[run_tent_adapt] wrote {md_path}")


if __name__ == "__main__":
    main()
