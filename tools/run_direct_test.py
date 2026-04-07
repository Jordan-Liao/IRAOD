from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rsar_controls_common import (
    evaluate_suite,
    json_dump,
    resolve_source_defaults,
    set_random_seed,
    write_results_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SOURCE_CKPT on clean and corruption test splits.")
    parser.add_argument("--source-config", default=None, help="Detector config used for direct evaluation.")
    parser.add_argument("--source-ckpt", default=None, help="SOURCE_CKPT path.")
    parser.add_argument("--data-root", required=True, help="RSAR dataset root.")
    parser.add_argument("--result-root", required=True, help="Root directory for control-baseline outputs.")
    parser.add_argument("--seed", type=int, default=3407, help="Evaluation seed recorded in metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config, source_ckpt = resolve_source_defaults(args.source_config, args.source_ckpt)
    set_random_seed(args.seed)

    direct_payload = evaluate_suite(
        method="direct",
        config_path=source_config,
        checkpoint_path=source_ckpt,
        data_root=args.data_root,
        result_root=args.result_root,
        seed=args.seed,
        env=None,
        extra_cfg_options=None,
        include_corruptions=True,
    )

    clean_dir = Path(args.result_root) / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_metrics = {
        "clean_test": direct_payload["metrics"]["clean_test"],
        "mean": direct_payload["metrics"]["clean_test"],
    }
    json_dump(
        clean_dir / "metrics.json",
        {
            "method": "clean",
            "config_path": source_config,
            "checkpoint_path": source_ckpt,
            "seed": args.seed,
            "metrics": clean_metrics,
            "timestamp": direct_payload["timestamp"],
        },
    )

    csv_path, md_path = write_results_tables(args.result_root)
    print(f"[run_direct_test] direct clean_test={direct_payload['metrics']['clean_test']:.4f}")
    print(f"[run_direct_test] wrote {csv_path}")
    print(f"[run_direct_test] wrote {md_path}")


if __name__ == "__main__":
    main()
