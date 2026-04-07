from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rsar_controls_common import write_results_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-method control results into CSV/Markdown tables.")
    parser.add_argument("--result-root", required=True, help="Root directory containing <method>/metrics.json files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, md_path = write_results_tables(args.result_root)
    print(f"[collect_controls_results] wrote {csv_path}")
    print(f"[collect_controls_results] wrote {md_path}")


if __name__ == "__main__":
    main()
