from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[ablation_table] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input metrics CSV (from tools/export_metrics.py)")
    parser.add_argument("--out-md", default="work_dirs/results/ablation_table.md", help="Output markdown path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        _log("empty metrics CSV; nothing to summarize")
        return 0

    # Keep one best row per (dataset, method, corrupt, seed) in case multiple eval files exist.
    df = df.copy()
    df["mAP"] = pd.to_numeric(df["mAP"], errors="coerce")
    df = df.sort_values(["dataset", "method", "corrupt", "seed", "mAP"], ascending=[True, True, True, True, False])
    df = df.drop_duplicates(["dataset", "method", "corrupt", "seed"], keep="first")

    # Simple pivot: methods as rows, corrupt as columns (per dataset).
    lines: list[str] = []
    for dataset_name, g in df.groupby("dataset", dropna=False):
        ds = str(dataset_name) if dataset_name else "UNKNOWN"
        lines.append(f"## {ds}")
        pivot = g.pivot_table(index="method", columns="corrupt", values="mAP", aggfunc="max")
        lines.append(pivot.to_markdown())
        lines.append("")

    out_path = Path(args.out_md).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    _log(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

