from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _log(msg: str) -> None:
    print(f"[plot_all] {msg}")


def _err(msg: str) -> None:
    print(f"[plot_all] ERROR: {msg}", file=sys.stderr)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_map(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    df["corrupt"] = df["corrupt"].astype(str)

    method_order = ["baseline", "ut", "ut+cga", "cga"]

    for dataset in sorted([d for d in df["dataset"].unique() if d]):
        ddf = df[df["dataset"] == dataset].copy()
        if ddf.empty:
            continue

        # Reduce duplicates if any.
        ddf = (
            ddf.groupby(["method", "corrupt"], as_index=False)
            .agg(mAP=("mAP", "mean"))
            .sort_values(["method", "corrupt"])
        )

        methods = [m for m in method_order if m in set(ddf["method"])]
        methods += [m for m in sorted(set(ddf["method"])) if m not in methods]

        corrupt_order = ["clean", "cloudy", "brightness", "contrast"]
        corrupts = [c for c in corrupt_order if c in set(ddf["corrupt"])]
        corrupts += [c for c in sorted(set(ddf["corrupt"])) if c not in corrupts]

        pivot = ddf.pivot(index="corrupt", columns="method", values="mAP").reindex(corrupts)

        x = range(len(corrupts))
        bar_w = 0.8 / max(1, len(methods))

        plt.figure(figsize=(10, 4))
        for j, m in enumerate(methods):
            ys = pivot[m].fillna(0.0).to_list() if m in pivot.columns else [0.0] * len(corrupts)
            plt.bar([i + j * bar_w for i in x], ys, width=bar_w, label=m)

        plt.xticks([i + bar_w * (len(methods) - 1) / 2 for i in x], corrupts, rotation=0)
        plt.ylim(0.0, 1.0)
        plt.ylabel("mAP")
        plt.title(f"{dataset} mAP (from metrics.csv)")
        plt.legend(ncol=min(4, len(methods)), fontsize=9)

        _savefig(out_dir / f"map_{dataset.lower()}.png")


def _read_mmcv_log_json(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("mode") != "train":
                continue
            rows.append(obj)
    return pd.DataFrame(rows)


def _plot_training_curves(log_json_paths: list[Path], out_dir: Path) -> None:
    if not log_json_paths:
        return

    keys = [
        ("loss", "loss"),
        ("pseudo_num", "pseudo_num"),
        ("pseudo_num(acc)", "pseudo_num(acc)"),
    ]

    for p in log_json_paths:
        df = _read_mmcv_log_json(p)
        if df.empty:
            continue

        if "iter" not in df.columns:
            continue
        df = df.sort_values("iter")

        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3))
        if len(keys) == 1:
            axes = [axes]

        for ax, (col, title) in zip(axes, keys, strict=True):
            if col not in df.columns:
                ax.set_axis_off()
                continue
            ax.plot(df["iter"], df[col], linewidth=1.0)
            ax.set_title(title)
            ax.set_xlabel("iter")
            ax.grid(True, linewidth=0.3, alpha=0.6)

        fig.suptitle(f"{p.parent.name} ({p.name})", fontsize=10)
        fig.tight_layout()
        out_path = out_dir / "curves" / f"{p.parent.name}__{p.stem}.png"
        _savefig(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, help="Path to metrics.csv")
    parser.add_argument("--out-dir", required=True, help="Output dir for plots")
    parser.add_argument(
        "--log-json-glob",
        default="",
        help="Optional glob for mmcv json logs (e.g. 'work_dirs/exp_*/*.log.json')",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_csv).resolve()
    if not metrics_path.is_file():
        _err(f"metrics csv not found: {metrics_path}")
        return 2

    df = pd.read_csv(metrics_path)
    if df.empty:
        _err("metrics.csv is empty")
        return 2

    if "mAP" in df.columns:
        df["mAP"] = pd.to_numeric(df["mAP"], errors="coerce")

    _plot_map(df, out_dir)

    log_glob = str(args.log_json_glob).strip()
    if log_glob:
        paths = [Path(p).resolve() for p in glob.glob(log_glob)]
        paths = [p for p in paths if p.is_file()]
        _plot_training_curves(sorted(paths), out_dir)

    _log(f"wrote plots under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

