from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


CORRUPTIONS = [
    "chaff",
    "gaussian_white_noise",
    "point_target",
    "noise_suppression",
    "am_noise_horizontal",
    "smart_suppression",
    "am_noise_vertical",
]


def _log(msg: str) -> None:
    print(f"[collect_rsar_sfodrs_results] {msg}")


def _latest_eval_json(d: Path) -> Path | None:
    cands = sorted(d.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _read_map(eval_json: Path) -> float | None:
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    metric = payload.get("metric", None)
    if not isinstance(metric, dict):
        return None
    m = metric.get("mAP", None)
    try:
        return float(m)
    except Exception:
        return None


def _fmt(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.4f}"


@dataclass(frozen=True)
class _Row:
    name: str
    values: dict[str, float | None]  # domain -> mAP


def _mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _resolve_workdir(work_root: Path, *, corr: str, kind: str) -> Path:
    # Keep in sync with scripts/exp_rsar_sfodrs_adapt.sh and scripts/run_rsar_sfodrs_7corr.sh
    if kind == "source_clean_test":
        return work_root / "clean" / "source_clean_test"
    if kind == "direct_test":
        return work_root / corr / "direct_test"
    if kind == "self_training":
        return work_root / corr / "self_training" / "eval_target"
    if kind == "self_training_plus_cga":
        return work_root / corr / "self_training_plus_cga" / "eval_target"
    raise ValueError(f"unknown kind={kind}")


def _collect_one(work_root: Path, *, corr: str, kind: str) -> float | None:
    d = _resolve_workdir(work_root, corr=corr, kind=kind)
    if not d.is_dir():
        return None
    j = _latest_eval_json(d)
    if j is None:
        return None
    return _read_map(j)


def _write_csv(rows: list[_Row], out_csv: Path) -> None:
    cols = ["row", "clean", *CORRUPTIONS, "mean"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            clean_v = r.values.get("clean")
            corr_vs = [r.values.get(c) for c in CORRUPTIONS]
            mean_v = _mean([clean_v, *corr_vs])
            w.writerow([r.name, _fmt(clean_v), *[_fmt(v) for v in corr_vs], _fmt(mean_v)])


def _write_md(rows: list[_Row], out_md: Path) -> None:
    cols = ["row", "clean", *CORRUPTIONS, "mean"]
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def cell(v: str) -> str:
        return v if v != "" else "-"

    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        clean_v = r.values.get("clean")
        corr_vs = [r.values.get(c) for c in CORRUPTIONS]
        mean_v = _mean([clean_v, *corr_vs])
        row = [r.name, _fmt(clean_v), *[_fmt(v) for v in corr_vs], _fmt(mean_v)]
        lines.append("| " + " | ".join([cell(x) for x in row]) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True, help="Root work dir created by scripts/run_rsar_sfodrs_7corr.sh")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    work_root = Path(args.work_root).expanduser().resolve()
    if not work_root.is_dir():
        raise SystemExit(f"work_root not found: {work_root}")

    clean_map = _collect_one(work_root, corr="clean", kind="source_clean_test")
    if clean_map is None:
        _log("WARNING: missing clean source_clean_test eval json")

    rows: list[_Row] = []

    # Row 1: source_clean_test (only clean column is meaningful, but keep full shape)
    rows.append(_Row("source_clean_test", {"clean": clean_map, **{c: None for c in CORRUPTIONS}}))

    # Row 2: direct_test
    direct_vals = {"clean": clean_map}
    for c in CORRUPTIONS:
        direct_vals[c] = _collect_one(work_root, corr=c, kind="direct_test")
    rows.append(_Row("direct_test", direct_vals))

    # Row 3: self_training
    st_vals = {"clean": clean_map}
    for c in CORRUPTIONS:
        st_vals[c] = _collect_one(work_root, corr=c, kind="self_training")
    rows.append(_Row("self_training", st_vals))

    # Row 4: self_training_plus_cga
    stcga_vals = {"clean": clean_map}
    for c in CORRUPTIONS:
        stcga_vals[c] = _collect_one(work_root, corr=c, kind="self_training_plus_cga")
    rows.append(_Row("self_training_plus_cga", stcga_vals))

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    _write_csv(rows, out_csv)
    _write_md(rows, out_md)

    _log(f"wrote: {out_csv}")
    _log(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

