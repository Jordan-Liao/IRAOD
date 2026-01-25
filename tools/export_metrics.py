from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[export_metrics] {msg}")


def _infer_dataset(work_dir: Path) -> str:
    name = work_dir.name.lower()
    if "dior" in name:
        return "DIOR"
    if "rsar" in name:
        return "RSAR"
    return ""


def _infer_corrupt(work_dir: Path) -> str:
    name = work_dir.name.lower()
    for c in ["cloudy", "brightness", "contrast", "clean"]:
        if c in name:
            return c
    # RSAR interference / custom corruptions (e.g. interf_jamA, interf_jamB_s3).
    m = re.search(r"(interf_[a-z0-9]+(?:_[a-z0-9]+)*)", name)
    if m:
        return m.group(1)
    return "clean"


def _infer_method(work_dir: Path) -> str:
    name = work_dir.name.lower()
    if "baseline" in name:
        return "baseline"
    if "unbiased" in name or "ut" in name:
        no_cga_markers = ["nocga", "no_cga", "no-cga", "cgaoff", "cga_off"]
        if "cga" in name and not any(m in name for m in no_cga_markers):
            return "ut+cga"
        return "ut"
    if "cga" in name:
        return "cga"
    return ""


def _infer_seed(work_dir: Path) -> int | None:
    # Only accept explicit `seedNN` patterns. Avoid matching corruption severities
    # like `interf_jamB_s3` (the `_s3` is not a random seed).
    m = re.search(r"(?:^|[_-])seed(\\d+)(?:$|[_-])", work_dir.name.lower())
    if not m:
        return 42  # train.py hard-codes seed=42 in this repo
    try:
        return int(m.group(1))
    except Exception:
        return 42


def _latest_eval_json(eval_dir: Path) -> Path | None:
    cands = sorted(eval_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _collect_eval_dirs(work_dir: Path) -> list[Path]:
    out: list[Path] = []
    if any(work_dir.glob("eval_*.json")):
        out.append(work_dir)

    for p in sorted(work_dir.glob("eval_*")):
        if not p.is_dir():
            continue
        if any(p.glob("eval_*.json")):
            out.append(p)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dirs",
        nargs="+",
        required=True,
        help="One or more work-dir globs (e.g. work_dirs/exp_*)",
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    work_dirs: list[Path] = []
    for pat in args.work_dirs:
        work_dirs.extend([p for p in Path().glob(pat) if p.is_dir()])

    rows: list[dict[str, object]] = []
    for wd in sorted(set(work_dirs)):
        method = _infer_method(wd)
        dataset = _infer_dataset(wd)
        seed = _infer_seed(wd)

        for eval_dir in _collect_eval_dirs(wd):
            eval_json = _latest_eval_json(eval_dir)
            if eval_json is None:
                continue

            payload = json.loads(eval_json.read_text(encoding="utf-8"))
            metric = payload.get("metric", {})
            mAP = metric.get("mAP", None) if isinstance(metric, dict) else None

            corrupt = _infer_corrupt(eval_dir if eval_dir != wd else wd)
            run_id = wd.name if eval_dir == wd else f"{wd.name}/{eval_dir.name}"

            rows.append(
                dict(
                    exp_id=wd.name,
                    run_id=run_id,
                    method=method,
                    dataset=dataset,
                    corrupt=corrupt,
                    seed=seed,
                    work_dir=str(wd),
                    eval_dir=str(eval_dir),
                    eval_json=str(eval_json),
                    mAP=mAP,
                )
            )

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        _log("no eval_*.json found; wrote empty CSV")
    df.to_csv(out_path, index=False)
    _log(f"wrote: {out_path} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
