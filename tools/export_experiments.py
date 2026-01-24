from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[export_experiments] {msg}")


def _err(msg: str) -> None:
    print(f"[export_experiments] ERROR: {msg}", file=sys.stderr)


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _latest_in_dir(dir_path: Path, pattern: str) -> Path | None:
    cands = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _infer_script_hint(exp_id: str) -> str:
    mapping = {
        "exp_smoke_dior": "bash scripts/smoke_dior.sh",
        "exp_smoke_rsar": "bash scripts/smoke_rsar.sh",
        "exp_dior_baseline_eval": "bash scripts/exp_dior_baseline_eval.sh",
        "exp_dior_ut": "bash scripts/exp_dior_ut.sh",
        "exp_dior_ut_cga_clip": "bash scripts/exp_dior_ut_cga_clip.sh",
        "exp_rsar_baseline": "bash scripts/exp_rsar_baseline.sh",
        "exp_rsar_ut_nocga": "CGA_SCORER=none WORK_DIR=work_dirs/exp_rsar_ut_nocga bash scripts/exp_rsar_ut.sh",
        "exp_rsar_ut_cga_clip": "CGA_SCORER=clip WORK_DIR=work_dirs/exp_rsar_ut_cga_clip bash scripts/exp_rsar_ut.sh",
    }
    return mapping.get(exp_id, "")


def _infer_config_hint(exp_id: str) -> str:
    mapping = {
        "exp_dior_baseline_eval": "configs/experiments/dior/baseline_oriented_rcnn_dior.py",
        "exp_dior_ut": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining.py",
        "exp_dior_ut_cga_clip": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py",
        "exp_rsar_baseline": "configs/experiments/rsar/baseline_oriented_rcnn_rsar.py",
        "exp_rsar_ut_nocga": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py",
        "exp_rsar_ut_cga_clip": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py",
        "exp_smoke_dior": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py",
        "exp_smoke_rsar": "configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py",
    }
    return mapping.get(exp_id, "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, help="Input metrics.csv")
    parser.add_argument("--out-csv", required=True, help="Output experiments.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = Path(args.metrics_csv).resolve()
    if not metrics_path.is_file():
        _err(f"metrics csv not found: {metrics_path}")
        return 2

    df = pd.read_csv(metrics_path)
    if df.empty:
        _err("metrics.csv is empty")
        return 2

    git_sha = _git_sha(repo_root)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def enrich_row(row: pd.Series) -> pd.Series:
        work_dir = Path(str(row.get("work_dir", ""))).resolve()
        exp_id = str(row.get("exp_id", ""))

        config_copy = ""
        if work_dir.is_dir():
            py = [p for p in work_dir.glob("*.py") if p.is_file()]
            if len(py) == 1:
                config_copy = str(py[0])
            elif py:
                # Prefer configs copied by train.py (usually a single top-level config)
                scored = []
                for p in py:
                    score = 0
                    n = p.name.lower()
                    if "unbiased_teacher" in n or "baseline" in n:
                        score += 10
                    score += min(int(p.stat().st_size / 1024), 50)
                    scored.append((score, p))
                scored.sort(key=lambda t: t[0], reverse=True)
                config_copy = str(scored[0][1])

        ckpt = ""
        if work_dir.is_dir():
            latest = work_dir / "latest.pth"
            if latest.is_file():
                ckpt = str(latest)

        log_json = ""
        log_txt = ""
        if work_dir.is_dir():
            p = _latest_in_dir(work_dir, "*.log.json")
            if p is not None:
                log_json = str(p)
            p = _latest_in_dir(work_dir, "*.log")
            if p is not None:
                log_txt = str(p)

        row["git_sha"] = git_sha
        row["generated_at"] = generated_at
        row["script_hint"] = _infer_script_hint(exp_id)
        row["config_hint"] = _infer_config_hint(exp_id)
        row["config_copy"] = config_copy
        row["train_ckpt"] = ckpt
        row["train_log_json"] = log_json
        row["train_log"] = log_txt
        return row

    df = df.apply(enrich_row, axis=1)

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(f"wrote: {out_path} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

