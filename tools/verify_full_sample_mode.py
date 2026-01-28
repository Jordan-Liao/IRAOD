import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        return
    errors.append(message)


def _count_annfiles(dir_path: Path) -> int:
    return sum(1 for p in dir_path.glob("*.txt") if p.is_file())


def _check_script_smoke_default(path: Path, *, expect_default: str = "0") -> dict:
    text = _read_text(path)
    pat = rf'^\s*SMOKE="\$\{{SMOKE:-{re.escape(expect_default)}\}}"\s*$'
    ok_default = bool(re.search(pat, text, flags=re.MULTILINE))
    ok_gate = ("if [[ \"${SMOKE}\" == \"1\" ]]" in text) or bool(
        re.search(r'if\s+\[\[\s*"\$\{SMOKE\}"\s*==\s*"1"\s*\]\]', text)
    )
    return {
        "path": str(path),
        "smoke_default": expect_default,
        "smoke_default_ok": ok_default,
        "has_smoke_gate": ok_gate,
    }


def _check_file_contains(path: Path, needles: list[str]) -> dict:
    text = _read_text(path)
    missing = [s for s in needles if s not in text]
    return {"path": str(path), "missing": missing}


def _infer_rsar_split(path_str: str) -> str | None:
    p = str(path_str).replace("\\", "/")
    for split in ("train", "val", "test"):
        if f"/{split}/" in p:
            return split
    return None


def _apply_rsar_data_root(cfg, data_root: Path) -> None:
    if cfg.get("data", None) is None:
        return
    root = data_root.expanduser().resolve()
    for split_key in ("train", "val", "test"):
        if split_key not in cfg.data:
            continue
        ds = cfg.data[split_key]
        if not isinstance(ds, dict):
            continue
        for field in ("ann_file", "ann_file_u", "img_prefix", "img_prefix_u"):
            if field not in ds or not isinstance(ds[field], str):
                continue
            split = _infer_rsar_split(ds[field])
            if split is None:
                continue
            subdir = "annfiles" if field.startswith("ann_") else "images"
            ds[field] = str(root / split / subdir) + "/"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="RSAR config path")
    parser.add_argument("--data-root", required=True, help="RSAR dataset root containing train/val/test")
    parser.add_argument(
        "--out",
        default="work_dirs/sanity/full_sample_mode.json",
        help="Output JSON path (default: work_dirs/sanity/full_sample_mode.json)",
    )
    parser.add_argument(
        "--min-annfiles",
        type=int,
        default=51,
        help="Fail if any split has <= this many annfiles (default: 51)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    data_root = Path(args.data_root).expanduser().resolve()
    out_path = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    checks: dict[str, object] = {}

    # 1) Script defaults: exp scripts should be full-data by default (SMOKE=0).
    checks["exp_rsar_ut"] = _check_script_smoke_default(repo_root / "scripts/exp_rsar_ut.sh", expect_default="0")
    checks["exp_rsar_baseline"] = _check_script_smoke_default(
        repo_root / "scripts/exp_rsar_baseline.sh", expect_default="0"
    )

    _check(checks["exp_rsar_ut"]["smoke_default_ok"], "scripts/exp_rsar_ut.sh: SMOKE default is not 0", errors)
    _check(
        checks["exp_rsar_baseline"]["smoke_default_ok"],
        "scripts/exp_rsar_baseline.sh: SMOKE default is not 0",
        errors,
    )

    # 2) Train/Test CLI args exist (short command style).
    cli_needles = ["--data-root", "--cga-scorer", "--sarclip-model", "--sarclip-pretrained"]
    checks["train_py_args"] = _check_file_contains(repo_root / "train.py", cli_needles)
    checks["test_py_args"] = _check_file_contains(repo_root / "test.py", cli_needles)
    _check(not checks["train_py_args"]["missing"], f"train.py missing args: {checks['train_py_args']['missing']}", errors)
    _check(not checks["test_py_args"]["missing"], f"test.py missing args: {checks['test_py_args']['missing']}", errors)

    # 3) Config uses mmcv's {{ fileDirname }} (robust against temp copy execution).
    cfg_text = _read_text(config_path)
    checks["config_fileDirname"] = {"path": str(config_path), "has_fileDirname": "{{ fileDirname }}" in cfg_text}
    _check(checks["config_fileDirname"]["has_fileDirname"], f"{config_path}: missing {{ fileDirname }}", errors)

    # 4) Dataset root and full-sample sizes.
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ann_dir = data_root / split / "annfiles"
        img_dir = data_root / split / "images"
        _check(ann_dir.is_dir(), f"missing ann dir: {ann_dir}", errors)
        _check(img_dir.is_dir(), f"missing img dir: {img_dir}", errors)
        split_counts[split] = _count_annfiles(ann_dir) if ann_dir.is_dir() else 0
        _check(
            split_counts[split] >= args.min_annfiles,
            f"{split}: annfiles too small ({split_counts[split]}) under {ann_dir}",
            errors,
        )
    checks["rsar_annfile_counts"] = {
        "data_root": str(data_root),
        "counts": split_counts,
        "min_required": args.min_annfiles,
    }

    # 5) Config resolution + --data-root rewrite (same behavior as train.py/test.py).
    t0 = time.time()
    # Ensure datasets are registered (SemiDOTADataset, etc.).
    # Importing sfod triggers registration in sfod/__init__.py.
    import sfod  # noqa: F401

    from mmcv import Config
    from mmrotate.datasets import build_dataset

    cfg = Config.fromfile(str(config_path))
    _apply_rsar_data_root(cfg, data_root)
    resolved_paths: dict[str, dict[str, str]] = {}
    for split_key in ("train", "val", "test"):
        ds = cfg.data.get(split_key)
        if not isinstance(ds, dict):
            continue
        resolved_paths[split_key] = {}
        for field in ("ann_file", "ann_file_u", "img_prefix", "img_prefix_u"):
            v = ds.get(field)
            if isinstance(v, str):
                resolved_paths[split_key][field] = v

    # Build datasets to confirm the config is runnable and uses full splits.
    ds_train = build_dataset(cfg.data.train)
    ds_val = build_dataset(cfg.data.val)
    ds_test = build_dataset(cfg.data.test)
    lens = {"train": len(ds_train), "val": len(ds_val), "test": len(ds_test)}
    dt = time.time() - t0

    checks["config_paths_after_data_root"] = resolved_paths
    checks["dataset_lens"] = {"lens": lens, "seconds": round(dt, 3)}

    # 6) Sampler-critical invariant: if dataset exposes `flag`, it must match len(dataset).
    # Otherwise group samplers may silently shorten an epoch and break "full-sample" intent.
    flag_info = {}
    for name, ds in (("train", ds_train), ("val", ds_val), ("test", ds_test)):
        flag = getattr(ds, "flag", None)
        if flag is None:
            flag_info[name] = {"has_flag": False}
            continue
        try:
            flag_len = len(flag)
        except Exception:
            flag_len = None
        flag_info[name] = {"has_flag": True, "flag_len": flag_len, "dataset_len": len(ds)}
        if flag_len is not None:
            _check(flag_len == len(ds), f"{name} dataset flag_len={flag_len} != dataset_len={len(ds)}", errors)
    checks["dataset_flag_lens"] = flag_info

    _check(lens["train"] >= args.min_annfiles, f"train dataset len too small: {lens['train']}", errors)
    _check(lens["val"] >= args.min_annfiles, f"val dataset len too small: {lens['val']}", errors)
    _check(lens["test"] >= args.min_annfiles, f"test dataset len too small: {lens['test']}", errors)

    result = {
        "ok": not errors,
        "repo_root": str(repo_root),
        "config": str(config_path),
        "data_root": str(data_root),
        "cwd": os.getcwd(),
        "checks": checks,
        "errors": errors,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if errors:
        print("[verify_full_sample_mode] FAIL")
        for e in errors:
            print(f"[verify_full_sample_mode] ERROR: {e}", file=sys.stderr)
        print(f"[verify_full_sample_mode] wrote: {out_path}")
        return 1

    print("[verify_full_sample_mode] OK")
    print(f"[verify_full_sample_mode] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
