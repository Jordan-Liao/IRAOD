from __future__ import annotations

import argparse
import os
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

SPLITS = ["train", "val", "test"]


def _log(msg: str) -> None:
    print(f"[prepare_rsar_sfodrs_corruptions_layout] {msg}")


def _rel_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return
    rel = os.path.relpath(str(target), str(link.parent))
    link.symlink_to(rel, target_is_directory=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rsar-root",
        default=None,
        help="Path to dataset/RSAR (defaults to ./dataset/RSAR relative to repo root).",
    )
    ap.add_argument("--corr", default=None, help="Single corruption name (default: all 7)")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rsar_root = Path(args.rsar_root).expanduser().resolve() if args.rsar_root else (repo_root / "dataset" / "RSAR")
    if not rsar_root.is_dir():
        raise SystemExit(f"RSAR root not found: {rsar_root}")

    corrs = [args.corr] if args.corr else CORRUPTIONS

    n_linked = 0
    n_copied = 0
    n_missing = 0

    for corr in corrs:
        for split in SPLITS:
            dst = rsar_root / "corruptions" / corr / split / "images"
            if dst.is_dir() or dst.is_symlink():
                continue

            legacy = rsar_root / split / f"images-{corr}"
            if not legacy.is_dir():
                n_missing += 1
                _log(f"missing: {dst} (no legacy dir {legacy})")
                continue

            if args.dry_run:
                _log(f"would {'copy' if args.copy else 'symlink'}: {dst} -> {legacy}")
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                # Copy directory tree (can be large; symlink preferred).
                import shutil

                shutil.copytree(str(legacy), str(dst))
                n_copied += 1
                _log(f"copied: {dst} <- {legacy}")
            else:
                _rel_symlink(legacy, dst)
                n_linked += 1
                _log(f"symlink: {dst} -> {legacy}")

    _log(f"done: linked={n_linked} copied={n_copied} missing={n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

