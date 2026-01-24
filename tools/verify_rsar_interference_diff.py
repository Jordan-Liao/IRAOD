from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _log(msg: str) -> None:
    print(f"[verify_rsar_interference_diff] {msg}")


def _stable_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _images_dir_for(split_dir: Path, corrupt: str) -> Path:
    c = (corrupt or "").strip()
    if c in ("", "clean", "none"):
        return split_dir / "images"
    return split_dir / f"images-{c}"


def _index_images(img_dir: Path) -> dict[str, list[Path]]:
    stem2: dict[str, list[Path]] = {}
    for root, _dirs, files in os.walk(img_dir):
        for fn in files:
            p = Path(root) / fn
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            stem2.setdefault(p.stem, []).append(p)
    return stem2


def _pick_one(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    prio = {".jpg": 0, ".jpeg": 1, ".png": 2, ".bmp": 3, ".tif": 4, ".tiff": 5}

    def keyfn(p: Path):
        return prio.get(p.suffix.lower(), 99), p.name

    return sorted(paths, key=keyfn)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="RSAR root containing train/val/test")
    parser.add_argument("--corrupt", required=True, help="e.g. interf_jamA")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    ann_dir = split_dir / "annfiles"
    clean_dir = _images_dir_for(split_dir, "clean")
    corrupt_dir = _images_dir_for(split_dir, args.corrupt)

    if not ann_dir.is_dir():
        raise SystemExit(f"missing: {ann_dir}")
    if not clean_dir.is_dir():
        raise SystemExit(f"missing: {clean_dir}")
    if not corrupt_dir.is_dir():
        raise SystemExit(f"missing: {corrupt_dir}")

    clean_idx = _index_images(clean_dir)
    corrupt_idx = _index_images(corrupt_dir)

    ann_files = sorted([p for p in ann_dir.glob("*.txt") if p.is_file()])
    if not ann_files:
        raise SystemExit(f"no annfiles under: {ann_dir}")

    rng = np.random.RandomState(int(args.seed) ^ _stable_seed(str(split_dir), str(args.corrupt), str(args.seed)))
    rng.shuffle(ann_files)
    picked = ann_files[: min(int(args.samples), len(ann_files))]

    checked = 0
    identical = 0
    missing = 0
    mean_abs_diffs: list[float] = []

    for ann in picked:
        stem = ann.stem
        a = _pick_one(clean_idx.get(stem, []))
        b = _pick_one(corrupt_idx.get(stem, []))
        if a is None or b is None:
            missing += 1
            continue
        ia = _read(a)
        ib = _read(b)
        if ia.shape != ib.shape:
            _log(f"WARNING: shape mismatch for {stem}: clean={ia.shape} corrupt={ib.shape}")
            continue
        checked += 1
        if np.array_equal(ia, ib):
            identical += 1
            continue
        mean_abs_diffs.append(float(np.mean(np.abs(ia.astype(np.float32) - ib.astype(np.float32)))))

    if checked == 0:
        raise SystemExit("no valid pairs checked (are images missing?)")

    mean_diff = float(np.mean(mean_abs_diffs)) if mean_abs_diffs else 0.0
    _log(
        f"split={args.split} corrupt={args.corrupt} checked={checked} missing={missing} identical={identical} "
        f"mean_abs_diff={mean_diff:.4f}"
    )

    if identical == checked:
        raise SystemExit("all checked pairs are identical (no interference applied?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

