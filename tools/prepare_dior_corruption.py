from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _log(msg: str) -> None:
    print(f"[prepare_dior_corruption] {msg}")


def _stable_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _read_ids(split_files: Iterable[Path]) -> list[str]:
    ids: list[str] = []
    for p in split_files:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def _ensure_clean_symlink(data_root: Path) -> None:
    src = (data_root / "JPEGImages").resolve()
    dst = data_root / "Corruption" / "JPEGImages-clean"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            return
        dst.unlink()
    dst.symlink_to(src)


def _apply_brightness(img: Image.Image, *, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def _apply_contrast(img: Image.Image, *, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _apply_cloudy(img: Image.Image, *, seed: int, alpha: float) -> Image.Image:
    rng = np.random.RandomState(seed)
    w, h = img.size

    noise = (rng.rand(h, w) * 255.0).astype(np.uint8)
    noise_img = Image.fromarray(noise, mode="L").filter(ImageFilter.GaussianBlur(radius=max(8, int(min(h, w) * 0.02))))
    m = np.array(noise_img).astype(np.float32) / 255.0

    # Create a soft cloud mask with a gentle threshold.
    t1, t2 = 0.45, 0.85
    mask = np.clip((m - t1) / (t2 - t1), 0.0, 1.0)
    mask = (mask * alpha).astype(np.float32)

    arr = np.array(img.convert("RGB")).astype(np.float32)
    out = arr * (1.0 - mask[..., None]) + 255.0 * mask[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _process_one(
    *,
    src_img: Path,
    dst_img: Path,
    corrupt: str,
    strength: float,
    seed: int,
) -> tuple[str, str]:
    if dst_img.is_file():
        return "skip", str(dst_img)
    dst_img.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_img).convert("RGB")
    if corrupt == "brightness":
        out = _apply_brightness(img, factor=strength)
    elif corrupt == "contrast":
        out = _apply_contrast(img, factor=strength)
    elif corrupt == "cloudy":
        out = _apply_cloudy(img, seed=seed, alpha=strength)
    else:
        raise ValueError(f"unknown corrupt: {corrupt}")

    tmp = dst_img.with_suffix(dst_img.suffix + ".tmp")
    out.save(tmp, format="JPEG", quality=95)
    tmp.replace(dst_img)
    return "ok", str(dst_img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/DIOR", help="DIOR root (contains JPEGImages/ and ImageSets/)")
    parser.add_argument(
        "--corrupt",
        nargs="+",
        required=True,
        help="Corrupt types: clean, cloudy, brightness, contrast",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated splits under ImageSets/ (e.g. val,test)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    # Strength meanings:
    # - brightness/contrast: PIL enhance factor
    # - cloudy: alpha blending (0..1)
    parser.add_argument("--brightness-factor", type=float, default=1.4)
    parser.add_argument("--contrast-factor", type=float, default=1.4)
    parser.add_argument("--cloudy-alpha", type=float, default=0.55)

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()

    jpeg_root = data_root / "JPEGImages"
    if not jpeg_root.is_dir():
        raise SystemExit(f"missing: {jpeg_root}")

    split_names = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    split_files = [data_root / "ImageSets" / f"{s}.txt" for s in split_names]
    missing = [str(p) for p in split_files if not p.is_file()]
    if missing:
        raise SystemExit(f"missing split files: {missing}")

    image_ids = _read_ids(split_files)
    if not image_ids:
        raise SystemExit("no image ids found in splits")

    corrupts = [c.strip().lower() for c in args.corrupt]
    supported = {"clean", "cloudy", "brightness", "contrast"}
    unknown = [c for c in corrupts if c not in supported]
    if unknown:
        raise SystemExit(f"unknown --corrupt: {unknown} (supported: {sorted(supported)})")

    # Ensure Corruption root exists.
    (data_root / "Corruption").mkdir(parents=True, exist_ok=True)

    if "clean" in corrupts:
        _log("prepare: JPEGImages-clean (symlink to JPEGImages)")
        _ensure_clean_symlink(data_root)

    jobs = []
    total_missing_src = 0
    for cid in image_ids:
        src = jpeg_root / f"{cid}.jpg"
        if not src.is_file():
            total_missing_src += 1
            continue

        for c in corrupts:
            if c == "clean":
                continue
            dst_dir = data_root / "Corruption" / f"JPEGImages-{c}"
            dst = dst_dir / f"{cid}.jpg"

            if c == "brightness":
                strength = float(args.brightness_factor)
            elif c == "contrast":
                strength = float(args.contrast_factor)
            elif c == "cloudy":
                strength = float(args.cloudy_alpha)
            else:
                raise AssertionError(c)

            jobs.append(
                dict(
                    src_img=src,
                    dst_img=dst,
                    corrupt=c,
                    strength=strength,
                    seed=int(args.seed) ^ _stable_seed(str(src), c, str(args.seed)),
                )
            )

    if total_missing_src:
        _log(f"WARNING: missing source jpg for {total_missing_src} ids (skipped)")

    if not jobs:
        _log("nothing to do (all requested corruptions already present)")
        return 0

    _log(f"generate: jobs={len(jobs)} workers={args.workers}")
    ok = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_process_one, **j) for j in jobs]
        for fut in as_completed(futs):
            try:
                status, _path = fut.result()
            except Exception as e:
                failed += 1
                _log(f"ERROR: {e}")
                continue
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1

    _log(f"done: ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

