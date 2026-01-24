from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _log(msg: str) -> None:
    print(f"[vis_random_samples] {msg}")


def _err(msg: str) -> None:
    print(f"[vis_random_samples] ERROR: {msg}", file=sys.stderr)


def _list_images(dir_path: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out: dict[str, Path] = {}
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        out[p.name] = p
    return out


def _safe_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return None  # type: ignore[return-value]


def _hstack_with_header(images: list[Image.Image], labels: list[str], title: str) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images/labels length mismatch")
    if not images:
        raise ValueError("no images to stack")

    target_h = min(im.height for im in images)
    resized: list[Image.Image] = []
    for im in images:
        if im.height == target_h:
            resized.append(im)
            continue
        w = max(1, int(im.width * (target_h / im.height)))
        resized.append(im.resize((w, target_h), resample=Image.BILINEAR))

    pad = 6
    header_h = 40
    total_w = sum(im.width for im in resized) + pad * (len(resized) + 1)
    total_h = header_h + target_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _safe_font()

    draw.text((pad, 6), title, fill=(240, 240, 240), font=font)

    x = pad
    for im, lab in zip(resized, labels, strict=True):
        draw.text((x, header_h - 18), lab, fill=(220, 220, 220), font=font)
        canvas.paste(im, (x, header_h))
        x += im.width + pad

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-dirs", nargs="+", required=True, help="One or more show-dir folders")
    parser.add_argument("--num", type=int, default=8, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels for each vis-dir (defaults to folder name)",
    )
    args = parser.parse_args()

    vis_dirs = [Path(p).resolve() for p in args.vis_dirs]
    for d in vis_dirs:
        if not d.is_dir():
            _err(f"not a dir: {d}")
            return 2

    labels = list(args.labels)
    if labels and len(labels) != len(vis_dirs):
        _err("--labels length must match --vis-dirs length (or omit --labels)")
        return 2
    if not labels:
        labels = [d.name for d in vis_dirs]

    pools = [_list_images(d) for d in vis_dirs]
    common_names = set(pools[0].keys())
    for p in pools[1:]:
        common_names &= set(p.keys())

    common = sorted(common_names)
    if not common:
        _err("no common image filenames across all --vis-dirs")
        return 3

    if args.num <= 0:
        _err("--num must be > 0")
        return 2

    rng = random.Random(args.seed)
    picks = rng.sample(common, k=min(args.num, len(common)))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(picks):
        ims: list[Image.Image] = []
        for p in pools:
            ims.append(Image.open(p[name]).convert("RGB"))

        title = f"{i+1}/{len(picks)}  {name}"
        stacked = _hstack_with_header(ims, labels, title)
        out_path = out_dir / f"sample_{i+1:03d}_{Path(name).stem}.png"
        stacked.save(out_path)

    _log(f"wrote: {out_dir} samples={len(picks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

