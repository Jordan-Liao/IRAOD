import argparse
import sys
from pathlib import Path


def _err(msg: str) -> None:
    print(f"[verify_dataset_layout] ERROR: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[verify_dataset_layout] WARNING: {msg}", file=sys.stderr)


def _require_dir(path: Path, what: str) -> bool:
    if not path.exists():
        _err(f"missing {what}: {path}")
        return False
    if not path.is_dir():
        _err(f"not a directory for {what}: {path}")
        return False
    return True


def _has_any_file(path: Path, patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if any(path.glob(pat)):
            return True
    return False


def verify_dior(dior_root: Path, *, require_corruption: bool) -> bool:
    ok = True
    ok &= _require_dir(dior_root, "DIOR root")

    ann_dir = dior_root / "Annotations"
    jpeg_dir = dior_root / "JPEGImages"
    sets_dir = dior_root / "ImageSets"

    ok &= _require_dir(ann_dir, "DIOR Annotations")
    ok &= _require_dir(jpeg_dir, "DIOR JPEGImages")
    ok &= _require_dir(sets_dir, "DIOR ImageSets")

    # image sets
    for name in ("train.txt", "val.txt", "test.txt"):
        p = sets_dir / name
        if not p.exists():
            ok = False
            _err(f"missing DIOR split file: {p}")

    # annotations: either directly under Annotations/*.xml, or under
    # Annotations/Oriented Bounding Boxes/*.xml (common DIOR structure).
    obb_dir = ann_dir / "Oriented Bounding Boxes"
    if obb_dir.exists() and obb_dir.is_dir():
        if not _has_any_file(obb_dir, ("*.xml", "*.XML")):
            ok = False
            _err(f"no XML found under: {obb_dir}")
    else:
        if not _has_any_file(ann_dir, ("*.xml", "*.XML")):
            ok = False
            _err(f"no XML found under: {ann_dir} (and missing {obb_dir})")

    # images
    if not _has_any_file(jpeg_dir, ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")):
        ok = False
        _err(f"no JPG found under: {jpeg_dir}")

    # corruption (optional)
    corr_dir = dior_root / "Corruption"
    if corr_dir.exists():
        jpg_corr = list(corr_dir.glob("JPEGImages-*"))
        if not jpg_corr:
            _warn(f"found {corr_dir} but no 'JPEGImages-*' subdirs")
    elif require_corruption:
        ok = False
        _err(f"missing required DIOR corruption dir: {corr_dir}")

    return ok


def verify_rsar(rsar_root: Path) -> bool:
    ok = True
    ok &= _require_dir(rsar_root, "RSAR root")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for split in ("train", "val", "test"):
        split_root = rsar_root / split
        ann_dir = split_root / "annfiles"
        img_dir = split_root / "images"
        ok &= _require_dir(split_root, f"RSAR split {split}")
        ok &= _require_dir(ann_dir, f"RSAR {split}/annfiles")
        ok &= _require_dir(img_dir, f"RSAR {split}/images")

        ann_files = list(ann_dir.glob("*.txt"))
        if not ann_files:
            ok = False
            _err(f"no annfiles (*.txt) under: {ann_dir}")

        if not any(img_dir.rglob(f"*{e}") for e in exts):
            ok = False
            _err(f"no images ({', '.join(exts)}) under: {img_dir}")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dior", required=True, help="DIOR root directory")
    parser.add_argument("--rsar", required=True, help="RSAR root directory")
    parser.add_argument(
        "--require-dior-corruption",
        action="store_true",
        help="Fail if DIOR Corruption/ does not exist",
    )
    args = parser.parse_args()

    dior_root = Path(args.dior)
    rsar_root = Path(args.rsar)

    ok_dior = verify_dior(dior_root, require_corruption=args.require_dior_corruption)
    ok_rsar = verify_rsar(rsar_root)
    if ok_dior and ok_rsar:
        print("[verify_dataset_layout] OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

