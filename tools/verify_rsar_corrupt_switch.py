import argparse
import csv
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path


def _pick_by_priority(paths: list[str]) -> str | None:
    if not paths:
        return None
    prio = {".jpg": 0, ".jpeg": 1, ".png": 2, ".bmp": 3, ".tif": 4, ".tiff": 5}

    def keyfn(p: str):
        ext = osp.splitext(p)[1].lower()
        return prio.get(ext, 99), p

    return sorted(paths, key=keyfn)[0]


def build_index(img_dir: Path, exts: tuple[str, ...]) -> dict[str, list[str]]:
    stem2paths: dict[str, list[str]] = defaultdict(list)
    for root, _dirs, files in os.walk(img_dir):
        for fn in files:
            ext = osp.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            stem = osp.splitext(fn)[0]
            full = osp.join(root, fn)
            stem2paths[stem].append(str(Path(full).resolve()))
    return stem2paths


def _images_dir_for(split_dir: Path, corrupt: str) -> Path:
    corrupt = (corrupt or "").strip()
    if corrupt in ("", "clean", "none"):
        return split_dir / "images"
    return split_dir / f"images-{corrupt}"


def _check_split(
    *,
    split: str,
    ann_dir: Path,
    img_dir: Path,
    out_csv: Path,
    exts: tuple[str, ...],
) -> tuple[int, int, int]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stem2paths = build_index(img_dir, exts)
    ann_files = sorted([p for p in ann_dir.glob("*.txt") if p.is_file()])
    missing = 0
    conflict = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "annfile",
                "image_stem",
                "status",
                "resolved_image_path",
                "candidates",
            ],
        )
        w.writeheader()
        for ann_path in ann_files:
            stem = ann_path.stem
            candidates = stem2paths.get(stem, [])
            if not candidates:
                status = "missing"
                missing += 1
                resolved = ""
            elif len(candidates) == 1:
                status = "ok"
                resolved = candidates[0]
            else:
                status = "conflict"
                conflict += 1
                resolved = _pick_by_priority(candidates) or candidates[0]

            w.writerow(
                dict(
                    split=split,
                    annfile=str(ann_path),
                    image_stem=stem,
                    status=status,
                    resolved_image_path=resolved,
                    candidates="|".join(candidates),
                )
            )

    total = len(ann_files)
    print(f"[verify_rsar_corrupt_switch] split={split} total={total} missing={missing} conflict={conflict}")
    print(f"[verify_rsar_corrupt_switch] wrote {out_csv}")
    return total, missing, conflict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="RSAR root containing train/val/test")
    parser.add_argument("--corrupt", default="clean", help="clean or interf_xxx (maps to images-interf_xxx/)")
    parser.add_argument("--out-dir", default="work_dirs/sanity/rsar_corrupt_switch", help="Report output dir")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="comma-separated splits to verify (default: train,val,test)",
    )
    parser.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="comma-separated allowed extensions",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    splits = tuple(s.strip() for s in str(args.splits).split(",") if s.strip())
    if not splits:
        raise SystemExit("--splits must not be empty")
    total_missing = 0
    total_conflict = 0

    # Always verify "clean" first so we don't hide basic issues.
    corrupts = ["clean"]
    if (args.corrupt or "").strip() not in ("", "clean", "none"):
        corrupts.append(args.corrupt)

    for c in corrupts:
        print(f"[verify_rsar_corrupt_switch] === corrupt={c} ===")
        for split in splits:
            split_dir = data_root / split
            ann_dir = split_dir / "annfiles"
            img_dir = _images_dir_for(split_dir, c)
            if not ann_dir.is_dir():
                raise FileNotFoundError(ann_dir)
            if not img_dir.is_dir():
                raise FileNotFoundError(img_dir)

            _t, miss, conf = _check_split(
                split=split,
                ann_dir=ann_dir,
                img_dir=img_dir,
                out_csv=out_dir / f"{split}_corrupt-{c}.csv",
                exts=exts,
            )
            total_missing += miss
            total_conflict += conf

    print(f"[verify_rsar_corrupt_switch] summary missing={total_missing} conflict={total_conflict}")
    return 0 if (total_missing == 0 and total_conflict == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
