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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", required=True, help="e.g., dataset/RSAR/train/annfiles")
    parser.add_argument("--img-dir", required=True, help="e.g., dataset/RSAR/train/images")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--exts",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="comma-separated allowed extensions",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir).resolve()
    img_dir = Path(args.img_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    if not ann_dir.is_dir():
        raise FileNotFoundError(ann_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    stem2paths = build_index(img_dir, exts)

    ann_files = sorted([p for p in ann_dir.glob("*.txt") if p.is_file()])
    missing = 0
    conflict = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
                    annfile=str(ann_path),
                    image_stem=stem,
                    status=status,
                    resolved_image_path=resolved,
                    candidates="|".join(candidates),
                )
            )

    total = len(ann_files)
    print(f"[check_image_ann_alignment] total={total} missing={missing} conflict={conflict}")
    print(f"[check_image_ann_alignment] wrote {out_csv}")
    return 0 if (missing == 0 and conflict == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

