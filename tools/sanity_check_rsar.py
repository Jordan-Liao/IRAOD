import argparse
import json
import os
import os.path as osp
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


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
            stem2paths[stem].append(str(Path(root, fn).resolve()))
    return stem2paths


def parse_dota_txt(ann_path: Path) -> list[dict]:
    lines = ann_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    objs: list[dict] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("imagesource") or low.startswith("gsd"):
            continue
        parts = s.split()
        if len(parts) < 9:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except Exception:
            continue
        cls = parts[8].lower()
        objs.append({"cls": cls, "poly": coords})
    return objs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="dataset/RSAR")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / args.split
    ann_dir = split_root / "annfiles"
    img_dir = split_root / "images"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir.parent / "rsar_sanity_report.json"

    if not ann_dir.is_dir():
        raise FileNotFoundError(ann_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    stem2paths = build_index(img_dir, exts)

    ann_files = sorted([p for p in ann_dir.glob("*.txt") if p.is_file()])
    if not ann_files:
        raise RuntimeError(f"no annfiles under: {ann_dir}")

    rng = random.Random(args.seed)
    sample = ann_files if args.num >= len(ann_files) else rng.sample(ann_files, args.num)

    stats = {
        "data_root": str(data_root),
        "split": args.split,
        "annfiles_total": len(ann_files),
        "requested": args.num,
        "sampled": len(sample),
        "missing_image": 0,
        "conflict_image": 0,
        "images_ok": 0,
        "objects_total": 0,
        "empty_ann": 0,
        "invalid_poly": 0,
        "class_counts": {},
        "examples_missing": [],
        "examples_conflict": [],
    }
    class_counts: dict[str, int] = defaultdict(int)

    for ann_path in sample:
        stem = ann_path.stem
        candidates = stem2paths.get(stem, [])
        if not candidates:
            stats["missing_image"] += 1
            stats["examples_missing"].append(stem)
            continue
        if len(candidates) > 1:
            stats["conflict_image"] += 1
            stats["examples_conflict"].append({"stem": stem, "candidates": candidates[:10]})
        img_path = _pick_by_priority(candidates) or candidates[0]

        img = cv2.imread(img_path)
        if img is None:
            stats["missing_image"] += 1
            stats["examples_missing"].append(stem)
            continue

        objs = parse_dota_txt(ann_path)
        if not objs:
            stats["empty_ann"] += 1

        for obj in objs:
            poly = obj["poly"]
            if len(poly) != 8:
                stats["invalid_poly"] += 1
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if not np.isfinite(pts).all():
                stats["invalid_poly"] += 1
                continue
            pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)
            class_counts[obj["cls"]] += 1
            stats["objects_total"] += 1

        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), img)
        stats["images_ok"] += 1

    stats["class_counts"] = dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sanity_check_rsar] wrote {report_path}")
    print(f"[sanity_check_rsar] wrote vis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

