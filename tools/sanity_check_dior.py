import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def _read_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def _find_image(jpeg_dir: Path, img_id: str) -> Path | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"):
        p = jpeg_dir / f"{img_id}{ext}"
        if p.is_file():
            return p
    return None


def _parse_obb_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bnd = obj.find("robndbox")
        if bnd is None:
            continue
        try:
            pts = [
                float(bnd.findtext("x_left_top")),
                float(bnd.findtext("y_left_top")),
                float(bnd.findtext("x_right_top")),
                float(bnd.findtext("y_right_top")),
                float(bnd.findtext("x_right_bottom")),
                float(bnd.findtext("y_right_bottom")),
                float(bnd.findtext("x_left_bottom")),
                float(bnd.findtext("y_left_bottom")),
            ]
        except Exception:
            continue
        objects.append({"name": name, "poly": pts})
    return objects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="dataset/DIOR")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir.parent / "dior_sanity_report.json"

    sets_dir = data_root / "ImageSets"
    ids_path = sets_dir / f"{args.split}.txt"
    if not ids_path.is_file():
        raise FileNotFoundError(ids_path)

    jpeg_dir = data_root / "JPEGImages"
    ann_dir = data_root / "Annotations" / "Oriented Bounding Boxes"
    if not ann_dir.is_dir():
        ann_dir = data_root / "Annotations"

    ids = _read_ids(ids_path)
    if not ids:
        raise RuntimeError(f"empty split list: {ids_path}")

    rng = random.Random(args.seed)
    sample_ids = ids if args.num >= len(ids) else rng.sample(ids, args.num)

    stats = {
        "data_root": str(data_root),
        "split": args.split,
        "requested": args.num,
        "sampled": len(sample_ids),
        "images_total_in_split": len(ids),
        "images_ok": 0,
        "images_missing": 0,
        "xml_missing": 0,
        "objects_total": 0,
        "empty_ann": 0,
        "invalid_poly": 0,
        "examples_missing": [],
        "examples_invalid": [],
    }

    for img_id in sample_ids:
        img_path = _find_image(jpeg_dir, img_id)
        if img_path is None:
            stats["images_missing"] += 1
            stats["examples_missing"].append({"id": img_id, "reason": "image_missing"})
            continue

        xml_path = ann_dir / f"{img_id}.xml"
        if not xml_path.is_file():
            stats["xml_missing"] += 1
            stats["examples_missing"].append({"id": img_id, "reason": "xml_missing"})
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats["images_missing"] += 1
            stats["examples_missing"].append({"id": img_id, "reason": "cv2_imread_failed"})
            continue

        h, w = img.shape[:2]
        objects = _parse_obb_xml(xml_path)
        if not objects:
            stats["empty_ann"] += 1

        for obj in objects:
            poly = obj["poly"]
            if len(poly) != 8:
                stats["invalid_poly"] += 1
                stats["examples_invalid"].append({"id": img_id, "reason": "poly_len", "poly": poly})
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, 8, 2)]
            if any((x != x) or (y != y) for x, y in pts):  # NaN
                stats["invalid_poly"] += 1
                stats["examples_invalid"].append({"id": img_id, "reason": "nan", "poly": poly})
                continue

            stats["objects_total"] += 1

            pts_i = []
            out_of_range = False
            for x, y in pts:
                xi, yi = int(round(x)), int(round(y))
                pts_i.append([xi, yi])
                if xi < -10 or yi < -10 or xi > w + 10 or yi > h + 10:
                    out_of_range = True

            if out_of_range:
                stats["examples_invalid"].append({"id": img_id, "reason": "out_of_range", "poly": poly})

            pts_np = np.array(pts_i, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            if obj["name"]:
                x0, y0 = pts_i[0]
                cv2.putText(
                    img,
                    obj["name"],
                    (max(0, x0), max(0, y0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        out_path = out_dir / f"{img_id}.jpg"
        cv2.imwrite(str(out_path), img)
        stats["images_ok"] += 1

    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sanity_check_dior] wrote {report_path}")
    print(f"[sanity_check_dior] wrote vis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
