from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _log(msg: str) -> None:
    print(f"[cga_smoke] {msg}")


def _parse_templates(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.split("|")]
    return tuple([p for p in parts if p])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image")
    parser.add_argument("--scorer", choices=("clip", "sarclip"), default="clip")
    parser.add_argument("--classes", required=True, help="Comma-separated class names")
    parser.add_argument(
        "--templates",
        default="",
        help='Prompt templates joined by "|", must contain "{}" placeholder (e.g. "an SAR image of a {}|this SAR patch shows a {}")',
    )
    parser.add_argument("--tau", type=float, default=100.0)
    parser.add_argument("--expand-ratio", type=float, default=0.4)
    parser.add_argument("--out-json", default="work_dirs/sanity/cga_smoke.json")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise SystemExit(f"missing image: {image_path}")

    class_names = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    if not class_names:
        raise SystemExit("empty --classes")

    templates = _parse_templates(args.templates) if args.templates.strip() else ("an image of a {}",)
    for t in templates:
        if "{}" not in t:
            raise SystemExit(f"template must contain '{{}}': {t}")

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    _log(f"device={device}")

    # A single centered crop box to exercise the scoring path.
    w, h = Image.open(image_path).size
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)
    boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    scores = np.array([1.0], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    scorer_used = args.scorer
    if args.scorer == "clip":
        from sfod.cga import _ClipCGA  # noqa: PLC2701

        backend = _ClipCGA(
            class_names=class_names,
            model=os.environ.get("CLIP_MODEL", "RN50"),
            templates=templates,
            tau=float(args.tau),
            expand_ratio=float(args.expand_ratio),
            device=device,
        )
    else:
        from sfod.cga import CGA

        backend = CGA(
            class_names=class_names,
            model=os.environ.get("SARCLIP_MODEL", "RN50"),
            pretrained=None,
            precision="amp",
            templates=templates,
            tau=float(args.tau),
            expand_ratio=float(args.expand_ratio),
            force_grayscale=False,
        )

    probs, _ = backend(str(image_path), boxes, scores, labels)
    probs_list = probs[0].tolist() if probs.shape[0] else []
    top_idx = int(np.argmax(probs[0])) if probs.shape[0] else -1
    top_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else ""

    out = {
        "image": str(image_path),
        "scorer_requested": args.scorer,
        "scorer_used": scorer_used,
        "classes": class_names,
        "templates": list(templates),
        "tau": float(args.tau),
        "expand_ratio": float(args.expand_ratio),
        "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "probs": probs_list,
        "top": {"index": top_idx, "class": top_name, "prob": float(probs[0][top_idx]) if probs.shape[0] else None},
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
