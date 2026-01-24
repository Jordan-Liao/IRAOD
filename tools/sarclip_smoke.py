from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SARCLIP_REPO = "https://github.com/CAESAR-Radi/SARCLIP.git"


def _log(msg: str) -> None:
    print(f"[sarclip_smoke] {msg}")


def _try_patch_file(path: Path, *, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        return
    path.write_text(text.replace(old, new), encoding="utf-8")


def _patch_sarclip_checkout(sarclip_dir: Path) -> None:
    # 1) Make GDAL optional (we don't need GeoTIFF for this smoke).
    data_py = sarclip_dir / "sar_clip" / "data.py"
    if data_py.is_file():
        _try_patch_file(
            data_py,
            old="from osgeo import gdal\n",
            new=(
                "try:\n"
                "    from osgeo import gdal\n"
                "except Exception:\n"
                "    gdal = None\n"
            ),
        )

    # 2) Make `transformers` optional (SimpleTokenizer works without it).
    tok_py = sarclip_dir / "sar_clip" / "tokenizer.py"
    if tok_py.is_file():
        _try_patch_file(
            tok_py,
            old="from transformers import CLIPTokenizer\n",
            new=(
                "try:\n"
                "    from transformers import CLIPTokenizer\n"
                "except Exception:\n"
                "    class CLIPTokenizer:  # type: ignore\n"
                "        pass\n"
            ),
        )


def _ensure_sarclip(sarclip_dir: Path, *, auto_setup: bool) -> None:
    sarclip_dir = sarclip_dir.resolve()
    if (sarclip_dir / "sar_clip").is_dir():
        _patch_sarclip_checkout(sarclip_dir)
        sys.path.insert(0, str(sarclip_dir))
        return

    if not auto_setup:
        raise RuntimeError(
            f"SARCLIP code not found at: {sarclip_dir}\n"
            f"Clone it via: git clone --depth 1 {SARCLIP_REPO} {sarclip_dir}"
        )

    sarclip_dir.parent.mkdir(parents=True, exist_ok=True)
    _log(f"cloning SARCLIP -> {sarclip_dir}")
    subprocess.check_call(["git", "clone", "--depth", "1", SARCLIP_REPO, str(sarclip_dir)])
    _patch_sarclip_checkout(sarclip_dir)
    sys.path.insert(0, str(sarclip_dir))


def _expected_weight_candidates(model_name: str) -> list[str]:
    name = model_name.replace("/", "-")
    if name in ("RN50", "RN101"):
        base = f"{name.lower()}_model"
    elif name == "ViT-B-16":
        base = "vit_b_16_model"
    elif name == "ViT-B-32":
        base = "vit_b_32_model"
    elif name == "ViT-L-14":
        base = "vit_l_14_model"
    else:
        return []
    return [f"{base}.safetensors", f"{base}.pth", f"{base}.pt"]


def _pick_default_pretrained(
    model_name: str,
    repo_root: Path,
    sarclip_dir: Path,
    *,
    prefer_safetensors: bool,
) -> Path | None:
    model_name_norm = model_name.replace("/", "-")
    candidates = _expected_weight_candidates(model_name_norm)
    if not prefer_safetensors:
        candidates = [p for p in candidates if not p.endswith(".safetensors")] + [
            p for p in candidates if p.endswith(".safetensors")
        ]

    # Prefer repo-local weights/ for reproducibility.
    weights_dir = repo_root / "weights" / "sarclip" / model_name_norm
    for cand in candidates:
        if (weights_dir / cand).is_file():
            return (weights_dir / cand).resolve()
    if weights_dir.is_dir():
        patterns = ["*.safetensors", "*.pth", "*.pt"] if prefer_safetensors else ["*.pth", "*.pt", "*.safetensors"]
        for pat in patterns:
            for cand in sorted(weights_dir.glob(pat)):
                return cand.resolve()

    # Fallback: SARCLIP checkout suggested path sar_clip/model_configs/{MODEL_NAME}/
    model_cfg_dir = sarclip_dir / "sar_clip" / "model_configs" / model_name_norm
    for cand in candidates:
        if (model_cfg_dir / cand).is_file():
            return (model_cfg_dir / cand).resolve()
    if model_cfg_dir.is_dir():
        patterns = ["*.safetensors", "*.pth", "*.pt"] if prefer_safetensors else ["*.pth", "*.pt", "*.safetensors"]
        for pat in patterns:
            for cand in sorted(model_cfg_dir.glob(pat)):
                return cand.resolve()

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to one SAR image (png/jpg/bmp/...)")
    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="One or more prompts, e.g. \"an SAR image of ship\"",
    )
    parser.add_argument("--model", default="RN50", help="SARCLIP model name (e.g., RN50, ViT-B-32)")
    parser.add_argument("--pretrained", default="", help="Path to SARCLIP weights (*.safetensors/*.pt). Optional.")
    parser.add_argument(
        "--sarclip-dir",
        default=os.environ.get("SARCLIP_DIR", "third_party/SARCLIP"),
        help="Checkout dir containing sar_clip/ (default: third_party/SARCLIP or $SARCLIP_DIR)",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--precision", default="amp", help="amp/fp32/fp16/bf16/pure_fp16/pure_bf16")
    parser.add_argument("--no-auto-setup", action="store_true", help="Disable auto clone SARCLIP if missing")
    parser.add_argument("--out", default="work_dirs/sanity/sarclip_smoke.log", help="Write a short log here")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sarclip_dir = (repo_root / args.sarclip_dir).resolve()
    img_path = Path(args.image).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_path.is_file():
        _log(f"ERROR: image not found: {img_path}")
        return 2

    t0 = time.time()
    _ensure_sarclip(sarclip_dir, auto_setup=not args.no_auto_setup)

    try:
        import sar_clip  # type: ignore
    except Exception as e:
        _log(f"ERROR: failed to import sar_clip from {sarclip_dir}: {e}")
        return 2

    import torch
    from PIL import Image
    from torchvision import transforms as T

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    can_load_safetensors = hasattr(torch, "frombuffer")
    if can_load_safetensors:
        try:
            import safetensors.torch  # type: ignore  # noqa: F401
        except Exception:
            can_load_safetensors = False

    pretrained_path: str | None
    if args.pretrained:
        p = Path(args.pretrained).resolve()
        if not p.is_file():
            _log(f"ERROR: pretrained weights not found: {p}")
            return 2
        pretrained_path = str(p)
    else:
        default_pretrained = _pick_default_pretrained(
            args.model,
            repo_root,
            sarclip_dir,
            prefer_safetensors=can_load_safetensors,
        )
        pretrained_path = str(default_pretrained) if default_pretrained else None

    if pretrained_path is None:
        _log("WARNING: no pretrained weights found; using random init (scores are not meaningful).")
        _log(
            "Download SARCLIP weights from the upstream README (BaiduNetDisk) and place under "
            "`weights/sarclip/{MODEL_NAME}/` (recommended) or `third_party/SARCLIP/sar_clip/model_configs/{MODEL_NAME}/`."
        )

    model = sar_clip.create_model_with_args(
        args.model,
        pretrained=pretrained_path,
        precision=args.precision,
        device=str(device),
        cache_dir=str((sarclip_dir / "sar_clip" / "model_configs" / args.model.replace('/', '-')).resolve()),
        output_dict=False,
    )
    model.eval()

    tokenizer = sar_clip.get_tokenizer(
        args.model,
        cache_dir=str((sarclip_dir / "sar_clip" / "model_configs" / args.model.replace("/", "-")).resolve()),
    )

    img = Image.open(img_path).convert("RGB")
    image_size = getattr(getattr(model, "visual", None), "image_size", 224) or 224
    image_size = int(image_size) if not isinstance(image_size, (tuple, list)) else int(image_size[0])
    preprocess = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = preprocess(img).unsqueeze(0).to(device)
    text = tokenizer(args.prompts).to(device)

    if device.type == "cuda" and args.precision == "amp":
        autocast = torch.cuda.amp.autocast
    else:
        from contextlib import nullcontext

        autocast = nullcontext  # type: ignore

    with torch.no_grad(), autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].detach().cpu().tolist()

    elapsed = time.time() - t0
    lines = [
        f"image={img_path}",
        f"model={args.model}",
        f"device={device}",
        f"pretrained={pretrained_path}",
        f"elapsed_sec={elapsed:.3f}",
        "",
        "Predictions:",
    ] + [f"{p}\t{prob:.6f}" for p, prob in zip(args.prompts, probs)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _log(f"wrote: {out_path}")
    for line in lines[-(len(args.prompts) + 1) :]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
