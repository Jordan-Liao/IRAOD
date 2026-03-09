from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms as T


@dataclass(frozen=True)
class SarclipScorerConfig:
    model: str = "RN50"
    pretrained: str | None = None
    precision: str = "amp"
    lora: str | None = None


def _repo_root() -> Path:
    # sfod/scorers -> sfod -> repo root
    return Path(__file__).resolve().parents[2]


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


def _can_load_safetensors() -> bool:
    if not hasattr(torch, "frombuffer"):
        return False
    try:
        import safetensors.torch  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _pick_default_pretrained(model_name: str) -> str | None:
    repo_root = _repo_root()
    model_name_norm = model_name.replace("/", "-")
    prefer_safetensors = _can_load_safetensors()
    candidates = _expected_weight_candidates(model_name_norm)
    if not prefer_safetensors:
        candidates = [p for p in candidates if not p.endswith(".safetensors")] + [
            p for p in candidates if p.endswith(".safetensors")
        ]

    weights_dir = repo_root / "weights" / "sarclip" / model_name_norm
    for cand in candidates:
        if (weights_dir / cand).is_file():
            return str((weights_dir / cand).resolve())

    return None


def _ensure_sarclip_on_path() -> Path:
    repo_root = _repo_root()
    sarclip_dir = Path(os.environ.get("SARCLIP_DIR", repo_root / "third_party" / "SARCLIP")).resolve()
    if (sarclip_dir / "sar_clip").is_dir():
        sys.path.insert(0, str(sarclip_dir))
        return sarclip_dir
    raise RuntimeError(
        "SARCLIP code not found. Run `python tools/sarclip_smoke.py --image <img> --prompts <p>` "
        "once to auto-clone it into `third_party/SARCLIP`."
    )


class SarclipScorer:
    def __init__(self, config: SarclipScorerConfig | None = None, *, device: str | None = None):
        self.config = config or SarclipScorerConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        _ensure_sarclip_on_path()
        import sar_clip  # type: ignore

        self._sar_clip = sar_clip

        pretrained = self.config.pretrained
        if pretrained is None:
            pretrained = os.environ.get("SARCLIP_PRETRAINED", "").strip() or None
        if pretrained is None:
            pretrained = _pick_default_pretrained(self.config.model)

        self.pretrained = pretrained
        self.model = sar_clip.create_model_with_args(
            self.config.model,
            pretrained=pretrained,
            precision=self.config.precision,
            device=str(self.device),
            cache_dir=str(
                (Path(os.environ.get("SARCLIP_DIR", _repo_root() / "third_party" / "SARCLIP")) / "sar_clip" / "model_configs" / self.config.model.replace("/", "-")).resolve()
            ),
            output_dict=False,
        )
        self.model.eval()
        self.tokenizer = sar_clip.get_tokenizer(
            self.config.model,
            cache_dir=str(
                (Path(os.environ.get("SARCLIP_DIR", _repo_root() / "third_party" / "SARCLIP")) / "sar_clip" / "model_configs" / self.config.model.replace("/", "-")).resolve()
            ),
        )

        image_size = getattr(getattr(self.model, "visual", None), "image_size", 224) or 224
        image_size = int(image_size) if not isinstance(image_size, (tuple, list)) else int(image_size[0])
        self.preprocess = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        lora_path = self.config.lora
        if lora_path is None:
            lora_path = os.environ.get("SARCLIP_LORA", "").strip() or None
        if lora_path:
            lora_path = str(Path(lora_path).expanduser().resolve())
            if not Path(lora_path).is_file():
                raise FileNotFoundError(f"SARCLIP_LORA not found: {lora_path}")

            from tools.lora_utils import LoraConfig, inject_lora, load_lora_state_dict

            ckpt = torch.load(lora_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
                state = ckpt["state_dict"]
            else:
                meta = {}
                state = ckpt

            r = int(meta.get("r", 8))
            alpha = float(meta.get("alpha", 16.0))
            dropout = float(meta.get("dropout", 0.0))
            target = str(meta.get("target", "vision")).strip().lower()

            def _filter(full_name: str, _m: torch.nn.Linear) -> bool:
                if target in ("vision", "visual"):
                    return full_name.startswith("visual.")
                if target in ("vision+text", "both", "all", "visual+text"):
                    return full_name.startswith("visual.") or full_name.startswith("transformer.") or full_name.startswith("text.")
                # fallback: allow all
                return True

            replaced = inject_lora(self.model, config=LoraConfig(r=r, alpha=alpha, dropout=dropout), module_filter=_filter)
            load_lora_state_dict(self.model, state)
            # keep eval mode
            self.model.eval()
            print(f"[SarclipScorer] loaded LoRA: path={lora_path} target={target} replaced_linears={replaced}")

    def signature(self) -> dict[str, Any]:
        return {
            "scorer": "sarclip",
            "model": self.config.model,
            "pretrained": self.pretrained,
            "lora": (self.config.lora or os.environ.get("SARCLIP_LORA", "").strip() or None),
        }

    @torch.no_grad()
    def score(self, *, image_path: str, prompts: list[str]) -> list[float]:
        img = Image.open(image_path).convert("RGB")
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = self.tokenizer(prompts).to(self.device)

        if self.device.type == "cuda" and self.config.precision == "amp":
            autocast = torch.cuda.amp.autocast
        else:
            from contextlib import nullcontext

            autocast = nullcontext  # type: ignore

        with autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        return probs.detach().cpu().tolist()
