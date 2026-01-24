from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass(frozen=True)
class ClipScorerConfig:
    model: str = "RN50"


class ClipScorer:
    def __init__(self, config: ClipScorerConfig | None = None, *, device: str | None = None):
        self.config = config or ClipScorerConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        import clip  # type: ignore

        self._clip = clip
        self.model, self.preprocess = clip.load(self.config.model, device=self.device)
        self.model.eval()

    def signature(self) -> dict[str, Any]:
        return {
            "scorer": "clip",
            "model": self.config.model,
        }

    @torch.no_grad()
    def score(self, *, image_path: str, prompts: list[str]) -> list[float]:
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        text = self._clip.tokenize(prompts).to(self.device)

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        return probs.detach().cpu().tolist()

