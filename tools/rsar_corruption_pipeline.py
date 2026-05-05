from __future__ import annotations

import os
import random

from mmrotate.datasets.builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RsarOnlineCorruptionAugment:
    """Randomly apply one RSAR corruption to clean SAR images during source training.

    Env vars (override constructor args):
      RSAR_CORR_AUG_PROB  – per-image application probability (default 0.5)
      RSAR_CORR_AUG_TYPES – comma-separated corruption names (default: all 7)

    Insert after LoadImageFromFile, before RResize.
    """

    _ALL_TYPES = [
        "chaff",
        "gaussian_white_noise",
        "point_target",
        "noise_suppression",
        "am_noise_horizontal",
        "smart_suppression",
        "am_noise_vertical",
    ]

    def __init__(
        self,
        prob: float = 0.5,
        corruption_types: list[str] | None = None,
    ) -> None:
        from tools.interference_generator import add_interference, default_rsar_corruptions

        _prob_env = os.environ.get("RSAR_CORR_AUG_PROB", "").strip()
        self.prob = float(_prob_env) if _prob_env else float(prob)

        _types_env = os.environ.get("RSAR_CORR_AUG_TYPES", "").strip()
        if _types_env:
            active_types = [t.strip() for t in _types_env.split(",") if t.strip()]
        elif corruption_types is not None:
            active_types = list(corruption_types)
        else:
            active_types = list(self._ALL_TYPES)

        _spec_map = {s.name: s for s in default_rsar_corruptions()}
        self._specs = [
            (name, _spec_map[name].itype, dict(_spec_map[name].params or {}))
            for name in active_types
            if name in _spec_map
        ]
        if not self._specs:
            raise ValueError(f"No valid corruption types found in: {active_types}")

        self._add_interference = add_interference
        print(
            f"[RsarOnlineCorruptionAugment] prob={self.prob:.2f} "
            f"types={[s[0] for s in self._specs]}"
        )

    def __call__(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results

        img = results["img"]
        name, itype, params = random.choice(self._specs)

        filename = results.get("filename") or results.get("ori_filename") or ""
        seed = abs(hash(filename)) % (2 ** 31)

        results["img"] = self._add_interference(img, itype=itype, params=params, seed=seed)
        return results
