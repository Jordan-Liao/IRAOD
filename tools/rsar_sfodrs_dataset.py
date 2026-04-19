from __future__ import annotations

import os
import os.path as osp
import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from mmcv.utils import build_from_cfg
from mmrotate.datasets.builder import ROTATED_DATASETS, ROTATED_PIPELINES


class _Compose:
    def __init__(self, transforms: Sequence[object]):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, ROTATED_PIPELINES)
            elif not callable(t):
                raise TypeError(f"transform must be callable or a dict, got: {type(t)}")
            self.transforms.append(t)

    def __call__(self, data: dict) -> dict | None:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


def _iter_image_relpaths(img_dir: str, *, exts: Iterable[str]) -> list[str]:
    out: list[str] = []
    exts_l = tuple([e.lower() for e in exts])
    for root, _, files in os.walk(img_dir):
        for fn in files:
            ext = osp.splitext(fn)[1].lower()
            if ext in exts_l:
                full = osp.join(root, fn)
                out.append(osp.relpath(full, img_dir).replace("\\", "/"))
    out.sort()
    return out


@dataclass(frozen=True)
class _Pipelines:
    share: _Compose
    weak: _Compose
    strong: _Compose


@ROTATED_DATASETS.register_module()
class RSARSourceFreeSelfTrainingDataset(Dataset):
    """Target-only dataset for UnbiasedTeacher forward_train_semi.

    Produces three views from *the same* target image:
    - labeled branch: empty GT, used only to satisfy the UT signature (weight_l must be 0)
    - teacher branch (weak): used to produce pseudo labels
    - student branch (strong): used to learn from pseudo labels
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        *,
        img_prefix: str,
        pipeline_share: Sequence[object],
        pipeline_weak: Sequence[object],
        pipeline_strong: Sequence[object],
        classes: Sequence[str] | None = None,
        img_exts: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.img_prefix = str(img_prefix)
        if not self.img_prefix.endswith("/"):
            self.img_prefix += "/"

        if img_exts is None:
            img_exts = self._IMG_EXTS

        if not osp.isdir(self.img_prefix):
            raise FileNotFoundError(f"img_prefix not found: {self.img_prefix}")

        self._relpaths = _iter_image_relpaths(self.img_prefix, exts=img_exts)
        if len(self._relpaths) == 0:
            raise RuntimeError(f"no images found under: {self.img_prefix} exts={tuple(img_exts)}")

        self.CLASSES = tuple(classes) if classes is not None else None

        self._pipes = _Pipelines(
            share=_Compose(list(pipeline_share)),
            weak=_Compose(list(pipeline_weak)),
            strong=_Compose(list(pipeline_strong)),
        )

        # For GroupSampler compatibility (all 0s => single group).
        import numpy as np

        self.flag = np.zeros(len(self._relpaths), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._relpaths)

    def _base_results(self, idx: int) -> dict:
        rel = self._relpaths[idx]
        return dict(
            img_prefix=self.img_prefix,
            img_info=dict(filename=rel),
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
            img_fields=["img"],
        )

    def __getitem__(self, idx: int) -> dict:
        # Shared geometric aug first (must be identical for weak/strong).
        shared = self._pipes.share(self._base_results(idx))
        if shared is None:
            # Rare; retry with a deterministic fallback.
            shared = self._pipes.share(self._base_results(0))
            if shared is None:
                raise RuntimeError("pipeline_share returned None for both idx and fallback idx=0")

        weak = self._pipes.weak(copy.deepcopy(shared))
        strong = self._pipes.strong(copy.deepcopy(shared))
        if weak is None or strong is None:
            # Retry with a deterministic fallback.
            shared = self._pipes.share(self._base_results(0))
            weak = self._pipes.weak(copy.deepcopy(shared))
            strong = self._pipes.strong(copy.deepcopy(shared))
            if weak is None or strong is None:
                raise RuntimeError("pipeline_weak/strong returned None after retry")

        # UT forward_train_semi signature requires:
        #   img/img_metas/gt_* (labeled) + img_unlabeled(*) (weak) + img_unlabeled_1(*) (strong)
        out: dict = {}
        for k, v in strong.items():
            out[k] = v
        out.update({f"{k}_unlabeled": v for k, v in weak.items()})
        out.update({f"{k}_unlabeled_1": v for k, v in strong.items()})
        return out
