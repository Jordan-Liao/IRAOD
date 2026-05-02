from __future__ import annotations

import copy
import os
import os.path as osp
import random
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
    exts_l = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if osp.splitext(fn)[1].lower() in exts_l:
                full = osp.join(root, fn)
                out.append(osp.relpath(full, img_dir).replace("\\", "/"))
    out.sort()
    return out


@ROTATED_DATASETS.register_module()
class SemiRSARSFODDataset(Dataset):
    """Faithful SFOD-RS dataloader for RSAR adaptation.

    Labeled branch  : DOTADatasetAnySuffix on clean RSAR train (real GT).
    Unlabeled branch: directory scan of corruption val (no ann file),
                      processed through share → weak / strong augmentation.

    __len__ follows the labeled source set.  Each __getitem__ pairs a random
    labeled sample with target[idx % len_u], matching the SemiDOTADataset
    / SemiDIORDataset convention used by SFOD-RS.

    Intended use: weight_l=0.0, use_labeled=True in UnbiasedTeacher — the
    labeled source forward-pass guides teacher EMA without contributing a
    supervised loss.

    By default ``__len__`` equals ``min(len(labeled), len(unlabeled))`` so that
    epoch length matches the strict (target-only) baseline for a fair comparison.
    Pass ``max_labeled`` to override.
    """

    _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        *,
        ann_file: str,
        img_prefix: str,
        img_prefix_u: str,
        pipeline: Sequence[object],
        pipeline_u_share: Sequence[object],
        pipeline_u_weak: Sequence[object],
        pipeline_u_strong: Sequence[object],
        classes: Sequence[str] | None = None,
        data_root: str | None = None,
        filter_empty_gt: bool = True,
        img_exts: Sequence[str] | None = None,
        max_labeled: int | None = None,
    ) -> None:
        super().__init__()

        self.dota_labeled = ROTATED_DATASETS.build(
            dict(
                type="DOTADatasetAnySuffix",
                ann_file=ann_file,
                pipeline=list(pipeline),
                data_root=data_root,
                img_prefix=img_prefix,
                test_mode=False,
                filter_empty_gt=filter_empty_gt,
                classes=classes,
            )
        )

        img_prefix_u = str(img_prefix_u)
        if not img_prefix_u.endswith("/"):
            img_prefix_u += "/"
        if not osp.isdir(img_prefix_u):
            raise FileNotFoundError(f"img_prefix_u not found: {img_prefix_u}")

        _exts = tuple(img_exts) if img_exts is not None else self._IMG_EXTS
        self._u_img_prefix = img_prefix_u
        self._u_relpaths = _iter_image_relpaths(img_prefix_u, exts=_exts)
        if not self._u_relpaths:
            raise RuntimeError(
                f"no images found under: {img_prefix_u} exts={_exts}"
            )

        # Effective epoch length: match unlabeled target by default so that
        # runtime is comparable to the strict (target-only) baseline.
        if max_labeled is None:
            self._eff_len = min(len(self.dota_labeled), len(self._u_relpaths))
        else:
            self._eff_len = min(len(self.dota_labeled), int(max_labeled))

        self.CLASSES = tuple(classes) if classes is not None else None
        self._pipe_u_share = _Compose(list(pipeline_u_share))
        self._pipe_u_weak = _Compose(list(pipeline_u_weak))
        self._pipe_u_strong = _Compose(list(pipeline_u_strong))

        import numpy as np

        _flag = getattr(self.dota_labeled, "flag", None)
        if _flag is not None:
            self.flag = _flag[: self._eff_len]
        else:
            self.flag = np.zeros(self._eff_len, dtype=np.uint8)

    def __len__(self) -> int:
        return self._eff_len

    def _u_base(self, idx: int) -> dict:
        return dict(
            img_prefix=self._u_img_prefix,
            img_info=dict(filename=self._u_relpaths[idx]),
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[],
            img_fields=["img"],
        )

    def __getitem__(self, idx: int) -> dict:
        idx_l = random.randint(0, len(self.dota_labeled) - 1)
        results = self.dota_labeled[idx_l]

        u_idx = idx % len(self._u_relpaths)
        shared = self._pipe_u_share(self._u_base(u_idx))
        if shared is None:
            shared = self._pipe_u_share(self._u_base(0))
            if shared is None:
                raise RuntimeError(
                    "pipeline_u_share returned None for both idx and fallback idx=0"
                )

        weak = self._pipe_u_weak(copy.deepcopy(shared))
        strong = self._pipe_u_strong(copy.deepcopy(shared))
        if weak is None or strong is None:
            shared = self._pipe_u_share(self._u_base(0))
            weak = self._pipe_u_weak(copy.deepcopy(shared))
            strong = self._pipe_u_strong(copy.deepcopy(shared))
            if weak is None or strong is None:
                raise RuntimeError(
                    "pipeline_u_weak/strong returned None after retry"
                )

        results.update({f"{k}_unlabeled": v for k, v in weak.items()})
        results.update({f"{k}_unlabeled_1": v for k, v in strong.items()})
        return results
