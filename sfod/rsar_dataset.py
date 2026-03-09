from __future__ import annotations

from mmrotate.datasets.builder import ROTATED_DATASETS

from .semi_dota_dataset import DOTADatasetAnySuffix


@ROTATED_DATASETS.register_module()
class RSARDataset(DOTADatasetAnySuffix):
    """RSAR dataset (DOTA-style annfiles/images) with any-suffix image resolve.

    This is a thin alias of :class:`DOTADatasetAnySuffix` so configs can use a
    semantically clear dataset type name (and leave room for RSAR-specific
    extensions later).
    """

    CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")

