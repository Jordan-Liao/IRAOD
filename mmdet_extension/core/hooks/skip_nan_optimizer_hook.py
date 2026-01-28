# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Optimizer hook utilities.

SkipNanOptimizerHook:
- Detect non-finite (NaN/Inf) loss.
- Skip optimizer.step() to avoid poisoning weights.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mmcv.runner import HOOKS, OptimizerHook, get_dist_info


@HOOKS.register_module()
class SkipNanOptimizerHook(OptimizerHook):
    """Skip optimizer step when loss is non-finite.

    This is useful for long runs where a single bad batch could turn the whole
    training into NaNs while still producing checkpoints.
    """

    def __init__(
        self,
        grad_clip: dict | None = None,
        detect_anomalous_params: bool = False,
        *,
        max_skips: int | None = None,
        log_interval: int = 1,
    ) -> None:
        super().__init__(grad_clip=grad_clip, detect_anomalous_params=detect_anomalous_params)
        self.max_skips = None if max_skips is None else int(max_skips)
        self.log_interval = max(1, int(log_interval))
        self._skipped = 0

    def _is_finite_loss(self, loss: Any) -> bool:
        if loss is None:
            return True
        if isinstance(loss, (float, int)):
            return math.isfinite(float(loss))
        if torch.is_tensor(loss):
            return bool(torch.isfinite(loss).all().item())
        try:
            return math.isfinite(float(loss))
        except Exception:
            return False

    def after_train_iter(self, runner) -> None:
        loss = runner.outputs.get("loss", None)
        if not self._is_finite_loss(loss):
            self._skipped += 1
            runner.optimizer.zero_grad()
            rank, _ = get_dist_info()
            if rank == 0 and (self._skipped == 1 or self._skipped % self.log_interval == 0):
                runner.logger.warning(
                    f"[SkipNanOptimizerHook] Non-finite loss detected: loss={loss} "
                    f"(epoch={runner.epoch}, iter={runner.iter}). Skip optimizer step. skipped={self._skipped}"
                )
            if self.max_skips is not None and self._skipped >= self.max_skips:
                raise RuntimeError(
                    f"[SkipNanOptimizerHook] Reached max_skips={self.max_skips} with non-finite loss. "
                    "Please inspect data/config for numerical issues."
                )
            return

        super().after_train_iter(runner)

