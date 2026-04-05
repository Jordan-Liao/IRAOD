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

    def _save_bn_stats(self, model):
        self._bn_backup = {}
        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                              torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                if m.running_mean is not None:
                    self._bn_backup[name] = (
                        m.running_mean.clone(),
                        m.running_var.clone(),
                        m.num_batches_tracked.clone(),
                    )

    def _restore_bn_stats(self, model):
        for name, m in model.named_modules():
            if name in self._bn_backup:
                mean, var, cnt = self._bn_backup[name]
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(cnt)

    def before_train_iter(self, runner) -> None:
        self._save_bn_stats(runner.model)

    def after_train_iter(self, runner) -> None:
        loss = runner.outputs.get("loss", None)
        if not self._is_finite_loss(loss):
            self._skipped += 1
            runner.optimizer.zero_grad()
            self._restore_bn_stats(runner.model)
            rank, _ = get_dist_info()
            if rank == 0 and (self._skipped == 1 or self._skipped % self.log_interval == 0):
                runner.logger.warning(
                    f"[SkipNanOptimizerHook] Non-finite loss={loss} "
                    f"(epoch={runner.epoch}, iter={runner.iter}). "
                    f"Skip step + restore BN. skipped={self._skipped}"
                )
            if self.max_skips is not None and self._skipped >= self.max_skips:
                raise RuntimeError(
                    f"[SkipNanOptimizerHook] Reached max_skips={self.max_skips}. "
                    "Please inspect data/config for numerical issues."
                )
            return
        self._skipped = 0
        super().after_train_iter(runner)

