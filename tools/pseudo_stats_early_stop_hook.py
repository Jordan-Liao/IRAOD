from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import get_dist_info
from mmdet.utils import get_root_logger


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or str(default))
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip() or str(default))
    except Exception:
        return float(default)


@dataclass
class _EpochStats:
    epoch: int
    images: int
    pseudo_kept: int
    pseudo_pre_thr: int | None
    kept_ratio: float | None
    mean_score: float | None
    pseudo_per_img: float | None
    per_class_kept: dict[str, int]
    majority_class: str | None
    majority_frac: float | None
    early_stop: bool
    early_stop_reason: str | None


@HOOKS.register_module()
class PseudoStatsAndEarlyStopHook(Hook):
    """Log per-epoch pseudo-label stats and optionally early-stop on collapse.

    Enable early stop by setting env `PSEUDO_EARLYSTOP=1`.
    The hook is designed to be safe: if required model fields are missing, it no-ops.
    """

    def __init__(self, *, out_name: str = "pseudo_stats.json") -> None:
        self.out_name = str(out_name)

        self._prev_img_num: int | None = None
        self._prev_pseudo_num = None
        self._prev_pre_thr: int | None = None
        self._prev_kept: int | None = None
        self._prev_score_sum: float | None = None

        self._low_pseudo_streak = 0
        self._majority_streak = 0

    def _unwrap_model(self, runner):
        model = runner.model
        return getattr(model, "module", model)

    def before_train_epoch(self, runner) -> None:
        model = self._unwrap_model(runner)
        ut = model
        if not hasattr(ut, "pseudo_num") or not hasattr(ut, "image_num"):
            return

        try:
            import numpy as np  # local import: training env always has numpy

            self._prev_pseudo_num = np.array(getattr(ut, "pseudo_num"), copy=True)
        except Exception:
            self._prev_pseudo_num = None

        try:
            self._prev_img_num = int(getattr(ut, "image_num"))
        except Exception:
            self._prev_img_num = None

        self._prev_pre_thr = int(getattr(ut, "_pseudo_total_pre_thr", 0) or 0)
        self._prev_kept = int(getattr(ut, "_pseudo_kept_post_thr", 0) or 0)
        self._prev_score_sum = float(getattr(ut, "_pseudo_kept_score_sum_post_thr", 0.0) or 0.0)

    def _write_json(self, runner, rec: _EpochStats) -> None:
        rank, _ = get_dist_info()
        if rank != 0:
            return

        out_path = Path(str(getattr(runner, "work_dir", "."))) / self.out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": rec.epoch,
            "images": rec.images,
            "pseudo_kept": rec.pseudo_kept,
            "pseudo_pre_thr": rec.pseudo_pre_thr,
            "kept_ratio": rec.kept_ratio,
            "mean_score": rec.mean_score,
            "pseudo_per_img": rec.pseudo_per_img,
            "per_class_kept": rec.per_class_kept,
            "majority_class": rec.majority_class,
            "majority_frac": rec.majority_frac,
            "early_stop": rec.early_stop,
            "early_stop_reason": rec.early_stop_reason,
        }

        # Append to a list for easy plotting later.
        if out_path.exists():
            try:
                obj = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                obj = []
        else:
            obj = []
        if not isinstance(obj, list):
            obj = []
        obj.append(payload)
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def after_train_epoch(self, runner) -> None:
        model = self._unwrap_model(runner)
        ut = model
        if not hasattr(ut, "pseudo_num") or not hasattr(ut, "image_num"):
            return

        logger = get_root_logger()
        rank, _ = get_dist_info()

        # -------- compute deltas (best-effort) --------
        epoch = int(getattr(runner, "epoch", 0) or 0)

        img_now = int(getattr(ut, "image_num", 0) or 0)
        img_prev = int(self._prev_img_num or 0)
        delta_img_local = max(0, img_now - img_prev)

        pre_now = int(getattr(ut, "_pseudo_total_pre_thr", 0) or 0)
        kept_now = int(getattr(ut, "_pseudo_kept_post_thr", 0) or 0)
        score_sum_now = float(getattr(ut, "_pseudo_kept_score_sum_post_thr", 0.0) or 0.0)

        pre_prev = int(self._prev_pre_thr or 0)
        kept_prev = int(self._prev_kept or 0)
        score_sum_prev = float(self._prev_score_sum or 0.0)

        delta_pre_local = max(0, pre_now - pre_prev)
        delta_kept_local = max(0, kept_now - kept_prev)
        delta_score_sum_local = max(0.0, score_sum_now - score_sum_prev)

        # Per-class counts (local deltas)
        per_class_kept: dict[str, int] = {}
        majority_class = None
        majority_frac = None
        delta_per_class_local = None
        class_names = getattr(ut, "CLASSES", None)
        try:
            import numpy as np

            cur = np.array(getattr(ut, "pseudo_num"), dtype=np.float64)
            prev = np.array(self._prev_pseudo_num, dtype=np.float64) if self._prev_pseudo_num is not None else None
            delta = cur - prev if prev is not None else cur
            delta = np.maximum(delta, 0.0)
            delta_per_class_local = delta

            if not class_names:
                class_names = [f"class_{i}" for i in range(int(delta.shape[0]))]
        except Exception:
            delta_per_class_local = None
            if not class_names:
                class_names = None

        # -------- distributed aggregation (to keep all ranks consistent) --------
        delta_img = delta_img_local
        delta_pre = delta_pre_local
        delta_kept = delta_kept_local
        delta_score_sum = delta_score_sum_local
        delta_per_class = delta_per_class_local

        try:
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

                t = torch.tensor(
                    [float(delta_img_local), float(delta_pre_local), float(delta_kept_local), float(delta_score_sum_local)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                delta_img = int(t[0].item())
                delta_pre = int(t[1].item())
                delta_kept = int(t[2].item())
                delta_score_sum = float(t[3].item())

                if delta_per_class_local is not None:
                    tp = torch.tensor(delta_per_class_local, device=device, dtype=torch.float64)
                    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                    try:
                        import numpy as np

                        delta_per_class = tp.detach().cpu().numpy().astype(np.float64)
                    except Exception:
                        delta_per_class = None
                else:
                    delta_per_class = None
        except Exception:
            # Best-effort: keep local stats when torch/dist is unavailable.
            pass

        kept_ratio = (delta_kept / delta_pre) if delta_pre > 0 else None
        mean_score = (delta_score_sum / delta_kept) if delta_kept > 0 else None
        pseudo_per_img = (delta_kept / delta_img) if delta_img > 0 else None

        # Per-class counts (aggregated if possible)
        try:
            if delta_per_class is not None:
                total = float(delta_per_class.sum())
                if not class_names:
                    class_names = [f"class_{i}" for i in range(int(delta_per_class.shape[0]))]

                for i, name in enumerate(list(class_names)[: int(delta_per_class.shape[0])]):
                    per_class_kept[str(name)] = int(delta_per_class[i])
                if total > 0:
                    mi = int(delta_per_class.argmax())
                    majority_class = str(list(class_names)[mi])
                    majority_frac = float(delta_per_class[mi] / total)
        except Exception:
            pass

        # -------- early stop (optional) --------
        early_stop = False
        early_stop_reason = None

        if _as_bool(os.environ.get("PSEUDO_EARLYSTOP", "0")):
            warmup_epochs = _int_env("PSEUDO_EARLYSTOP_WARMUP_EPOCHS", 2)
            patience = _int_env("PSEUDO_EARLYSTOP_PATIENCE", 2)
            min_pseudo_per_img = _float_env("PSEUDO_EARLYSTOP_MIN_PSEUDO_PER_IMG", 0.05)
            max_majority_frac = _float_env("PSEUDO_EARLYSTOP_MAX_MAJORITY_FRAC", 0.995)
            min_epoch_pseudo = _int_env("PSEUDO_EARLYSTOP_MIN_EPOCH_PSEUDO", 1000)

            if epoch + 1 > warmup_epochs:
                if pseudo_per_img is not None and pseudo_per_img < min_pseudo_per_img:
                    self._low_pseudo_streak += 1
                else:
                    self._low_pseudo_streak = 0

                if (
                    majority_frac is not None
                    and majority_frac >= max_majority_frac
                    and delta_kept >= min_epoch_pseudo
                ):
                    self._majority_streak += 1
                else:
                    self._majority_streak = 0

                if self._low_pseudo_streak >= patience:
                    early_stop = True
                    early_stop_reason = f"low_pseudo_per_img<{min_pseudo_per_img} for {patience} epochs"
                elif self._majority_streak >= patience:
                    early_stop = True
                    early_stop_reason = f"majority_class_frac>={max_majority_frac} for {patience} epochs"

            if early_stop:
                # Stop before starting the next epoch.
                try:
                    if hasattr(runner, "_max_epochs"):
                        runner._max_epochs = min(int(runner._max_epochs), int(runner.epoch) + 1)
                except Exception:
                    pass
                try:
                    if hasattr(runner, "max_epochs"):
                        runner.max_epochs = int(runner.epoch) + 1
                except Exception:
                    pass

        # -------- log + persist --------
        if rank == 0:
            msg = (
                f"[PseudoStats] epoch={epoch} images={delta_img} kept={delta_kept}"
                + (f" pre={delta_pre}" if delta_pre is not None else "")
                + (f" kept_ratio={kept_ratio:.4f}" if kept_ratio is not None else "")
                + (f" mean_score={mean_score:.4f}" if mean_score is not None else "")
                + (f" pseudo/img={pseudo_per_img:.4f}" if pseudo_per_img is not None else "")
            )
            logger.info(msg)
            if majority_class is not None and majority_frac is not None:
                logger.info("[PseudoStats] majority=%s frac=%.4f", majority_class, majority_frac)
            if per_class_kept:
                logger.info("[PseudoStats] per_class=%s", json.dumps(per_class_kept, ensure_ascii=False))
            if early_stop:
                logger.warning("[PseudoStats] EARLY-STOP: %s", early_stop_reason or "(unknown)")

        self._write_json(
            runner,
            _EpochStats(
                epoch=epoch,
                images=delta_img,
                pseudo_kept=delta_kept,
                pseudo_pre_thr=delta_pre,
                kept_ratio=kept_ratio,
                mean_score=mean_score,
                pseudo_per_img=pseudo_per_img,
                per_class_kept=per_class_kept,
                majority_class=majority_class,
                majority_frac=majority_frac,
                early_stop=early_stop,
                early_stop_reason=early_stop_reason,
            ),
        )
