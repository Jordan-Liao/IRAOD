from __future__ import annotations

import os
from pathlib import Path
from typing import Any


CGA_ENV_ARGS = (
    ("CGA_SCORER", "cga_scorer"),
    ("CGA_TEMPLATES", "cga_templates"),
    ("CGA_TAU", "cga_tau"),
    ("CGA_EXPAND_RATIO", "cga_expand_ratio"),
    ("SARCLIP_MODEL", "sarclip_model"),
    ("SARCLIP_PRETRAINED", "sarclip_pretrained"),
    ("CLIP_MODEL", "clip_model"),
)


def set_env_if_provided(key: str, value: Any) -> None:
    if value is None:
        return
    str_value = str(value).strip()
    if str_value:
        os.environ[key] = str_value


def apply_cga_runtime_env(args: Any) -> None:
    for env_key, attr in CGA_ENV_ARGS:
        set_env_if_provided(env_key, getattr(args, attr, None))


def infer_rsar_split(path_str: str) -> str | None:
    path = str(path_str).replace("\\", "/").lower()
    if "/corruptions/" in path:
        return None
    for split in ("train", "val", "test"):
        if f"/{split}/" in path:
            return split
    return None


def apply_rsar_data_root(cfg: Any, data_root: str | None) -> None:
    if data_root is None or cfg.get("data", None) is None:
        return

    root = Path(data_root).expanduser().resolve()
    for split_key in ("train", "val", "test"):
        if split_key not in cfg.data:
            continue
        ds_cfg = cfg.data[split_key]
        if not isinstance(ds_cfg, dict):
            continue
        for field in ("ann_file", "ann_file_u", "img_prefix", "img_prefix_u"):
            if field not in ds_cfg or not isinstance(ds_cfg[field], str):
                continue
            split = infer_rsar_split(ds_cfg[field])
            if split is None:
                continue
            subdir = "annfiles" if field.startswith("ann_") else "images"
            ds_cfg[field] = str(root / split / subdir) + "/"


def apply_train_dataloader_overrides(cfg: Any, *, samples_per_gpu: int | None, workers_per_gpu: int | None) -> None:
    if cfg.get("data", None) is None:
        return
    if samples_per_gpu is not None:
        cfg.data.samples_per_gpu = int(samples_per_gpu)
    if workers_per_gpu is not None:
        cfg.data.workers_per_gpu = int(workers_per_gpu)


def apply_test_dataloader_overrides(cfg: Any, *, samples_per_gpu: int | None, workers_per_gpu: int | None) -> None:
    if cfg.get("data", None) is None:
        return
    if workers_per_gpu is not None:
        cfg.data.workers_per_gpu = int(workers_per_gpu)
    if samples_per_gpu is None or "test" not in cfg.data:
        return

    samples = int(samples_per_gpu)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.samples_per_gpu = samples
        return
    if isinstance(cfg.data.test, (list, tuple)):
        for ds_cfg in cfg.data.test:
            if isinstance(ds_cfg, dict):
                ds_cfg["samples_per_gpu"] = samples


def apply_max_epochs_override(cfg: Any, max_epochs: int | None) -> None:
    if max_epochs is None or cfg.get("runner", None) is None or "max_epochs" not in cfg.runner:
        return

    new_max_epochs = int(max_epochs)
    old_max_epochs = int(cfg.runner.max_epochs)
    cfg.runner.max_epochs = new_max_epochs

    if cfg.get("lr_config", None) is None:
        return
    step = cfg.lr_config.get("step", None)
    if isinstance(step, list) and len(step) == 1 and int(step[0]) == old_max_epochs:
        cfg.lr_config.step = [new_max_epochs]


def apply_teacher_ckpt(cfg: Any, teacher_ckpt: str | None) -> None:
    if teacher_ckpt is None:
        return
    ckpt = str(teacher_ckpt).strip()
    if not ckpt or ckpt.lower() == "none":
        return

    cfg.load_from = ckpt
    if isinstance(cfg.get("model", None), dict) and "ema_ckpt" in cfg.model:
        cfg.model.ema_ckpt = ckpt
