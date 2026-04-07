from __future__ import annotations

import csv
import json
import os
import os.path as osp
import random
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, save_checkpoint
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmrotate.core import rbbox2roi
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
from torch.nn.modules.batchnorm import _BatchNorm

from sfod.utils import patch_config

RSAR_CORRUPTIONS = (
    "gaussian_white_noise",
    "point_target",
    "chaff",
    "noise_suppression",
    "smart_suppression",
    "am_noise_vertical",
    "am_noise_horizontal",
)

METHOD_ORDER = ("clean", "direct", "bn", "tent", "shot", "selftrain", "cga")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def json_load(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_source_defaults(
    source_config: str | None,
    source_ckpt: str | None,
) -> tuple[str, str]:
    root = repo_root()
    default_pairs = (
        (
            root / "configs/experiments/rsar/frontier_026_ocafpn_24ep_oriented_rcnn_rsar.py",
            root / "work_dirs/frontier_026_ocafpn_24ep/latest.pth",
        ),
        (
            root / "configs/experiments/rsar/baseline_oriented_rcnn_rsar.py",
            root / "work_dirs/exp_rsar_baseline_full_nanfix/latest.pth",
        ),
    )
    cfg = Path(source_config).expanduser() if source_config else None
    ckpt = Path(source_ckpt).expanduser() if source_ckpt else None

    if cfg is None and ckpt is None:
        for cand_cfg, cand_ckpt in default_pairs:
            if cand_cfg.is_file() and cand_ckpt.exists():
                return str(cand_cfg), str(cand_ckpt)
        raise FileNotFoundError("No default SOURCE_CONFIG/SOURCE_CKPT candidate exists.")

    if cfg is None and ckpt is not None:
        for cand_cfg, cand_ckpt in default_pairs:
            if cand_ckpt.resolve() == ckpt.resolve():
                cfg = cand_cfg
                break
        if cfg is None:
            raise ValueError("SOURCE_CONFIG is required when SOURCE_CKPT is custom.")

    if cfg is not None and ckpt is None:
        for cand_cfg, cand_ckpt in default_pairs:
            if cand_cfg.resolve() == cfg.resolve():
                ckpt = cand_ckpt
                break
        if ckpt is None:
            raise ValueError("SOURCE_CKPT is required when SOURCE_CONFIG is custom.")

    assert cfg is not None and ckpt is not None
    return str(cfg), str(ckpt)


def resolve_split_paths(
    data_root: str | Path,
    split: str,
    corruption: str | None = None,
) -> tuple[str, str]:
    root = Path(data_root).expanduser().resolve()
    ann_dir = root / "test" / "annfiles" if split == "test" else root / split / "annfiles"
    if corruption and corruption not in ("clean", "none"):
        img_dir = root / "corruptions" / corruption / split / "images"
    else:
        img_dir = root / split / "images"
    return str(ann_dir), str(img_dir)


def create_empty_ann_dir(
    data_root: str | Path,
    output_dir: str | Path,
    split: str = "train",
) -> str:
    root = Path(data_root).expanduser().resolve()
    src_ann_dir = root / split / "annfiles"
    dst_ann_dir = ensure_dir(output_dir)
    existing = {p.name for p in dst_ann_dir.glob("*.txt")}
    needed = {p.name for p in src_ann_dir.glob("*.txt")}
    if existing != needed:
        for path in dst_ann_dir.glob("*.txt"):
            path.unlink()
        for ann_path in sorted(src_ann_dir.glob("*.txt")):
            (dst_ann_dir / ann_path.name).write_text("", encoding="utf-8")
    return str(dst_ann_dir)


def load_config(path: str | Path) -> Config:
    cfg = Config.fromfile(str(path))
    return compat_cfg(cfg)


def build_eval_dataset_cfg(
    source_cfg: Config,
    ann_dir: str,
    img_dir: str,
) -> dict[str, Any]:
    test_cfg = deepcopy(source_cfg.data.test)
    if isinstance(test_cfg, (list, tuple)):
        raise TypeError("List-style test dataset configs are not supported for RSAR controls.")
    test_cfg["ann_file"] = ann_dir
    test_cfg["img_prefix"] = img_dir
    test_cfg["test_mode"] = True
    if test_cfg.get("samples_per_gpu", 1) > 1:
        test_cfg["pipeline"] = replace_ImageToTensor(test_cfg["pipeline"])
    return test_cfg


def build_adapt_loader(
    *,
    source_cfg: Config,
    data_root: str,
    split: str,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> Any:
    ann_dir, img_dir = resolve_split_paths(data_root, split=split, corruption=None)
    dataset_cfg = build_eval_dataset_cfg(source_cfg, ann_dir=ann_dir, img_dir=img_dir)
    dataset = build_dataset(dataset_cfg)
    return build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
    )


def prepare_model_cfg_for_build(source_cfg: Config, *, with_cga: bool = False) -> Config:
    cfg = deepcopy(source_cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    if with_cga:
        cfg.model.type = "OrientedRCNN_CGA"
    return cfg


def load_source_detector(
    *,
    source_config: str,
    source_ckpt: str,
    device: torch.device,
    with_cga: bool = False,
) -> tuple[Config, torch.nn.Module]:
    cfg = prepare_model_cfg_for_build(load_config(source_config), with_cga=with_cga)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, source_ckpt, map_location="cpu")
    model.cfg = cfg
    model.CLASSES = tuple(cfg.get("classes", getattr(model, "CLASSES", ())))
    model.to(device)
    return cfg, model


def scatter_data(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    target = device.index if device.type == "cuda" else -1
    return scatter(data, [target])[0]


def save_model_checkpoint(model: torch.nn.Module, path: str | Path, meta: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    save_checkpoint(model, str(path), optimizer=None, meta=meta)


def run_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    log_path: str | Path | None = None,
) -> None:
    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})
    cmd_str = shlex.join(cmd)
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"$ {cmd_str}\n")
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else str(repo_root()),
        env=proc_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            if completed.stdout:
                f.write(completed.stdout)
            if completed.stderr:
                f.write(completed.stderr)
            f.write(f"[exit_code={completed.returncode}]\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {cmd_str}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def run_streaming_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    log_path: str | Path,
) -> None:
    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})
    ensure_dir(Path(log_path).parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else str(repo_root()),
            env=proc_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ret = proc.wait()
        f.write(f"[exit_code={ret}]\n")
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {shlex.join(cmd)}; see {log_path}")


def latest_eval_json(work_dir: str | Path) -> Path:
    candidates = sorted(Path(work_dir).glob("eval_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No eval_*.json found under {work_dir}")
    return candidates[-1]


def evaluate_one_split(
    *,
    config_path: str,
    checkpoint_path: str,
    work_dir: str | Path,
    ann_dir: str,
    img_dir: str,
    seed: int,
    env: dict[str, str] | None = None,
    extra_cfg_options: list[str] | None = None,
) -> float:
    work_dir = ensure_dir(work_dir)
    before = set(work_dir.glob("eval_*.json"))
    test_samples_per_gpu = int(os.environ.get("CONTROL_TEST_SAMPLES_PER_GPU", "4"))
    test_workers_per_gpu = int(os.environ.get("CONTROL_TEST_WORKERS_PER_GPU", "2"))
    cmd = [
        sys.executable,
        "test.py",
        config_path,
        checkpoint_path,
        "--eval",
        "mAP",
        "--work-dir",
        str(work_dir),
        "--samples-per-gpu",
        str(test_samples_per_gpu),
        "--workers-per-gpu",
        str(test_workers_per_gpu),
        "--cfg-options",
        "load_from=None",
        f"seed={seed}",
        f"data.test.ann_file={ann_dir}",
        f"data.test.img_prefix={img_dir}",
    ]
    if extra_cfg_options:
        cmd.extend(extra_cfg_options)
    run_command(cmd, cwd=repo_root(), env=env, log_path=work_dir / "eval.log")
    after = set(work_dir.glob("eval_*.json"))
    new_files = sorted(after - before)
    eval_path = new_files[-1] if new_files else latest_eval_json(work_dir)
    payload = json_load(eval_path)
    return float(payload["metric"]["mAP"])


def evaluate_suite(
    *,
    method: str,
    config_path: str,
    checkpoint_path: str,
    data_root: str,
    result_root: str | Path,
    seed: int,
    env: dict[str, str] | None = None,
    extra_cfg_options: list[str] | None = None,
    include_corruptions: bool = True,
) -> dict[str, Any]:
    result_dir = ensure_dir(Path(result_root) / method)
    metrics: dict[str, float | str] = {}

    clean_ann, clean_img = resolve_split_paths(data_root, split="test", corruption=None)
    metrics["clean_test"] = evaluate_one_split(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        work_dir=result_dir / "eval_clean",
        ann_dir=clean_ann,
        img_dir=clean_img,
        seed=seed,
        env=env,
        extra_cfg_options=extra_cfg_options,
    )

    if include_corruptions:
        for corr in RSAR_CORRUPTIONS:
            ann_dir, img_dir = resolve_split_paths(data_root, split="test", corruption=corr)
            metrics[f"{corr}_test"] = evaluate_one_split(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                work_dir=result_dir / f"eval_{corr}",
                ann_dir=ann_dir,
                img_dir=img_dir,
                seed=seed,
                env=env,
                extra_cfg_options=extra_cfg_options,
            )

    numeric_values = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
    metrics["mean"] = float(np.mean(numeric_values)) if numeric_values else float("nan")

    payload = {
        "method": method,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "metrics": metrics,
        "env": env or {},
        "timestamp": timestamp(),
    }
    json_dump(result_dir / "metrics.json", payload)
    return payload


def collect_result_rows(result_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(result_root)
    for method in METHOD_ORDER:
        metrics_path = root / method / "metrics.json"
        if metrics_path.is_file():
            rows.append(json_load(metrics_path))
    return rows


def table_columns() -> list[str]:
    return ["method", "clean_test", *[f"{c}_test" for c in RSAR_CORRUPTIONS], "mean"]


def write_results_tables(result_root: str | Path) -> tuple[Path, Path]:
    root = ensure_dir(result_root)
    rows = {row["method"]: row for row in collect_result_rows(root)}
    csv_path = root / "results_controls.csv"
    md_path = root / "results_controls.md"
    columns = table_columns()

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for method in METHOD_ORDER:
            row = rows.get(method)
            metrics = row["metrics"] if row else {}
            writer.writerow([method, *[metrics.get(col, "") for col in columns[1:]]])

    md_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for method in METHOD_ORDER:
        row = rows.get(method)
        metrics = row["metrics"] if row else {}
        rendered = []
        for col in columns[1:]:
            value = metrics.get(col, "")
            if isinstance(value, (int, float)):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        md_lines.append("| " + " | ".join([method, *rendered]) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def flatten_cfg_for_dump(cfg: Config) -> Config:
    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict = patch_config(Config(cfg_dict))._cfg_dict.to_dict()
    return Config(cfg_dict)


def configure_tent_trainable(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_names: list[str] = []
    trainable_params: list[torch.nn.Parameter] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, _BatchNorm):
            continue
        if module.weight is not None:
            module.weight.requires_grad = True
            trainable_params.append(module.weight)
            trainable_names.append(f"{module_name}.weight")
        if module.bias is not None:
            module.bias.requires_grad = True
            trainable_params.append(module.bias)
            trainable_names.append(f"{module_name}.bias")
    return trainable_params, trainable_names


def configure_shot_trainable(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_params: list[torch.nn.Parameter] = []
    trainable_names: list[str] = []
    for prefix in ("backbone", "neck"):
        module = getattr(model, prefix, None)
        if module is None:
            continue
        for name, param in module.named_parameters():
            param.requires_grad = True
            trainable_params.append(param)
            trainable_names.append(f"{prefix}.{name}")
    return trainable_params, trainable_names


def compute_roi_entropy_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    topk: int,
    min_fg_conf: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    imgs = batch["img"]
    img_metas = batch["img_metas"]
    if isinstance(imgs, list):
        imgs = imgs[0]
    if isinstance(img_metas, list) and img_metas and isinstance(img_metas[0], list):
        img_metas = img_metas[0]

    x = model.extract_feat(imgs)
    proposals = model.rpn_head.simple_test_rpn(x, img_metas)
    rois = rbbox2roi(proposals)
    if rois.numel() == 0:
        return None, {"num_rois": 0.0, "num_selected": 0.0, "entropy": 0.0}

    bbox_results = model.roi_head._bbox_forward(x, rois)
    cls_score = bbox_results["cls_score"]
    probs = torch.softmax(cls_score, dim=-1)
    num_classes = int(getattr(model.roi_head.bbox_head, "num_classes", probs.shape[1]))

    if probs.shape[1] == num_classes + 1:
        fg_probs = probs[:, :-1]
        bg_prob = probs[:, -1]
    else:
        fg_probs = probs
        bg_prob = torch.zeros(fg_probs.shape[0], device=fg_probs.device, dtype=fg_probs.dtype)

    fg_mass = fg_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fg_dist = fg_probs / fg_mass
    fg_conf = fg_probs.max(dim=1).values
    select_score = fg_conf * (1.0 - bg_prob)

    if topk > 0:
        keep = min(topk, int(select_score.shape[0]))
        topk_idx = torch.topk(select_score, k=keep, largest=True).indices
        select_mask = torch.zeros_like(select_score, dtype=torch.bool)
        select_mask[topk_idx] = True
        select_mask &= select_score >= min_fg_conf
        if not select_mask.any():
            select_mask[topk_idx] = True
    else:
        select_mask = select_score >= min_fg_conf
        if not select_mask.any():
            top1_idx = torch.topk(select_score, k=1, largest=True).indices
            select_mask[top1_idx] = True

    entropy = -(fg_dist.clamp_min(1e-6) * fg_dist.clamp_min(1e-6).log()).sum(dim=1)
    loss = entropy[select_mask].mean()
    stats = {
        "num_rois": float(rois.shape[0]),
        "num_selected": float(select_mask.sum().item()),
        "entropy": float(loss.detach().item()),
    }
    return loss, stats
