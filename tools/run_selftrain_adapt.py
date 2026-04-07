from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

from mmcv import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rsar_controls_common import (
    evaluate_suite,
    json_dump,
    create_empty_ann_dir,
    repo_root,
    resolve_source_defaults,
    run_streaming_command,
    set_random_seed,
    timestamp,
    write_results_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-free self-training / CGA adaptation on RSAR/train.")
    parser.add_argument("--method", choices=["selftrain", "cga"], required=True, help="Output row name.")
    parser.add_argument("--source-config", default=None, help="Detector config for the shared source model.")
    parser.add_argument("--source-ckpt", default=None, help="SOURCE_CKPT path.")
    parser.add_argument(
        "--template-config",
        default="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py",
        help="Semi-supervised template config used for pipelines/runner defaults.",
    )
    parser.add_argument("--data-root", required=True, help="RSAR dataset root.")
    parser.add_argument("--result-root", required=True, help="Root directory for control-baseline outputs.")
    parser.add_argument("--samples-per-gpu", type=int, default=8, help="Per-GPU train batch size.")
    parser.add_argument("--workers-per-gpu", type=int, default=4, help="Per-GPU dataloader workers.")
    parser.add_argument("--max-epochs", type=int, default=24, help="Total adaptation epochs.")
    parser.add_argument("--lr", type=float, default=0.02, help="Optimizer learning rate.")
    parser.add_argument("--weight-u", type=float, default=0.5, help="Unsupervised loss weight.")
    parser.add_argument("--tau", type=float, default=0.5, help="Pseudo-label score threshold.")
    parser.add_argument("--ema-momentum", type=float, default=0.998, help="EMA momentum.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--cuda-visible-devices", default="", help="Comma-separated GPU ids passed to the child train.py process.")
    parser.add_argument("--master-port", type=int, default=29621, help="DDP master port when multiple GPUs are used.")
    parser.add_argument("--sarclip-model", default="ViT-L-14", help="SARCLIP model for CGA runs.")
    parser.add_argument("--sarclip-pretrained", default="", help="SARCLIP checkpoint path for CGA runs.")
    parser.add_argument("--sarclip-lora", default="lora_finetune/SARCLIP_LoRA_Interference.pt", help="Optional SARCLIP LoRA adapter.")
    parser.add_argument("--cga-lambda", type=float, default=0.2, help="Teacher/CLIP fusion weight when labels disagree.")
    parser.add_argument("--strict-paper-prompt", action="store_true", help='Use "An aerial image of a {}" instead of SAR prompts.')
    parser.add_argument("--cga-templates", default="", help='Override prompt templates separated by "|".')
    return parser.parse_args()


def union_imports(*configs: Config) -> dict[str, object]:
    imports: list[str] = []
    for cfg in configs:
        cfg_imports = cfg.get("custom_imports", {})
        for item in cfg_imports.get("imports", []) if isinstance(cfg_imports, dict) else []:
            if item not in imports:
                imports.append(item)
    for extra in ("sfod", "mmdet_extension", "mmrotate.datasets.pipelines"):
        if extra not in imports:
            imports.append(extra)
    return {"imports": imports, "allow_failed_imports": False}


def default_lr_steps(max_epochs: int) -> list[int]:
    if max_epochs == 24:
        return [16, 22]
    if max_epochs <= 2:
        return [max_epochs]
    return sorted({max(1, int(max_epochs * 0.66)), max(1, int(max_epochs * 0.9))})


def finalize_dumped_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    prefix_lines: list[str] = []
    if "ColorJitter(" in text and "from torchvision.transforms import ColorJitter" not in text:
        prefix_lines.append("from torchvision.transforms import ColorJitter")
    if prefix_lines:
        text = "\n".join([*prefix_lines, "", text])
        path.write_text(text, encoding="utf-8")


def build_generated_configs(
    *,
    source_config: str,
    source_ckpt: str,
    template_config: str,
    work_dir: Path,
    data_root: str,
    max_epochs: int,
    lr: float,
    weight_u: float,
    tau: float,
    ema_momentum: float,
) -> tuple[Path, Path]:
    source_cfg = Config.fromfile(source_config)
    template_cfg = Config.fromfile(template_config)

    source_model = deepcopy(source_cfg.model)
    if "pretrained" in source_model:
        source_model["pretrained"] = None

    classes = tuple(source_cfg.get("classes", template_cfg.get("classes", ())))
    data_root_path = Path(data_root).expanduser().resolve()
    train_img = str((data_root_path / "train" / "images").resolve()) + "/"
    test_img = str((data_root_path / "test" / "images").resolve()) + "/"
    test_ann = str((data_root_path / "test" / "annfiles").resolve()) + "/"
    empty_ann_dir = create_empty_ann_dir(data_root_path, work_dir / "empty_ann_train", split="train")
    empty_ann_dir = str(Path(empty_ann_dir).resolve()) + "/"

    ema_cfg_dict = source_cfg._cfg_dict.to_dict()
    ema_cfg_dict["custom_imports"] = union_imports(source_cfg, template_cfg)
    ema_model = deepcopy(source_model)
    ema_model["type"] = "OrientedRCNN_CGA"
    ema_cfg_dict["model"] = ema_model
    if "data" not in ema_cfg_dict:
        ema_cfg_dict["data"] = deepcopy(template_cfg.data)
    ema_cfg_dict["classes"] = classes
    if "runner" not in ema_cfg_dict:
        ema_cfg_dict["runner"] = dict(type="EpochBasedRunner", max_epochs=max_epochs)
    ema_cfg = Config(ema_cfg_dict)
    ema_cfg_path = work_dir / "generated_ema_config.py"
    ema_cfg.dump(str(ema_cfg_path))
    finalize_dumped_config(ema_cfg_path)

    ut_cfg_dict = template_cfg._cfg_dict.to_dict()
    ut_cfg_dict["custom_imports"] = union_imports(source_cfg, template_cfg)
    ut_cfg_dict["classes"] = classes
    ut_cfg_dict["load_from"] = source_ckpt
    ut_cfg_dict["work_dir"] = str(work_dir)
    ut_cfg_dict["runner"] = dict(type="SemiEpochBasedRunner", max_epochs=max_epochs)
    ut_cfg_dict["lr_config"] = dict(
        policy="step",
        warmup="linear",
        warmup_iters=100,
        warmup_ratio=0.001,
        step=default_lr_steps(max_epochs),
    )
    ut_cfg_dict["optimizer"] = dict(type="SGD", lr=lr, momentum=0.9, weight_decay=0.0001)
    ut_cfg_dict["checkpoint_config"] = dict(interval=1)
    ut_cfg_dict["data"]["samples_per_gpu"] = 1
    ut_cfg_dict["data"]["workers_per_gpu"] = 1
    ut_cfg_dict["data"]["train"]["ann_file"] = empty_ann_dir
    ut_cfg_dict["data"]["train"]["ann_file_u"] = empty_ann_dir
    ut_cfg_dict["data"]["train"]["img_prefix"] = train_img
    ut_cfg_dict["data"]["train"]["img_prefix_u"] = train_img
    ut_cfg_dict["data"]["train"]["classes"] = classes
    ut_cfg_dict["data"]["train"]["filter_empty_gt"] = False
    ut_cfg_dict["data"]["val"]["ann_file"] = test_ann
    ut_cfg_dict["data"]["val"]["img_prefix"] = test_img
    ut_cfg_dict["data"]["val"]["classes"] = classes
    ut_cfg_dict["data"]["test"]["ann_file"] = test_ann
    ut_cfg_dict["data"]["test"]["img_prefix"] = test_img
    ut_cfg_dict["data"]["test"]["classes"] = classes

    ut_model = deepcopy(ut_cfg_dict["model"])
    for key in ("backbone", "neck", "rpn_head", "roi_head", "train_cfg", "test_cfg"):
        ut_model[key] = deepcopy(source_model[key])
    ut_model["ema_config"] = str(ema_cfg_path.resolve())
    ut_model["ema_ckpt"] = source_ckpt
    ut_model["cfg"]["weight_l"] = 0.0
    ut_model["cfg"]["weight_u"] = weight_u
    ut_model["cfg"]["score_thr"] = tau
    ut_model["cfg"]["momentum"] = ema_momentum
    ut_model["cfg"]["use_bbox_reg"] = True
    ut_cfg_dict["model"] = ut_model

    ut_cfg = Config(ut_cfg_dict)
    ut_cfg_path = work_dir / "generated_selftrain_config.py"
    ut_cfg.dump(str(ut_cfg_path))
    finalize_dumped_config(ut_cfg_path)
    return ut_cfg_path, ema_cfg_path


def main() -> None:
    args = parse_args()
    source_config, source_ckpt = resolve_source_defaults(args.source_config, args.source_ckpt)
    set_random_seed(args.seed)

    method_dir = Path(args.result_root) / args.method
    method_dir.mkdir(parents=True, exist_ok=True)

    generated_cfg, generated_ema_cfg = build_generated_configs(
        source_config=source_config,
        source_ckpt=source_ckpt,
        template_config=args.template_config,
        work_dir=method_dir,
        data_root=args.data_root,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_u=args.weight_u,
        tau=args.tau,
        ema_momentum=args.ema_momentum,
    )

    if args.cga_templates:
        cga_templates = args.cga_templates
    elif args.strict_paper_prompt:
        cga_templates = "An aerial image of a {}"
    else:
        cga_templates = "a SAR image of a {}"

    child_env = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "CGA_SCORER": "sarclip" if args.method == "cga" else "none",
        "CGA_TEMPLATES": cga_templates,
        "CGA_LAMBDA": str(args.cga_lambda),
        "SARCLIP_MODEL": args.sarclip_model,
        "SARCLIP_PRETRAINED": args.sarclip_pretrained,
        "SARCLIP_LORA": args.sarclip_lora,
    }
    if args.cuda_visible_devices.strip():
        child_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()
    gpu_tokens = [x for x in args.cuda_visible_devices.split(",") if x.strip()]
    nproc_per_node = max(len(gpu_tokens), 1)

    cmd: list[str] = []
    if nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={args.master_port}",
            "train.py",
            str(generated_cfg),
            "--work-dir",
            str(method_dir),
            "--no-validate",
            "--seed",
            str(args.seed),
            "--launcher",
            "pytorch",
            "--teacher-ckpt",
            source_ckpt,
            "--samples-per-gpu",
            str(args.samples_per_gpu),
            "--workers-per-gpu",
            str(args.workers_per_gpu),
            "--max-epochs",
            str(args.max_epochs),
            "--cga-scorer",
            child_env["CGA_SCORER"],
            "--cga-templates",
            cga_templates,
        ]
    else:
        cmd = [
            sys.executable,
            "train.py",
            str(generated_cfg),
            "--work-dir",
            str(method_dir),
            "--no-validate",
            "--seed",
            str(args.seed),
            "--teacher-ckpt",
            source_ckpt,
            "--samples-per-gpu",
            str(args.samples_per_gpu),
            "--workers-per-gpu",
            str(args.workers_per_gpu),
            "--max-epochs",
            str(args.max_epochs),
            "--cga-scorer",
            child_env["CGA_SCORER"],
            "--cga-templates",
            cga_templates,
        ]
    if args.method == "cga":
        if args.sarclip_model:
            cmd.extend(["--sarclip-model", args.sarclip_model])
        if args.sarclip_pretrained:
            cmd.extend(["--sarclip-pretrained", args.sarclip_pretrained])

    train_log = method_dir / f"train_{timestamp()}.log"
    run_streaming_command(cmd, cwd=repo_root(), env=child_env, log_path=train_log)

    ema_ckpt = method_dir / "latest_ema.pth"
    student_ckpt = method_dir / "latest.pth"
    final_ckpt = ema_ckpt if ema_ckpt.exists() else student_ckpt
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Expected adapted checkpoint not found under {method_dir}")

    json_dump(
        method_dir / "run_meta.json",
        {
            "method": args.method,
            "source_config": source_config,
            "source_ckpt": source_ckpt,
            "generated_train_config": str(generated_cfg),
            "generated_ema_config": str(generated_ema_cfg),
            "final_ckpt": str(final_ckpt),
            "seed": args.seed,
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weight_u": args.weight_u,
            "tau": args.tau,
            "ema_momentum": args.ema_momentum,
            "cga_scorer": child_env["CGA_SCORER"],
            "cga_templates": cga_templates,
            "cga_lambda": args.cga_lambda,
            "cuda_visible_devices": args.cuda_visible_devices,
            "nproc_per_node": nproc_per_node,
            "train_command": cmd,
            "train_log": str(train_log),
        },
    )

    payload = evaluate_suite(
        method=args.method,
        config_path=source_config,
        checkpoint_path=str(final_ckpt),
        data_root=args.data_root,
        result_root=args.result_root,
        seed=args.seed,
        env={"CGA_SCORER": "none"},
        extra_cfg_options=None,
        include_corruptions=True,
    )
    csv_path, md_path = write_results_tables(args.result_root)
    print(f"[run_selftrain_adapt] method={args.method} final_ckpt={final_ckpt}")
    print(f"[run_selftrain_adapt] mean={payload['metrics']['mean']:.4f}")
    print(f"[run_selftrain_adapt] wrote {csv_path}")
    print(f"[run_selftrain_adapt] wrote {md_path}")


if __name__ == "__main__":
    main()
