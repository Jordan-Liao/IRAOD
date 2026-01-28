# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
from pathlib import Path
import time
import warnings
import torch
import numpy as np
import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  

import mmcv
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.utils import get_device
# from mmrotate.apis import train_detector
from mmdet_extension.apis import train_detector #使用自定义mmdet_extension.apis
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (collect_env, get_root_logger,
                            setup_multi_processes)
from sfod.utils import patch_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--data-root',
        default=os.environ.get('RSAR_DATA_ROOT') or None,
        help='Dataset root dir (RSAR layout: {train,val,test}/{images,annfiles}). '
             'If set, rewrites data.* paths to absolute paths under this root.',
    )
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=None,
        help='Override dataloader batch size (sets cfg.data.samples_per_gpu).',
    )
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=None,
        help='Override dataloader workers (sets cfg.data.workers_per_gpu).',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override runner.max_epochs (also rewrites lr_config.step when it is a single-step schedule).',
    )
    parser.add_argument(
        '--teacher-ckpt',
        default=None,
        help='Teacher init checkpoint (sets cfg.load_from and cfg.model.ema_ckpt).',
    )

    # CGA / CLIP / SARCLIP runtime knobs (mapped to env vars so existing code paths work)
    parser.add_argument(
        '--cga-scorer',
        default=None,
        help='Set CGA scorer backend (maps to $CGA_SCORER, e.g. none|clip|sarclip).',
    )
    parser.add_argument(
        '--cga-templates',
        default=None,
        help='Prompt templates separated by "|" (maps to $CGA_TEMPLATES).',
    )
    parser.add_argument(
        '--cga-tau',
        type=float,
        default=None,
        help='Softmax temperature (maps to $CGA_TAU).',
    )
    parser.add_argument(
        '--cga-expand-ratio',
        type=float,
        default=None,
        help='Patch expand ratio (maps to $CGA_EXPAND_RATIO).',
    )
    parser.add_argument(
        '--sarclip-model',
        default=None,
        help='SARCLIP model name (maps to $SARCLIP_MODEL, e.g. RN50).',
    )
    parser.add_argument(
        '--sarclip-pretrained',
        default=None,
        help='Path to SARCLIP weights (maps to $SARCLIP_PRETRAINED).',
    )
    parser.add_argument(
        '--clip-model',
        default=None,
        help='CLIP model name (maps to $CLIP_MODEL, e.g. RN50).',
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()#获取解析后的命令行参数

    def _set_env_if_provided(key: str, value) -> None:
        if value is None:
            return
        v = str(value).strip()
        if v == "":
            return
        os.environ[key] = v

    _set_env_if_provided("CGA_SCORER", args.cga_scorer)
    _set_env_if_provided("CGA_TEMPLATES", args.cga_templates)
    _set_env_if_provided("CGA_TAU", args.cga_tau)
    _set_env_if_provided("CGA_EXPAND_RATIO", args.cga_expand_ratio)
    _set_env_if_provided("SARCLIP_MODEL", args.sarclip_model)
    _set_env_if_provided("SARCLIP_PRETRAINED", args.sarclip_pretrained)
    _set_env_if_provided("CLIP_MODEL", args.clip_model)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    def _infer_rsar_split(path_str: str):
        p = str(path_str).replace("\\", "/")
        for split in ("train", "val", "test"):
            if f"/{split}/" in p:
                return split
        return None

    def _apply_rsar_data_root(cfg: Config, data_root: str) -> None:
        root = Path(data_root).expanduser().resolve()
        if cfg.get("data", None) is None:
            return
        for split_key in ("train", "val", "test"):
            if split_key not in cfg.data:
                continue
            ds = cfg.data[split_key]
            if not isinstance(ds, dict):
                continue
            for field in ("ann_file", "ann_file_u", "img_prefix", "img_prefix_u"):
                if field not in ds or not isinstance(ds[field], str):
                    continue
                split = _infer_rsar_split(ds[field])
                if split is None:
                    continue
                subdir = "annfiles" if field.startswith("ann_") else "images"
                ds[field] = str(root / split / subdir) + "/"

    if args.data_root is not None:
        _apply_rsar_data_root(cfg, args.data_root)

    if args.samples_per_gpu is not None and cfg.get("data", None) is not None:
        cfg.data.samples_per_gpu = int(args.samples_per_gpu)
    if args.workers_per_gpu is not None and cfg.get("data", None) is not None:
        cfg.data.workers_per_gpu = int(args.workers_per_gpu)

    if args.max_epochs is not None and cfg.get("runner", None) is not None and "max_epochs" in cfg.runner:
        new_max_epochs = int(args.max_epochs)
        old_max_epochs = int(cfg.runner.max_epochs)
        cfg.runner.max_epochs = new_max_epochs
        if cfg.get("lr_config", None) is not None:
            step = cfg.lr_config.get("step", None)
            if isinstance(step, list) and len(step) == 1 and int(step[0]) == old_max_epochs:
                cfg.lr_config.step = [new_max_epochs]

    if args.teacher_ckpt is not None:
        teacher_ckpt = str(args.teacher_ckpt).strip()
        if teacher_ckpt != "" and teacher_ckpt.lower() != "none":
            cfg.load_from = teacher_ckpt
            if isinstance(cfg.get("model", None), dict) and "ema_ckpt" in cfg.model:
                cfg.model.ema_ckpt = teacher_ckpt

    # set multi-process settings
    setup_multi_processes(cfg)
    cfg.device = get_device() 
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg = patch_config(cfg)#调用patch_config函数对配置对象进行修改或增强
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        # Some clusters export aggressive NCCL tuning vars globally; these can
        # trigger NCCL "internal error" during DDP init. Prefer library defaults
        # unless the user explicitly sets them inside the job command.
        for k in (
            "NCCL_P2P_DISABLE",
            "NCCL_MIN_NCHANNELS",
            "NCCL_P2P_LEVEL",
            "NCCL_PROTO",
            "NCCL_MAX_NCHANNELS",
        ):
            os.environ.pop(k, None)
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # Optional: auto scale lr by total batch size.
    # Enable per-config via:
    #   auto_scale_lr = dict(enable=True, base_batch_size=16)
    auto_scale_lr_note = None
    auto_scale_lr_cfg = cfg.get("auto_scale_lr", None)
    if auto_scale_lr_cfg and bool(auto_scale_lr_cfg.get("enable", False)):
        base_bs = int(auto_scale_lr_cfg.get("base_batch_size", 0) or 0)
        if base_bs > 0 and cfg.get("data", None) is not None and cfg.get("optimizer", None) is not None:
            samples_per_gpu = int(cfg.data.get("samples_per_gpu", 0) or 0)
            world_size = len(cfg.gpu_ids) if cfg.get("gpu_ids", None) is not None else 1
            total_bs = samples_per_gpu * max(1, world_size)
            if total_bs > 0 and "lr" in cfg.optimizer:
                old_lr = float(cfg.optimizer["lr"])
                new_lr = old_lr * float(total_bs) / float(base_bs)
                cfg.optimizer["lr"] = new_lr
                auto_scale_lr_note = dict(old_lr=old_lr, new_lr=new_lr, base_batch_size=base_bs, total_batch_size=total_bs)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if auto_scale_lr_note is not None:
        logger.info(
            "Auto scale lr: %.6g -> %.6g (total_bs=%s, base_bs=%s)",
            auto_scale_lr_note["old_lr"],
            auto_scale_lr_note["new_lr"],
            auto_scale_lr_note["total_batch_size"],
            auto_scale_lr_note["base_batch_size"],
        )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged  创建 meta 字典用于记录重要信息，如环境信息和随机种子等
    meta = dict()
    # log env info 调用 collect_env() 收集环境信 将环境信息格式化为字符串并记录到日志中 使用虚线分隔符美化日志输出格式 将环境信息存储到 meta 字典中
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # # set random seeds
    # seed = init_random_seed(args.seed)
    # seed = seed + dist.get_rank() if args.diff_seed else seed
    # logger.info(f'Set random seed to {seed}, '
    #             f'deterministic: {args.deterministic}')
    # set_random_seed(seed, deterministic=args.deterministic)
    # cfg.seed = seed
    # meta['seed'] = seed
    # set random seed to fix
    seed = 42
    deterministic = True
    set_random_seed(seed, deterministic=deterministic)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {deterministic}')
    cfg.seed = seed
    meta['seed'] = seed

    meta['exp_name'] = osp.basename(args.config)

    # When loading from a checkpoint, avoid distributed init_cfg downloads
    # (e.g. torchvision://resnet50) which may trigger extra barriers and can
    # fail with NCCL/internal errors on some clusters.
    if cfg.get("load_from", None):
        try:
            model_cfg = cfg.get("model", None)
            if isinstance(model_cfg, dict):
                backbone_cfg = model_cfg.get("backbone", None)
                if isinstance(backbone_cfg, dict) and backbone_cfg.get("init_cfg", None) is not None:
                    backbone_cfg["init_cfg"] = None
        except Exception:
            pass

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:#如果workflow中有val，则将val数据集也加入datasets中
        val_dataset = copy.deepcopy(cfg.data.val)#创建val数据集的副本
        val_dataset.pipeline = cfg.data.train.pipeline#将val数据集的管道设置为训练数据集的管道
        datasets.append(build_dataset(val_dataset))#将val数据集加入datasets中
    if cfg.checkpoint_config is not None:#如果checkpoint_config不为空，则将checkpoint_config中的meta信息加入meta字典中
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],#获取git仓库的hash值前7位
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
