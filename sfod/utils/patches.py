import glob
import os
import os.path as osp
import shutil
import types
from copy import deepcopy

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

from .signature import parse_method_info
from .vars import resolve


def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def patch_checkpoint(runner: BaseRunner):
    # patch save_checkpoint
    old_save_checkpoint = runner.save_checkpoint
    params = parse_method_info(old_save_checkpoint)
    default_tmpl = params["filename_tmpl"].default

    def save_checkpoint(self, out_dir, **kwargs):
        create_symlink = kwargs.get("create_symlink", True)
        filename_tmpl = kwargs.get("filename_tmpl", default_tmpl)
        # create_symlink
        kwargs.update(create_symlink=False)
        old_save_checkpoint(out_dir, **kwargs)
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if isinstance(self, EpochBasedRunner):
                filename = filename_tmpl.format(self.epoch + 1)
            elif isinstance(self, IterBasedRunner):
                filename = filename_tmpl.format(self.iter + 1)
            else:
                raise NotImplementedError()
            filepath = osp.join(out_dir, filename)
            shutil.copy(filepath, dst_file)

    runner.save_checkpoint = types.MethodType(save_checkpoint, runner)
    return runner


def patch_runner(runner):
    runner = patch_checkpoint(runner)
    return runner


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg):

    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)

    def _is_true(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    # RSAR corruption switch: allow `--cfg-options corrupt=interf_xxx` to map
    # img_prefix/img_prefix_u from `.../images/` -> `.../images-interf_xxx/`.
    corrupt = str(cfg.get("corrupt", "")).strip()

    # Optional: mix clean + corrupt on supervised train split (for non-semi datasets).
    # This is implemented by wrapping cfg.data.train with ConcatDataset and (optional)
    # RepeatDataset to control the sampling ratio.
    #
    # Usage (example):
    #   --cfg-options corrupt=interf_jamB_s3 mix_train=1
    #   --cfg-options mix_train_clean_times=1 mix_train_corrupt_times=1
    mix_train = _is_true(cfg.get("mix_train", False))
    mix_train_clean_times = int(cfg.get("mix_train_clean_times", 1) or 1)
    mix_train_corrupt_times = int(cfg.get("mix_train_corrupt_times", 1) or 1)

    def _swap_images_dir(p: str, new_base: str) -> str:
        if not isinstance(p, str):
            return p
        trailing = "/" if p.endswith("/") else ""
        norm = p.rstrip("/")
        if osp.basename(norm) != "images":
            return p
        parent = osp.dirname(norm)
        return osp.join(parent, new_base) + trailing

    def _patch_images_dir(p: str) -> str:
        if not isinstance(p, str):
            return p
        trailing = "/" if p.endswith("/") else ""
        norm = p.rstrip("/")
        if osp.basename(norm) != "images":
            return p
        parent = osp.dirname(norm)
        return osp.join(parent, f"images-{corrupt}") + trailing

    if mix_train and corrupt and corrupt not in ("clean", "none") and cfg.get("data", None) is not None:
        train_ds = cfg.data.get("train")
        # Avoid wrapping SemiDOTADataset (it already mixes sup/unsup internally).
        if isinstance(train_ds, dict) and str(train_ds.get("type", "")).strip() != "SemiDOTADataset":
            clean_ds = deepcopy(train_ds)
            corrupt_ds = deepcopy(train_ds)
            if "img_prefix" in clean_ds and isinstance(clean_ds["img_prefix"], str):
                clean_ds["img_prefix"] = _swap_images_dir(clean_ds["img_prefix"], "images-clean")
            if "img_prefix" in corrupt_ds and isinstance(corrupt_ds["img_prefix"], str):
                corrupt_ds["img_prefix"] = _patch_images_dir(corrupt_ds["img_prefix"])

            def _maybe_repeat(ds: dict, times: int) -> dict:
                if times <= 1:
                    return ds
                return {"type": "RepeatDataset", "times": times, "dataset": ds}

            cfg.data["train"] = {
                "type": "ConcatDataset",
                "datasets": [
                    _maybe_repeat(clean_ds, mix_train_clean_times),
                    _maybe_repeat(corrupt_ds, mix_train_corrupt_times),
                ],
                "separate_eval": False,
            }

    if corrupt and corrupt not in ("clean", "none"):
        if cfg.get("data", None) is not None:
            for split_key in ("train", "val", "test"):
                if split_key not in cfg.data:
                    continue
                ds = cfg.data[split_key]
                for field in ("img_prefix", "img_prefix_u"):
                    if field in ds and isinstance(ds[field], str):
                        ds[field] = _patch_images_dir(ds[field])

    # wrap for semi
    if cfg.get("semi_wrapper", None) is not None:
        cfg.model = cfg.semi_wrapper
        cfg.pop("semi_wrapper")
    # enable environment variables
    # setup_env(cfg)
    return cfg
