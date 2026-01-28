# sfod/semi_dota_dataset.py
import os
import os.path as osp
import glob
import copy
import random
import collections

from torch.utils.data import Dataset
from mmcv.utils import build_from_cfg
from mmrotate.datasets.builder import ROTATED_DATASETS, ROTATED_PIPELINES

try:
    from mmrotate.datasets import DOTADataset
except Exception:
    from mmrotate.datasets.dota import DOTADataset


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, ROTATED_PIPELINES)
            elif not callable(t):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(t)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.transforms})'


# 简单的多后缀优先级：你可以按需调整
def _pick_by_priority(cands):
    if not cands:
        return None
    prio = {'.jpg': 0, '.jpeg': 1, '.png': 2, '.bmp': 3, '.tif': 4, '.tiff': 5}
    def keyfn(p):
        ext = osp.splitext(p)[1].lower()
        return prio.get(ext, 99)
    return sorted(cands, key=keyfn)[0]


@ROTATED_DATASETS.register_module()
class DOTADatasetAnySuffix(DOTADataset):
    """DOTA 数据集（兼容任意图片后缀）。
    思路：先按原逻辑 load_annotations，再把 data_infos 里的 filename 统一替换为磁盘上真实存在的同名文件（不论后缀）。
    """

    # 允许的图片后缀
    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    def load_annotations(self, ann_file):
        # 先走原实现，拿到 data_infos（其中 filename 里常常是写死的某个后缀）
        data_infos = super().load_annotations(ann_file)

        img_dir = self.img_prefix if isinstance(self.img_prefix, str) else ''

        # 1) 建立 stem -> 相对路径 的快速索引（一次性扫描，避免每张图都 glob）
        stem2rel = {}
        if img_dir and osp.isdir(img_dir):
            for root, _, files in os.walk(img_dir):
                for fn in files:
                    ext = osp.splitext(fn)[1].lower()
                    if ext in self._IMG_EXTS:
                        stem = osp.splitext(fn)[0]
                        full = osp.join(root, fn)
                        rel = osp.relpath(full, img_dir)
                        # 首次出现的后缀作为默认命中（若你想按优先级挑，可收集列表再挑）
                        stem2rel.setdefault(stem, rel)

        # 2) 回写 data_infos['filename'] 为真实存在的文件
        for info in data_infos:
            name_in_ann = info.get('filename', None)
            if not name_in_ann or not isinstance(name_in_ann, str):
                continue

            # 先尝试用 basename 的 stem 命中（DOTA 一般标注里只存基名）
            stem = osp.splitext(osp.basename(name_in_ann))[0]
            picked_rel = stem2rel.get(stem, None)

            if picked_rel is None:
                # 兜底：通配符尝试（既考虑保持原有相对子路径，也考虑仅用 basename）
                base = osp.splitext(name_in_ann)[0]
                c1 = glob.glob(osp.join(img_dir, base + '.*')) if img_dir else []
                c2 = glob.glob(osp.join(img_dir, stem + '.*')) if img_dir else []
                pick = _pick_by_priority(sorted(set(c1 + c2)))
                if pick and osp.isfile(pick):
                    picked_rel = osp.relpath(pick, img_dir) if img_dir else osp.basename(pick)

            if picked_rel is not None:
                info['filename'] = picked_rel
            # else: 没找到就保持原 filename，不强行报错；读图时会抛 FileNotFoundError，便于定位脏标注

        return data_infos


@ROTATED_DATASETS.register_module()
class SemiDOTADataset(Dataset):
    """基于 DOTA 的半监督封装（自动匹配任意图片后缀）。
    - labeled：走 DOTADatasetAnySuffix + 监督 pipeline
    - unlabeled：走 DOTADatasetAnySuffix + 共享弱几何 pipeline（后续我们再套 teacher 弱增强 / student 强增强）
    """

    def __init__(self,
                 ann_file,
                 ann_file_u,
                 ann_subdir=None,          # 兼容保留，无实际作用
                 pipeline=None,
                 pipeline_u_share=None,
                 pipeline_u=None,
                 pipeline_u_1=None,
                 data_root=None,
                 img_prefix='',            # /.../train/images/
                 seg_prefix=None,
                 proposal_file=None,
                 data_root_u=None,
                 img_prefix_u='',          # /.../val/images 或无标注目录
                 seg_prefix_u=None,
                 proposal_file_u=None,
                 # 为了兼容旧 cfg，保留 img_suffix 参数，但这里不再使用
                 img_suffix='.jpg',
                 img_suffix_u=None,
                 classes=None,
                 filter_empty_gt=True):
        super().__init__()

        # 标注集
        self.dota_labeled = ROTATED_DATASETS.build(dict(
            type='DOTADatasetAnySuffix',
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            test_mode=False,
            filter_empty_gt=filter_empty_gt,
            classes=classes,
        ))

        # 未标注集（共享弱几何增强）
        self.dota_unlabeled = ROTATED_DATASETS.build(dict(
            type='DOTADatasetAnySuffix',
            ann_file=ann_file_u,
            pipeline=pipeline_u_share,
            data_root=data_root_u,
            img_prefix=img_prefix_u,
            test_mode=False,
            filter_empty_gt=False,
            classes=classes,
        ))

        self.CLASSES = classes
        self.pipeline_u   = Compose(pipeline_u or [])
        self.pipeline_u_1 = Compose(pipeline_u_1) if pipeline_u_1 else None

        # NOTE: mmcv/mmdet group samplers rely on `len(dataset.flag) == len(dataset)`.
        # This dataset's `__len__` follows the labeled set, so we must mirror the
        # labeled flag here; otherwise the sampler may silently truncate the epoch.
        self.flag = getattr(self.dota_labeled, 'flag', None)

    def __len__(self):
        return len(self.dota_labeled)

    def __getitem__(self, idx):
        # 有标注随机采一个，未标注按 idx 配对（防越界）
        idx_label = random.randint(0, len(self.dota_labeled) - 1)
        results = self.dota_labeled[idx_label]

        u_idx = idx % len(self.dota_unlabeled)
        results_u = self.dota_unlabeled[u_idx]

        # student 强增强分支（如果有）
        if self.pipeline_u_1:
            results_u_1 = copy.deepcopy(results_u)
            results_u_1 = self.pipeline_u_1(results_u_1)
            results.update({f'{k}_unlabeled_1': v for k, v in results_u_1.items()})

        # teacher 弱增强分支
        results_u = self.pipeline_u(results_u)
        results.update({f'{k}_unlabeled': v for k, v in results_u.items()})
        return results
