# --- 必须项：确保注册我们在 sfod/semi_dota_dataset.py 里定义的两种数据集 ---
custom_imports = dict(imports=['sfod','mmrotate.datasets.pipelines'], allow_failed_imports=False)#

import torchvision.transforms as transforms
import os
import os.path as osp

# ------------------ 基本训练超参 ------------------
gpu = 1
score = 0.7
samples_per_gpu = 2
total_epoch = 12
test_interval = 1
save_interval = 1

# RSAR 六类
classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

# ------------------ 数据路径（DOTA/RSAR 目录结构） ------------------
# 默认使用仓库内的相对路径（推荐通过 `tools/verify_dataset_layout.py` 校验）：
#   dataset/RSAR/{train,val,test}/{images,annfiles}
#
# 也可在运行时覆盖：推荐用 `train.py/test.py --data-root /abs/path/to/RSAR`。
# 注意：mmcv 会把 config 复制到临时目录执行，所以这里用 `{{ fileDirname }}`
# 来拿到“原始 config 文件所在目录”的真实路径。
_repo_root = osp.abspath(osp.join('{{ fileDirname }}', '..', '..', '..'))
_rsar_root = os.environ.get("RSAR_DATA_ROOT", "").strip()
if _rsar_root:
    data_root = osp.abspath(osp.expanduser(_rsar_root)) + "/"
else:
    data_root = osp.abspath(osp.join(_repo_root, "dataset", "RSAR")) + "/"

train_img = data_root + 'train/images/'
train_ann = data_root + 'train/annfiles/'
val_img   = data_root + 'val/images/'
val_ann   = data_root + 'val/annfiles/'
test_img  = data_root + 'test/images/'
test_ann  = data_root + 'test/annfiles/'

# 所有图片后缀（RSAR 为 .jpg）
img_suffix_all = '.jpg'

# 角度表示
angle_version = 'le90'

# ------------------ 预处理/增强流水线 ------------------
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

image_size = (800, 800)

# 监督样本（有标注）
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    # 用 RLoadAnnotations（老版本 mmrotate 提供）直接读取旋转标注
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename','ori_filename','ori_shape','img_shape','pad_shape',
                    'scale_factor','flip','flip_direction','img_norm_cfg')),
]

# 无监督样本，共享几何变换分支（先读取 + 轻量几何）
unsup_pipeline_share = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal','vertical','diagonal'],
         version=angle_version),
]

# 无监督样本的弱增强（送 teacher）
unsup_pipeline_weak = [
    dict(type='RResize', img_scale=image_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 无监督样本的强增强（送 student）
unsup_pipeline_strong = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply',
         operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply',
         operations=[dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])]),
    dict(type='DTToNumpy'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename','ori_filename','ori_shape','img_shape','pad_shape',
                    'scale_factor','flip','flip_direction','img_norm_cfg')),
]

# 测试/验证
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# ------------------ 数据字典 ------------------
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type='SemiDOTADataset',
        ann_file=train_ann,             # 直接给 annfiles 目录
        ann_file_u=val_ann,             # 这里用 val 作为“无标注”源（可按需换）
        pipeline=sup_pipeline,
        pipeline_u_share=unsup_pipeline_share,
        pipeline_u=unsup_pipeline_weak,
        pipeline_u_1=unsup_pipeline_strong,
        img_prefix=train_img,
        img_prefix_u=val_img,
        # img_suffix=img_suffix_all,
        classes=classes,
    ),
    val=dict(
        # 用我们注册的带后缀控制的 DOTA 数据集
        type='DOTADatasetAnySuffix',
        ann_file=test_ann,
        img_prefix=test_img,
        # img_suffix=img_suffix_all,      # 验证/测试也指定 .jpg
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='DOTADatasetAnySuffix',
        ann_file=test_ann,
        img_prefix=test_img,
        # img_suffix=img_suffix_all,
        classes=classes,
        pipeline=test_pipeline),
)

# ------------------ 评测 ------------------
evaluation = dict(interval=test_interval, metric='mAP')

# ------------------ 优化器 / 训练策略 ------------------
# Base lr=0.02 for total batch size 32 (e.g. 16 GPUs * 2 imgs/gpu).
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# Auto scale lr by total batch size (samples_per_gpu * world_size).
auto_scale_lr = dict(enable=True, base_batch_size=32)
optimizer_config = dict(
    type='SkipNanOptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2),
    max_skips=20,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[total_epoch]
)
runner = dict(type='SemiEpochBasedRunner', max_epochs=total_epoch)

checkpoint_config = dict(interval=save_interval)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook')]
)

custom_hooks = [dict(type='SetEpochInfoHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None

# ------------------ EMA / teacher ------------------
# 默认不指定 teacher-init。若需要 teacher-init，请通过脚本/命令行传入：
#   --cfg-options load_from=<ckpt> model.ema_ckpt=<ckpt>
load_from = None
ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py'
workflow = [('train', 1)]

# ------------------ 模型 ------------------
model = dict(
    type='UnbiasedTeacher',
    ema_config=ema_config,
    ema_ckpt=load_from,
    cfg=dict(
        # NOTE: RSAR 全量训练默认保留监督分支，避免纯伪标签训练导致性能塌陷。
        weight_l=1.0,
        weight_u=1,
        debug=False,
        score_thr=score,
        # 伪标签回归通常能显著提升定位质量（低 mAP 常见根因之一是 bbox 回归被关掉）。
        use_bbox_reg=True,
    ),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
              out_channels=256, num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256, feat_channels=256, version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder', angle_range=angle_version,
            target_means=[0., 0., 0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1., 0.5, 0.5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)
    ),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder', angle_range=angle_version,
                norm_factor=None, edge_swap=True, proj_xy=True,
                target_means=(0., 0., 0., 0., 0.),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                          min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000,
                          nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                          min_pos_iou=0.5, match_low_quality=False,
                          iou_calculator=dict(type='RBboxOverlaps2D'),
                          ignore_iof_thr=-1),
            sampler=dict(type='RRandomSampler', num=512, pos_fraction=0.25,
                         neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000,
                 nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05,
                  nms=dict(iou_thr=0.1), max_per_img=2000)
    )
)
