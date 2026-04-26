custom_imports = dict(
    imports=[
        "sfod",
        "mmrotate.datasets.pipelines",
        "mmdet_extension",
        "tools.rsar_sfodrs_dataset",
        "tools.sfodrs_diagnostics_hook",
    ],
    allow_failed_imports=False,
)

import os
import os.path as osp
import torchvision.transforms as transforms

# ------------------ Runtime switches (env, evaluated at config load time) ------------------
# IMPORTANT: This config branches on env vars (not --cfg-options), because mmcv merges
# cfg-options *after* executing the python config file.
#
# Stage protocol:
# - source_train: train source detector on clean labeled RSAR train
# - source_clean_test: eval source ckpt on clean test
# - direct_test: eval source ckpt on one corruption test
# - target_adapt: target-only self-training on one corruption val (unlabeled), eval on that corruption test
# - target_eval: eval adapted ckpt on one corruption test
stage = os.environ.get("RSAR_STAGE", "source_train").strip()
target_domain = os.environ.get("RSAR_TARGET_DOMAIN", "clean").strip()
use_cga = os.environ.get("RSAR_USE_CGA", "0").strip().lower() in ("1", "true", "yes", "y", "on")

# ------------------ Dataset paths (SFOD-RS protocol compliant layout) ------------------
_repo_root = osp.abspath(osp.join("{{ fileDirname }}", "..", "..", ".."))
_rsar_root = os.environ.get("RSAR_DATA_ROOT", "").strip()
if _rsar_root:
    data_root = osp.abspath(osp.expanduser(_rsar_root)) + "/"
else:
    data_root = osp.abspath(osp.join(_repo_root, "dataset", "RSAR")) + "/"

clean_train_img = data_root + "train/images/"
clean_train_ann = data_root + "train/annfiles/"
clean_val_img = data_root + "val/images/"
clean_val_ann = data_root + "val/annfiles/"
clean_test_img = data_root + "test/images/"
clean_test_ann = data_root + "test/annfiles/"

def _is_clean_domain(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in ("", "clean", "none")


if _is_clean_domain(target_domain):
    target_val_img = clean_val_img
    target_test_img = clean_test_img
else:
    target_val_img = data_root + f"corruptions/{target_domain}/val/images/"
    target_test_img = data_root + f"corruptions/{target_domain}/test/images/"

# RSAR 六类
classes = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
num_classes = len(classes)

# ------------------ Common configs ------------------
angle_version = "le90"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (800, 800)

samples_per_gpu = int(os.environ.get("RSAR_SAMPLES_PER_GPU", "2").strip() or "2")
workers_per_gpu = int(os.environ.get("RSAR_WORKERS_PER_GPU", "2").strip() or "2")

evaluation = dict(interval=1, metric="mAP")
log_level = "INFO"
dist_params = dict(backend="nccl")
resume_from = None
load_from = None
workflow = [("train", 1)]

# ------------------ Pipelines ------------------
# Source supervised training (clean labeled)
source_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=image_size),
    dict(
        type="RRandomFlip",
        flip_ratio=[0.25, 0.25, 0.25],
        direction=["horizontal", "vertical", "diagonal"],
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type="RResize"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# SFOD-RS target-only self-training pipelines:
# - weak: only horizontal flip p=0.5 (shared) + resize/normalize
# - strong: color jitter + grayscale + gaussian blur + cutout (DTRandCrop)
sfodrs_share = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(
        type="RRandomFlip",
        flip_ratio=0.5,
        direction=["horizontal"],
        version=angle_version,
    ),
]

_meta_keys = (
    "filename",
    "ori_filename",
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
    "img_norm_cfg",
)

sfodrs_weak = [
    dict(type="RResize", img_scale=image_size),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"], meta_keys=_meta_keys),
]

sfodrs_strong = [
    dict(type="DTToPILImage"),
    dict(type="DTRandomApply", operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type="DTRandomGrayscale", p=0.2),
    dict(type="DTRandomApply", operations=[dict(type="DTGaussianBlur", rad_range=[0.1, 2.0])]),
    dict(type="DTRandCrop"),
    dict(type="DTToNumpy"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"], meta_keys=_meta_keys),
]

# ------------------ Architecture (student + teacher must match) ------------------
_backbone = dict(
    type="OrthoNet",
    depth=50,
    in_channels=3,
    base_channels=64,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type="BN", requires_grad=True),
    norm_eval=True,
    with_cp=False,
    zero_init_residual=True,
    reduction=16,
    init_cfg=None,
)

_neck = dict(
    type="OCAFPN",
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    reduction=16,
)

_rpn_head = dict(
    type="OrientedRPNHead",
    in_channels=256,
    feat_channels=256,
    version=angle_version,
    anchor_generator=dict(type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(
        type="MidpointOffsetCoder",
        angle_range=angle_version,
        target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    ),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
)

_roi_head = dict(
    type="OrientedStandardRoIHead",
    bbox_roi_extractor=dict(
        type="RotatedSingleRoIExtractor",
        roi_layer=dict(type="RoIAlignRotated", out_size=7, sample_num=2, clockwise=True),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    bbox_head=dict(
        type="RotatedShared2FCBBoxHead",
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=num_classes,
        bbox_coder=dict(
            type="DeltaXYWHAOBBoxCoder",
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
        ),
        reg_class_agnostic=True,
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
    ),
)

_train_cfg = dict(
    rpn=dict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    rpn_proposal=dict(nms_pre=2000, max_per_img=2000, nms=dict(type="nms", iou_threshold=0.8), min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            iou_calculator=dict(type="RBboxOverlaps2D"),
            ignore_iof_thr=-1,
        ),
        sampler=dict(type="RRandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False,
    ),
)

_test_cfg = dict(
    rpn=dict(nms_pre=2000, max_per_img=2000, nms=dict(type="nms", iou_threshold=0.8), min_bbox_size=0),
    rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000),
)

# ------------------ Stage-specific config ------------------
if stage == "source_train":
    runner = dict(type="EpochBasedRunner", max_epochs=12)
    lr_config = dict(policy="step", warmup="linear", warmup_iters=100, warmup_ratio=0.001, step=[8, 11])
    optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
    auto_scale_lr = dict(enable=True, base_batch_size=16)
    optimizer_config = dict(type="SkipNanOptimizerHook", grad_clip=dict(max_norm=35, norm_type=2), max_skips=5)
    checkpoint_config = dict(interval=1)
    log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

    data = dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_train_ann,
            img_prefix=clean_train_img,
            classes=classes,
            pipeline=source_train_pipeline,
        ),
        val=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_val_ann,
            img_prefix=clean_val_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
        test=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_test_ann,
            img_prefix=clean_test_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
    )

    model = dict(
        type="OrientedRCNN",
        backbone=_backbone,
        neck=_neck,
        rpn_head=_rpn_head,
        roi_head=_roi_head,
        train_cfg=_train_cfg,
        test_cfg=_test_cfg,
    )

    custom_hooks = [
        dict(
            type="SFODRSDiagnosticsHook",
            stage=stage,
            target_domain="clean",
            use_labeled_source_in_adaptation=True,
            cga_enabled=False,
        )
    ]

elif stage in ("source_clean_test", "direct_test", "target_eval"):
    # Evaluation-only stages (test.py)
    runner = dict(type="EpochBasedRunner", max_epochs=1)  # compat for test.py

    eval_img = clean_test_img if stage == "source_clean_test" else target_test_img

    data = dict(
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        val=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_test_ann,
            img_prefix=eval_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
        test=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_test_ann,
            img_prefix=eval_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
    )

    model = dict(
        type="OrientedRCNN",
        backbone=_backbone,
        neck=_neck,
        rpn_head=_rpn_head,
        roi_head=_roi_head,
        train_cfg=_train_cfg,
        test_cfg=_test_cfg,
    )

    custom_hooks = [
        dict(
            type="SFODRSDiagnosticsHook",
            stage=stage,
            target_domain=("clean" if stage == "source_clean_test" else target_domain),
            use_labeled_source_in_adaptation=False,
            cga_enabled=False,
        )
    ]

elif stage == "target_adapt":
    # Target-only self-training (SFOD-RS) on ONE corruption.
    total_epoch = int(os.environ.get("RSAR_ADAPT_EPOCHS", "12").strip() or "12")
    score_thr = float(os.environ.get("RSAR_PSEUDO_SCORE_THR", "0.7").strip() or "0.7")
    evaluation = dict(interval=1, metric="mAP", only_ema=True)

    runner = dict(type="SemiEpochBasedRunner", max_epochs=total_epoch)
    lr_config = dict(policy="step", warmup="linear", warmup_iters=100, warmup_ratio=0.001, step=[total_epoch])
    optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
    auto_scale_lr = dict(enable=True, base_batch_size=32)
    optimizer_config = dict(
        type="SkipNanOptimizerHook",
        grad_clip=dict(max_norm=35, norm_type=2),
        max_skips=20,
    )
    checkpoint_config = dict(interval=1)
    log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])

    ema_config = (
        "./configs/baseline/ema_config/sfodrs_oriented_rcnn_ema_rsar_cga.py"
        if use_cga
        else "./configs/baseline/ema_config/sfodrs_oriented_rcnn_ema_rsar.py"
    )

    # NOTE: annfiles are NOT used for adaptation data (unlabeled).
    data = dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        train=dict(
            type="RSARSourceFreeSelfTrainingDataset",
            img_prefix=target_val_img,
            pipeline_share=sfodrs_share,
            pipeline_weak=sfodrs_weak,
            pipeline_strong=sfodrs_strong,
            classes=classes,
        ),
        val=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_test_ann,
            img_prefix=target_test_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
        test=dict(
            type="DOTADatasetAnySuffix",
            ann_file=clean_test_ann,
            img_prefix=target_test_img,
            classes=classes,
            pipeline=test_pipeline,
        ),
    )

    load_from = None
    model = dict(
        type="UnbiasedTeacher",
        ema_config=ema_config,
        ema_ckpt=load_from,
        cfg=dict(
            momentum=0.998,  # SFOD-RS EMA alpha
            weight_l=0.0,  # STRICT: no supervised source branch in adaptation
            use_labeled=False,
            weight_u=1.0,
            debug=False,
            score_thr=score_thr,
            use_bbox_reg=False,
        ),
        backbone=_backbone,
        neck=_neck,
        rpn_head=_rpn_head,
        roi_head=_roi_head,
        train_cfg=_train_cfg,
        test_cfg=_test_cfg,
    )

    custom_hooks = [
        dict(type="SetEpochInfoHook"),
        dict(
            type="SFODRSDiagnosticsHook",
            stage=stage,
            target_domain=target_domain,
            use_labeled_source_in_adaptation=False,
            cga_enabled=use_cga,
            cga_mode="sfodrs",
            prompt_template="A SAR image of a {}",
            keep_label=True,
            score_rule="0.7*teacher + 0.3*clip_prob_orig",
        ),
    ]

else:
    raise ValueError(f"Unknown RSAR_STAGE={stage!r}")
