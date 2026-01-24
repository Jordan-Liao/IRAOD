custom_imports = dict(imports=["sfod", "mmrotate.datasets.pipelines"], allow_failed_imports=False)

# RSAR 6 classes
classes = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
num_classes = len(classes)

data_root = "dataset/RSAR/"
train_img = data_root + "train/images/"
train_ann = data_root + "train/annfiles/"
val_img = data_root + "val/images/"
val_ann = data_root + "val/annfiles/"
test_img = data_root + "test/images/"
test_ann = data_root + "test/annfiles/"

angle_version = "le90"

samples_per_gpu = 2
workers_per_gpu = 2

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (800, 800)

train_pipeline = [
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

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type="DOTADatasetAnySuffix",
        ann_file=train_ann,
        img_prefix=train_img,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type="DOTADatasetAnySuffix",
        ann_file=val_ann,
        img_prefix=val_img,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type="DOTADatasetAnySuffix",
        ann_file=test_ann,
        img_prefix=test_img,
        classes=classes,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric="mAP")

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy="step", warmup="linear", warmup_iters=100, warmup_ratio=0.001, step=[8, 11])
runner = dict(type="EpochBasedRunner", max_epochs=12)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

model = dict(
    type="OrientedRCNN",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
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
    ),
    roi_head=dict(
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
    ),
    train_cfg=dict(
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
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000, nms=dict(type="nms", iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000),
    ),
)

