"""frontier-005: PolyRandomRotate augmentation experiment.

Inherits the full baseline and overrides only train_pipeline to insert
PolyRandomRotate(rotate_ratio=0.5, angles_range=180) after RRandomFlip.
All other settings (arch, backbone, optimizer, schedule, losses, NMS) are
identical to baseline_oriented_rcnn_rsar.py.
"""

_base_ = "./baseline_oriented_rcnn_rsar.py"

angle_version = "le90"

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
    dict(
        type="PolyRandomRotate",
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

# Override the train dataset pipeline
data = dict(train=dict(pipeline=train_pipeline))
