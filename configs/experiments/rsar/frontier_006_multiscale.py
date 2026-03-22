"""frontier-006: Multi-scale training [640,960] with fixed 800 test scale."""

_base_ = "./baseline_oriented_rcnn_rsar.py"

# Override train pipeline: random scale in [640, 960] instead of fixed 800
angle_version = "le90"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RResize", img_scale=[(640, 640), (960, 960)], multiscale_mode="range"),
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

# Test pipeline stays at fixed 800x800 for fair comparison
# (inherited from baseline, no override needed)

# Override data.train.pipeline
data = dict(
    train=dict(pipeline=train_pipeline),
)
