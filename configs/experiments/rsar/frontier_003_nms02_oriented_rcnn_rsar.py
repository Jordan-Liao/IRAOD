_base_ = "./baseline_oriented_rcnn_rsar.py"

model = dict(
    test_cfg=dict(
        rcnn=dict(
            nms=dict(iou_thr=0.2),
        ),
    ),
)
