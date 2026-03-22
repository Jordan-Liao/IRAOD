_base_ = './baseline_oriented_rcnn_rsar.py'

# frontier-004: Replace RCNN bbox SmoothL1Loss with GDLoss(gwd)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(_delete_=True, type='GDLoss', loss_type='gwd', loss_weight=5.0),
        ),
    ),
)
