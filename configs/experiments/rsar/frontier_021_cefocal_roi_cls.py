_base_ = "./frontier_008_24ep_oriented_rcnn_rsar.py"

# Ensure the custom loss is registered.
custom_imports = dict(
    imports=["sfod", "mmrotate.datasets.pipelines", "mmdet_extension.models.loss"],
    allow_failed_imports=False,
)

# Single-axis change: RoI cls loss CE -> CE-Focal.
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type="CEFocalLoss",
                use_sigmoid=False,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            )
        )
    )
)
