_base_ = "./frontier_008_24ep_oriented_rcnn_rsar.py"

# Ensure the custom neck is registered.
custom_imports = dict(
    imports=["sfod", "mmrotate.datasets.pipelines", "mmdet_extension"],
    allow_failed_imports=False,
)

# Single-axis method change: swap neck FPN -> OCAFPN (keep channels/outs identical).
model = dict(
    neck=dict(
        type="OCAFPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        reduction=16,
    )
)
