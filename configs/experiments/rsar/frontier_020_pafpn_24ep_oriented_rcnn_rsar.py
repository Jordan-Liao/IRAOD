_base_ = "./frontier_008_24ep_oriented_rcnn_rsar.py"

# Single-axis method change: swap neck FPN -> PAFPN (keep channels/outs identical).
model = dict(
    neck=dict(
        type="PAFPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    )
)
