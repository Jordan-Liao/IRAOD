_base_ = ['./baseline_oriented_rcnn_rsar.py']

# frontier-010: backbone lr_mult=0.1
# Fine-tune pretrained ResNet50 backbone at 10% of head LR
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1)
        )
    )
)
