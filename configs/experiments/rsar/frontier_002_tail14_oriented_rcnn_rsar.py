_base_ = "./baseline_oriented_rcnn_rsar.py"

lr_config = dict(step=[8, 12])
runner = dict(max_epochs=14)
