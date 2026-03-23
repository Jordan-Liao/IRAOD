# Weight decay 10× (baseline: 1e-4 -> 1e-3)
_base_ = "./baseline_oriented_rcnn_rsar.py"

optimizer = dict(weight_decay=0.001)
