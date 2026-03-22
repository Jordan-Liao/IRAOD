_base_ = "./baseline_oriented_rcnn_rsar.py"

# Override LR schedule: cosine annealing replacing step-decay [8,11]
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=0.001,
    min_lr=0,
)
