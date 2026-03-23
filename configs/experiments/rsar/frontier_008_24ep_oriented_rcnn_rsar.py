_base_ = "./baseline_oriented_rcnn_rsar.py"

# Single-axis change: extend training schedule to 24 epochs and move LR step milestones.
runner = dict(type="EpochBasedRunner", max_epochs=24)
lr_config = dict(policy="step", warmup="linear", warmup_iters=100, warmup_ratio=0.001, step=[16, 22])
