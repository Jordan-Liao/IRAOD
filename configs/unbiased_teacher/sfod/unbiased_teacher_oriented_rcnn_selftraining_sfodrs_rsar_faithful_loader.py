# Faithful SFOD-RS dataloader for RSAR adaptation.
# Equivalent to running the base config with RSAR_LOADER_MODE=loose.
# Usage: RSAR_LOADER_MODE=loose ... python train.py configs/current/rsar_sfodrs_faithful.py ...
_base_ = ['./unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py']
