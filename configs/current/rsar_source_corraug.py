# Source training with online RSAR corruption augmentation (all 7 types, p=0.5 default).
# Usage:
#   RSAR_CORR_AUG=1 RSAR_CORR_AUG_PROB=0.5 \
#   RSAR_STAGE=source_train RSAR_TARGET_DOMAIN=clean \
#   python train.py configs/current/rsar_source_corraug.py --work-dir <work_dir>
#
# Or via full pipeline:
#   RSAR_CORR_AUG=1 bash scripts/run/rsar_sfodrs_full.sh auto <work_dir>
_base_ = ["../unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py"]
