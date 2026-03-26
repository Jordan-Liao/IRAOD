# User Constraints — Method-Level Innovation Only

## FORBIDDEN: Hyperparameter Tuning
The following are NOT research contributions and MUST NOT be proposed as experiments:
- Learning rate value changes, schedule changes (step/cosine/warmup)
- Training epoch count changes (the baseline 24ep is already optimized)
- Batch size changes
- Weight decay changes
- NMS IoU threshold or score threshold changes (0.30 is already optimized)
- Optimizer type swaps (SGD↔Adam) without a method component
- Simple data augmentation toggles (random flip, rotation angle range)

## REQUIRED: Method-Level Innovation
Every experiment MUST introduce code changes to model/loss/training logic that constitute a publishable contribution. Examples:
- New or modified backbone (Swin-T, ConvNeXt, etc.)
- Feature Pyramid improvements (PAFPN, BiFPN, feature fusion modules)
- Attention mechanisms (deformable attention, CBAM, coordinate attention)
- Rotation-aware modules (RoI Transformer, angle encoding like CSL/KLD)
- Detection head improvements (decoupled head, task-aligned head, quality branch)
- Loss function innovations (KLD loss for rotated boxes, IoU-aware losses, label assignment like SimOTA/ATSS)
- Training strategy innovations with novel components (knowledge distillation, contrastive learning)
- Small object / dense scene techniques (multi-scale feature enhancement, context modeling)

## Baseline Anchor (MUST BEAT)
- Config: 24 epochs, step=[16,22], NMS IoU=0.30
- Model: OrientedRCNN + ResNet50 + FPN
- Test mAP: 0.701 (work_dirs/frontier_008_24ep/epoch_21.pth)
- Training recipe (LR, schedule, epochs) MUST remain the same — the method must prove itself under identical conditions

## Quality Test
Before proposing an experiment, ask: "Can this be described as a technical contribution in a paper?" If the answer is "we just changed a number", reject it.
