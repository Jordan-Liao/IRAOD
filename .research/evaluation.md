# Evaluation Protocol — RSAR Oriented Object Detection

## Primary Metric
- **Name:** `mAP`
- **Direction:** higher_is_better
- **Dataset/Split:** RSAR `test`
- **Definition (MMRotate):** `DOTADataset.evaluate(metric='mAP', iou_thr=0.5)` → `eval_rbbox_map(..., iou_thr=0.5, use_07_metric=True)`.
  - Interpretable as **VOC-style AP@0.5 IoU** for rotated boxes (not COCO mAP@[.5:.95]).

## Anchor Baseline (MUST BEAT)
- **Training config (24ep schedule):** `configs/experiments/rsar/frontier_008_24ep_oriented_rcnn_rsar.py`
- **Canonical test-time NMS config patch:** `configs/experiments/rsar/frontier_015_nms03_oriented_rcnn_rsar.py` (sets `model.test_cfg.rcnn.nms.iou_thr=0.30`).
- **Checkpoint:** `work_dirs/frontier_008_24ep/epoch_21.pth`
- **Reported RSAR test mAP:** **0.701** (NMS IoU=0.30)

## Evaluation command (single GPU)
Use the same model config you trained with, but **force** NMS IoU=0.30 at test time:
```bash
bash -lc "source /home/zechuan/miniconda3/etc/profile.d/conda.sh && \
conda run --no-capture-output -n iraod \
python test.py configs/experiments/rsar/frontier_008_24ep_oriented_rcnn_rsar.py \
  work_dirs/<EXP_NAME>/epoch_<BEST>.pth \
  --eval mAP \
  --data-root /home/zechuan/IRAOD/dataset/RSAR \
  --cfg-options model.test_cfg.rcnn.nms.iou_thr=0.30"
```

## Training command (5-GPU DDP)
```bash
bash -lc "source /home/zechuan/miniconda3/etc/profile.d/conda.sh && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
conda run --no-capture-output -n iraod \
python -m torch.distributed.launch --nproc_per_node=5 --master_port=29501 \
train.py configs/experiments/rsar/frontier_008_24ep_oriented_rcnn_rsar.py \
  --work-dir work_dirs/<EXP_NAME> \
  --launcher pytorch \
  --data-root /home/zechuan/IRAOD/dataset/RSAR"
```

## Smoke / readiness probe
A short end-to-end sanity run exists:
```bash
bash -lc "source /home/zechuan/miniconda3/etc/profile.d/conda.sh && \
conda run --no-capture-output -n iraod bash scripts/smoke_rsar.sh"
```
(Uses a tiny subset split + 1 epoch; intended to validate environment + dataset wiring.)

## Expected duration (order-of-magnitude)
- 24 epochs on 5 GPUs is **multi-hour** (the repo’s existing note suggests ~7h on 5×A6000). Exact time depends on I/O and GPU model.
