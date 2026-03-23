#!/usr/bin/env bash
set -euo pipefail

# Evaluate RSAR test mAP for a sweep of RCNN rotated NMS IoU thresholds.
# Defaults follow `.research/graph.json` spec-016.

CKPT=${1:-work_dirs/exp_rsar_baseline/epoch_11.pth}
CONFIG=${2:-configs/experiments/rsar/frontier_003_nms02_oriented_rcnn_rsar.py}
DATA_ROOT=${RSAR_DATA_ROOT:-/home/zechuan/IRAOD/dataset/RSAR}
OUT_ROOT=${3:-work_dirs/frontier_014_nms_sweep}

THRS=(0.15 0.20 0.25 0.30)

mkdir -p "$OUT_ROOT"
RESULTS_TSV="$OUT_ROOT/results.tsv"

echo -e "thr\tmAP" | tee "$RESULTS_TSV" >/dev/null

for thr in "${THRS[@]}"; do
  thr_tag=${thr/./}
  work_dir="$OUT_ROOT/thr_${thr_tag}"
  mkdir -p "$work_dir"

  echo "[frontier-014] Evaluating iou_thr=$thr ..."

  log_path="$work_dir/stdout.log"
  bash -lc "source /home/zechuan/miniconda3/etc/profile.d/conda.sh && \
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n iraod \
python test.py $CONFIG $CKPT \
  --eval mAP \
  --work-dir $work_dir \
  --data-root $DATA_ROOT \
  --cfg-options model.test_cfg.rcnn.nms.iou_thr=$thr" 2>&1 | tee "$log_path"

  map_val=$(python - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
# MMRotate typically prints 'mAP: <float>'
ms = re.findall(r"\bmAP\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
if not ms:
    # Fallback: last float on a line containing 'mAP'
    for line in reversed(text.splitlines()):
        if 'mAP' in line:
            m = re.search(r"([0-9]*\.?[0-9]+)", line)
            if m:
                print(m.group(1))
                raise SystemExit(0)
    raise SystemExit("Could not parse mAP from log")
print(ms[-1])
PY
)

  echo -e "$thr\t$map_val" | tee -a "$RESULTS_TSV" >/dev/null

done

echo "[frontier-014] Sweep complete. Results: $RESULTS_TSV"
# Print best as a machine-friendly line
best_line=$(python - "$RESULTS_TSV" <<'PY'
import sys
from pathlib import Path

rows = [l.strip().split('\t') for l in Path(sys.argv[1]).read_text().splitlines()[1:] if l.strip()]
best = max(((float(m), t) for t, m in rows), key=lambda x: x[0])
print(f"BEST_THR={best[1]} BEST_mAP={best[0]}")
PY
)
echo "$best_line"
