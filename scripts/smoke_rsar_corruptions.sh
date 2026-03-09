#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke to validate the compliant RSAR corruption layout generator.
# It creates a tiny synthetic RSAR-like dataset under work_dirs/, generates the
# 7 corruption subsets, and runs the existing corrupt-switch verifier.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/work_dirs/sanity/rsar_corruption_smoke}"
DATA_ROOT="${OUT_ROOT}/rsar_smoke"
REPORT_ROOT="${OUT_ROOT}/reports"

mkdir -p "${OUT_ROOT}"

export DATA_ROOT
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np
import os

root = Path(os.environ["DATA_ROOT"]).resolve()
for split in ["train", "val", "test"]:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "annfiles").mkdir(parents=True, exist_ok=True)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 50
    arr[:, :, 1] = 100
    arr[:, :, 2] = 150
    Image.fromarray(arr).save(root / split / "images" / "000001.jpg")
    (root / split / "annfiles" / "000001.txt").write_text("dummy\n", encoding="utf-8")
print(f"[smoke_rsar_corruptions] created: {root}")
PY

python tools/prepare_rsar_corruption.py \
  --data-root "${DATA_ROOT}" \
  --workers 1 \
  --max-images 1 \
  --diff-samples 1

python tools/verify_rsar_corrupt_switch.py \
  --data-root "${DATA_ROOT}" \
  --corrupt chaff \
  --out-dir "${REPORT_ROOT}"

echo "[smoke_rsar_corruptions] OK: ${OUT_ROOT}"
