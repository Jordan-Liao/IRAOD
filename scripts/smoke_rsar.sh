#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-iraod}"
CONFIG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/RSAR}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_smoke_rsar}"
VIS_DIR="${VIS_DIR:-work_dirs/vis_rsar}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/rsar}"

N_TRAIN="${N_TRAIN:-50}"
N_VAL="${N_VAL:-50}"
N_TEST="${N_TEST:-50}"

# For future RSAR-Interference switching: CORRUPT=interf_jamA, etc.
CORRUPT="${CORRUPT:-clean}"

mkdir -p "${WORK_DIR}" "${VIS_DIR}" "${SPLIT_DIR}"

export DATA_ROOT SPLIT_DIR N_TRAIN N_VAL N_TEST

python - <<'PY'
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"]).resolve()
split_root = Path(os.environ["SPLIT_DIR"]).resolve()

counts = {
    "train": int(os.environ["N_TRAIN"]),
    "val": int(os.environ["N_VAL"]),
    "test": int(os.environ["N_TEST"]),
}

allowed_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
prio = {".jpg": 0, ".jpeg": 1, ".png": 2, ".bmp": 3, ".tif": 4, ".tiff": 5}


def pick_image(cands: list[Path]) -> Path:
    if not cands:
        raise RuntimeError("no candidates")
    return sorted(cands, key=lambda p: prio.get(p.suffix.lower(), 99))[0]


def safe_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def build_split(split: str, n: int) -> None:
    src_ann_dir = data_root / split / "annfiles"
    src_img_dir = data_root / split / "images"
    if not src_ann_dir.is_dir():
        raise SystemExit(f"missing ann dir: {src_ann_dir}")
    if not src_img_dir.is_dir():
        raise SystemExit(f"missing img dir: {src_img_dir}")

    dst_ann_dir = split_root / split / "annfiles"
    dst_img_dir = split_root / split / "images"
    dst_ann_dir.mkdir(parents=True, exist_ok=True)
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(src_ann_dir.glob("*.txt"))
    if len(ann_files) < n:
        raise SystemExit(f"{split}: requested n={n} but only {len(ann_files)} annfiles under {src_ann_dir}")

    picked = ann_files[:n]
    missing = 0
    conflict = 0
    for ann_path in picked:
        stem = ann_path.stem
        # Resolve images by stem + allowed extensions.
        cands = [p for p in src_img_dir.glob(stem + ".*") if p.suffix.lower() in allowed_exts]
        if not cands:
            missing += 1
            continue
        img_path = pick_image(cands)
        if len({p.suffix.lower() for p in cands}) > 1:
            conflict += 1

        safe_symlink(dst_ann_dir / ann_path.name, ann_path.resolve())
        safe_symlink(dst_img_dir / img_path.name, img_path.resolve())

    if missing != 0:
        raise SystemExit(f"{split}: missing images for {missing}/{n} annfiles (expected 0)")
    print(f"[smoke_rsar] split={split} n={n} conflicts(multi-suffix)={conflict}")


for s, n in counts.items():
    build_split(s, n)
print(f"[smoke_rsar] smoke split root: {split_root}")
PY

if [[ "${CORRUPT}" != "clean" && "${CORRUPT}" != "none" && "${CORRUPT}" != "" ]]; then
  echo "[smoke_rsar] prepare corrupt switch images-${CORRUPT} symlinks under ${SPLIT_DIR} ..."
  for s in train val test; do
    link_path="${SPLIT_DIR}/${s}/images-${CORRUPT}"
    if [[ ! -e "${link_path}" ]]; then
      ln -s images "${link_path}"
    fi
  done
fi

TRAIN_ANN_DIR="${SPLIT_DIR}/train/annfiles"
TRAIN_IMG_DIR="${SPLIT_DIR}/train/images"
VAL_ANN_DIR="${SPLIT_DIR}/val/annfiles"
VAL_IMG_DIR="${SPLIT_DIR}/val/images"
TEST_ANN_DIR="${SPLIT_DIR}/test/annfiles"
TEST_IMG_DIR="${SPLIT_DIR}/test/images"

echo "[smoke_rsar] train (ENV=${ENV_NAME}) ..."
conda run -n "${ENV_NAME}" python train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    corrupt="${CORRUPT}" \
    load_from=None \
    model.ema_ckpt=None \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=0 \
    runner.max_epochs=1 \
    lr_config.step=[1] \
    data.train.ann_file="${TRAIN_ANN_DIR}" \
    data.train.img_prefix="${TRAIN_IMG_DIR}" \
    data.train.ann_file_u="${VAL_ANN_DIR}" \
    data.train.img_prefix_u="${VAL_IMG_DIR}" \
    data.val.ann_file="${TEST_ANN_DIR}" \
    data.val.img_prefix="${TEST_IMG_DIR}" \
    data.test.ann_file="${TEST_ANN_DIR}" \
    data.test.img_prefix="${TEST_IMG_DIR}"

echo "[smoke_rsar] test (ENV=${ENV_NAME}) ..."
conda run -n "${ENV_NAME}" python test.py \
  "${CONFIG}" \
  "${WORK_DIR}/latest.pth" \
  --eval mAP \
  --show-dir "${VIS_DIR}" \
  --cfg-options \
    corrupt="${CORRUPT}" \
    load_from=None \
    model.ema_ckpt=None \
    data.test.ann_file="${TEST_ANN_DIR}" \
    data.test.img_prefix="${TEST_IMG_DIR}"

echo "[smoke_rsar] done: ${WORK_DIR} (vis: ${VIS_DIR})"
