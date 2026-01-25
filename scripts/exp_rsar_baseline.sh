#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dino_sar}"
CONFIG="${CONFIG:-configs/experiments/rsar/baseline_oriented_rcnn_rsar.py}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_rsar_baseline}"
VIS_DIR="${VIS_DIR:-work_dirs/vis_rsar_baseline}"
DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/rsar_baseline}"

# Default to a quick smoke subset; set SMOKE=0 for full data.
SMOKE="${SMOKE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_TEST="${DO_TEST:-1}"
CKPT="${CKPT:-${WORK_DIR}/latest.pth}"

SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"

N_TRAIN="${N_TRAIN:-50}"
N_VAL="${N_VAL:-50}"
N_TEST="${N_TEST:-50}"

CORRUPT="${CORRUPT:-clean}"
MIX_TRAIN="${MIX_TRAIN:-0}"
MIX_TRAIN_CLEAN_TIMES="${MIX_TRAIN_CLEAN_TIMES:-1}"
MIX_TRAIN_CORRUPT_TIMES="${MIX_TRAIN_CORRUPT_TIMES:-1}"

mkdir -p "${WORK_DIR}" "${VIS_DIR}"

TRAIN_ANN_DIR="${DATA_ROOT}/train/annfiles"
TRAIN_IMG_DIR="${DATA_ROOT}/train/images"
VAL_ANN_DIR="${DATA_ROOT}/val/annfiles"
VAL_IMG_DIR="${DATA_ROOT}/val/images"
TEST_ANN_DIR="${DATA_ROOT}/test/annfiles"
TEST_IMG_DIR="${DATA_ROOT}/test/images"

if [[ "${SMOKE}" == "1" ]]; then
  mkdir -p "${SPLIT_DIR}"
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

    # Ensure this split is exactly the requested subset (avoid leftover symlinks
    # from previous runs with a different N).
    for p in dst_ann_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
    for p in dst_img_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()

    ann_files = sorted(src_ann_dir.glob("*.txt"))
    if len(ann_files) < n:
        raise SystemExit(f"{split}: requested n={n} but only {len(ann_files)} annfiles under {src_ann_dir}")

    picked = ann_files[:n]
    missing = 0
    for ann_path in picked:
        stem = ann_path.stem
        cands = [p for p in src_img_dir.glob(stem + ".*") if p.suffix.lower() in allowed_exts]
        if not cands:
            missing += 1
            continue
        img_path = pick_image(cands)

        safe_symlink(dst_ann_dir / ann_path.name, ann_path.resolve())
        safe_symlink(dst_img_dir / img_path.name, img_path.resolve())

    if missing != 0:
        raise SystemExit(f"{split}: missing images for {missing}/{n} annfiles (expected 0)")
    print(f"[exp_rsar_baseline] split={split} n={n}")


for s, n in counts.items():
    build_split(s, n)
print(f"[exp_rsar_baseline] smoke split root: {split_root}")
PY

  TRAIN_ANN_DIR="${SPLIT_DIR}/train/annfiles"
  TRAIN_IMG_DIR="${SPLIT_DIR}/train/images"
  VAL_ANN_DIR="${SPLIT_DIR}/val/annfiles"
  VAL_IMG_DIR="${SPLIT_DIR}/val/images"
  TEST_ANN_DIR="${SPLIT_DIR}/test/annfiles"
  TEST_IMG_DIR="${SPLIT_DIR}/test/images"
fi

if [[ "${CORRUPT}" != "clean" && "${CORRUPT}" != "none" && "${CORRUPT}" != "" ]]; then
  echo "[exp_rsar_baseline] prepare corrupt switch images-${CORRUPT} links under ${SPLIT_DIR} ..."
  for s in train val test; do
    mkdir -p "${SPLIT_DIR}/${s}"
    link_path="${SPLIT_DIR}/${s}/images-${CORRUPT}"
    target_dir=""
    if [[ -d "${DATA_ROOT}/${s}/images-${CORRUPT}" ]]; then
      target_dir="$(cd "${DATA_ROOT}/${s}" && pwd)/images-${CORRUPT}"
    fi

    if [[ -n "${target_dir}" ]]; then
      if [[ -L "${link_path}" ]]; then
        rm -f "${link_path}"
      fi
      if [[ ! -e "${link_path}" ]]; then
        ln -s "${target_dir}" "${link_path}"
      fi
    else
      if [[ ! -e "${link_path}" ]]; then
        ln -s images "${link_path}"
      fi
    fi
  done
fi

# Ensure images-clean exists for mix_train (and won't be patched by corrupt switch).
if [[ "${MIX_TRAIN}" == "1" ]]; then
  for s in train; do
    base_dir="$(dirname "${TRAIN_IMG_DIR}")"
    clean_link="${base_dir}/images-clean"
    if [[ ! -e "${clean_link}" ]]; then
      ln -s images "${clean_link}"
    fi
  done
fi

echo "[exp_rsar_baseline] train (ENV=${ENV_NAME}, corrupt=${CORRUPT}, mix_train=${MIX_TRAIN}) ..."
if [[ "${DO_TRAIN}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
    --cfg-options \
      corrupt="${CORRUPT}" \
      mix_train="${MIX_TRAIN}" \
      mix_train_clean_times="${MIX_TRAIN_CLEAN_TIMES}" \
      mix_train_corrupt_times="${MIX_TRAIN_CORRUPT_TIMES}" \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
      runner.max_epochs="${MAX_EPOCHS}" \
      lr_config.step="[$((${MAX_EPOCHS}))]" \
      data.train.ann_file="${TRAIN_ANN_DIR}" \
      data.train.img_prefix="${TRAIN_IMG_DIR}" \
      data.val.ann_file="${VAL_ANN_DIR}" \
      data.val.img_prefix="${VAL_IMG_DIR}" \
      data.test.ann_file="${TEST_ANN_DIR}" \
      data.test.img_prefix="${TEST_IMG_DIR}"
else
  echo "[exp_rsar_baseline] skip train (DO_TRAIN=0)"
fi

if [[ "${DO_TEST}" == "1" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "[exp_rsar_baseline] ERROR: ckpt not found: ${CKPT}" >&2
    exit 2
  fi

  echo "[exp_rsar_baseline] test (ENV=${ENV_NAME}, corrupt=${CORRUPT}, CKPT=${CKPT}) ..."
  conda run -n "${ENV_NAME}" python test.py "${CONFIG}" "${CKPT}" \
    --eval mAP \
    --work-dir "${WORK_DIR}" \
    --show-dir "${VIS_DIR}" \
    --cfg-options \
      corrupt="${CORRUPT}" \
      mix_train=0 \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
      data.test.ann_file="${TEST_ANN_DIR}" \
      data.test.img_prefix="${TEST_IMG_DIR}"
else
  echo "[exp_rsar_baseline] skip test (DO_TEST=0)"
fi

echo "[exp_rsar_baseline] done: ${WORK_DIR}"
