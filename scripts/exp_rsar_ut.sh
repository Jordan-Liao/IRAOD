#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-iraod}"
CONFIG="${CONFIG:-configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py}"
DATA_ROOT="${DATA_ROOT:-dataset/RSAR}"
WORK_DIR="${WORK_DIR:-work_dirs/exp_rsar_ut}"
VIS_DIR="${VIS_DIR:-work_dirs/vis_rsar_ut}"
SPLIT_DIR="${SPLIT_DIR:-work_dirs/smoke_splits/rsar_ut}"

SMOKE="${SMOKE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
N_TRAIN="${N_TRAIN:-50}"
N_VAL="${N_VAL:-50}"
N_TEST="${N_TEST:-50}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_TEST="${DO_TEST:-1}"
CKPT="${CKPT:-${WORK_DIR}/latest.pth}"

SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"

CORRUPT="${CORRUPT:-clean}"
export CGA_SCORER="${CGA_SCORER:-none}"
if [[ -z "${CGA_TEMPLATES:-}" ]]; then
  export CGA_TEMPLATES='an SAR image of a {}|this SAR patch shows a {}'
fi
TEACHER_CKPT="${TEACHER_CKPT:-}"
SUP_CLEAN="${SUP_CLEAN:-0}"

mkdir -p "${WORK_DIR}" "${VIS_DIR}" "${SPLIT_DIR}"

TRAIN_ANN_DIR="${DATA_ROOT}/train/annfiles"
TRAIN_IMG_DIR="${DATA_ROOT}/train/images"
VAL_ANN_DIR="${DATA_ROOT}/val/annfiles"
VAL_IMG_DIR="${DATA_ROOT}/val/images"
TEST_ANN_DIR="${DATA_ROOT}/test/annfiles"
TEST_IMG_DIR="${DATA_ROOT}/test/images"

if [[ "${SMOKE}" == "1" ]]; then
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
    print(f"[exp_rsar_ut] split={split} n={n}")


for s, n in counts.items():
    build_split(s, n)
print(f"[exp_rsar_ut] smoke split root: {split_root}")
PY

  TRAIN_ANN_DIR="${SPLIT_DIR}/train/annfiles"
  TRAIN_IMG_DIR="${SPLIT_DIR}/train/images"
  VAL_ANN_DIR="${SPLIT_DIR}/val/annfiles"
  VAL_IMG_DIR="${SPLIT_DIR}/val/images"
  TEST_ANN_DIR="${SPLIT_DIR}/test/annfiles"
  TEST_IMG_DIR="${SPLIT_DIR}/test/images"
fi

if [[ "${CORRUPT}" != "clean" && "${CORRUPT}" != "none" && "${CORRUPT}" != "" ]]; then
  echo "[exp_rsar_ut] prepare corrupt switch images-${CORRUPT} links under ${SPLIT_DIR} ..."
  for s in train val test; do
    link_path="${SPLIT_DIR}/${s}/images-${CORRUPT}"
    target_dir=""
    if [[ -d "${DATA_ROOT}/${s}/images-${CORRUPT}" ]]; then
      target_dir="$(cd "${DATA_ROOT}/${s}" && pwd)/images-${CORRUPT}"
    fi

    # Prefer the real corrupt directory under DATA_ROOT (if present) so smoke
    # subsets can still evaluate true interference images; otherwise fall back
    # to the clean subset directory (previous behavior).
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

TRAIN_IMG_DIR_SUP="${TRAIN_IMG_DIR}"
if [[ "${SUP_CLEAN}" == "1" && "${CORRUPT}" != "clean" && "${CORRUPT}" != "none" && "${CORRUPT}" != "" ]]; then
  # Keep supervised branch on clean images while val/test (and unlabeled) use corrupt images.
  clean_link="$(dirname "${TRAIN_IMG_DIR}")/images-clean"
  if [[ ! -e "${clean_link}" ]]; then
    ln -s images "${clean_link}"
  fi
  TRAIN_IMG_DIR_SUP="${clean_link}"
fi

echo "[exp_rsar_ut] train (ENV=${ENV_NAME}, CGA_SCORER=${CGA_SCORER}, corrupt=${CORRUPT}) ..."
if [[ -n "${TEACHER_CKPT}" ]]; then
  echo "[exp_rsar_ut] teacher init: ${TEACHER_CKPT}"
else
  echo "[exp_rsar_ut] teacher init: None"
fi

CFG_OPTS=(
  corrupt="${CORRUPT}"
  data.samples_per_gpu="${SAMPLES_PER_GPU}"
  data.workers_per_gpu="${WORKERS_PER_GPU}"
  runner.max_epochs="${MAX_EPOCHS}"
  lr_config.step="[$((${MAX_EPOCHS}))]"
  data.train.ann_file="${TRAIN_ANN_DIR}"
  data.train.img_prefix="${TRAIN_IMG_DIR_SUP}"
  data.train.ann_file_u="${VAL_ANN_DIR}"
  data.train.img_prefix_u="${VAL_IMG_DIR}"
  data.val.ann_file="${TEST_ANN_DIR}"
  data.val.img_prefix="${TEST_IMG_DIR}"
  data.test.ann_file="${TEST_ANN_DIR}"
  data.test.img_prefix="${TEST_IMG_DIR}"
)

if [[ "${DO_TRAIN}" == "1" ]]; then
  if [[ -n "${TEACHER_CKPT}" && "${TEACHER_CKPT}" != "none" ]]; then
    CFG_OPTS+=(
      load_from="${TEACHER_CKPT}"
      model.ema_ckpt="${TEACHER_CKPT}"
    )
  else
    CFG_OPTS+=(
      load_from=None
      model.ema_ckpt=None
    )
  fi

  conda run -n "${ENV_NAME}" python train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options "${CFG_OPTS[@]}"
else
  echo "[exp_rsar_ut] skip train (DO_TRAIN=0)"
fi

if [[ "${DO_TEST}" == "1" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "[exp_rsar_ut] ERROR: ckpt not found: ${CKPT}" >&2
    exit 2
  fi

  echo "[exp_rsar_ut] test (ENV=${ENV_NAME}, CKPT=${CKPT}) ..."
  conda run -n "${ENV_NAME}" python test.py "${CONFIG}" "${CKPT}" \
    --eval mAP \
    --work-dir "${WORK_DIR}" \
    --show-dir "${VIS_DIR}" \
    --cfg-options \
      corrupt="${CORRUPT}" \
      load_from=None \
      model.ema_ckpt=None \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
      data.test.ann_file="${TEST_ANN_DIR}" \
      data.test.img_prefix="${TEST_IMG_DIR}"
else
  echo "[exp_rsar_ut] skip test (DO_TEST=0)"
fi

echo "[exp_rsar_ut] done: ${WORK_DIR}"
