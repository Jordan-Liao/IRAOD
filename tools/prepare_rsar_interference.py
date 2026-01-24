from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_TYPES = {
    "awgn",
    "noise_jamming",
    "corner_reflector",
    "chaff",
    "smart_noise_jamming",
    "noise_am_jamming",
}


def _log(msg: str) -> None:
    print(f"[prepare_rsar_interference] {msg}")


def _stable_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real image extension so OpenCV selects the right encoder.
    tmp = path.with_suffix(".tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), img)
    if not ok:
        raise IOError(f"failed to write: {tmp}")
    tmp.replace(path)


def _split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if img.ndim == 2:
        return img.astype(np.float32)[..., None], None
    if img.ndim != 3:
        raise ValueError(f"unsupported image shape: {img.shape}")

    h, w, c = img.shape
    if c == 4:
        base = img[..., :3].astype(np.float32)
        alpha = img[..., 3:].astype(np.float32)
        return base, alpha
    if c == 3:
        return img.astype(np.float32), None
    if c == 1:
        return img.astype(np.float32), None

    # Unknown channel count: keep first 3 as base, preserve 1 alpha channel if present.
    base = img[..., :3].astype(np.float32)
    alpha = img[..., 3:4].astype(np.float32) if c > 3 else None
    return base, alpha


def _merge_alpha(base: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if base.ndim == 2:
        base = base[..., None]
    if alpha is None:
        return base
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=2)
    return np.concatenate([base, alpha], axis=2)


def _apply_scalar_field(base: np.ndarray, field_2d: np.ndarray) -> np.ndarray:
    if base.ndim != 3:
        raise ValueError("base must be HxWxC")
    h, w, c = base.shape
    if field_2d.shape != (h, w):
        raise ValueError(f"field_2d must be shape (H,W), got {field_2d.shape} vs {(h, w)}")
    if c == 1:
        return (base[..., 0] + field_2d)[..., None]
    return base + field_2d[..., None]


def _clip_to_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)
    return np.clip(arr, 0.0, 255.0).astype(dtype)


def _parse_params(params_json: str) -> dict[str, Any]:
    s = (params_json or "").strip()
    if not s:
        return {}
    if s.startswith("@"):
        p = Path(s[1:]).expanduser()
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _ensure_real_dir(path: Path, *, force_replace_symlink: bool) -> None:
    if path.exists() and path.is_symlink():
        if not force_replace_symlink:
            raise SystemExit(
                f"refusing to write into symlink: {path} -> {path.resolve()}. "
                "Remove it or rerun with --force-replace-symlink."
            )
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _iter_images(img_dir: Path) -> Iterable[Path]:
    for root, _dirs, files in os.walk(img_dir):
        for fn in files:
            p = Path(root) / fn
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _to_locations(value: Any) -> list[tuple[float, float]]:
    locs: list[tuple[float, float]] = []
    if not value:
        return locs
    if not isinstance(value, list):
        raise ValueError("locations must be a list of [row_frac, col_frac]")
    for it in value:
        if not isinstance(it, (list, tuple)) or len(it) != 2:
            raise ValueError(f"invalid location entry: {it!r}")
        locs.append((float(it[0]), float(it[1])))
    return locs


def _noise_jamming_field(h: int, w: int, *, js_ratio_db: float, stripe_frequency: float, stripe_amplitude: float) -> np.ndarray:
    js_ratio_linear = 10 ** (float(js_ratio_db) / 10.0)
    amplitude = float(stripe_amplitude) * js_ratio_linear
    y = np.arange(h, dtype=np.float32)
    field_1d = amplitude * (np.sin(2.0 * np.pi * float(stripe_frequency) * y) + 1.0) / 2.0
    return np.tile(field_1d[:, None], (1, w)).astype(np.float32)


def _gaussian_mask(h: int, w: int, *, sigma_r: float, sigma_c: float) -> np.ndarray:
    sigma_r = max(float(sigma_r), 1.0)
    sigma_c = max(float(sigma_c), 1.0)
    rr, cc = np.mgrid[-h // 2 : h - h // 2, -w // 2 : w - w // 2]
    g = np.exp(-0.5 * ((rr / sigma_r) ** 2 + (cc / sigma_c) ** 2)).astype(np.float32)
    m = float(g.max())
    return (g / m) if m > 0 else np.zeros((h, w), dtype=np.float32)


def _apply_corner_reflector(
    base: np.ndarray,
    *,
    template_path: Path,
    locations: list[tuple[float, float]],
    intensity: float,
    blend_mode: str,
) -> np.ndarray:
    tmpl = _read_image(template_path)
    if tmpl.ndim == 3:
        tmpl = tmpl[..., :3]
        tmpl = cv2.cvtColor(tmpl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif tmpl.ndim != 2:
        raise ValueError(f"unsupported template shape: {tmpl.shape}")
    tmpl_f = tmpl.astype(np.float32) * float(intensity)

    h, w, c = base.shape
    th, tw = tmpl_f.shape
    th2, tw2 = th // 2, tw // 2
    out = base.copy()

    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)
        if not (0 <= r_center < h and 0 <= c_center < w):
            continue

        rs = max(0, r_center - th2)
        re = min(h, rs + th)
        cs = max(0, c_center - tw2)
        ce = min(w, cs + tw)

        trs = max(0, th2 - r_center)
        tcs = max(0, tw2 - c_center)
        tre = trs + (re - rs)
        tce = tcs + (ce - cs)

        tpatch = tmpl_f[trs:tre, tcs:tce]
        roi = out[rs:re, cs:ce]
        if blend_mode == "add":
            out[rs:re, cs:ce] = _apply_scalar_field(roi, tpatch)
        elif blend_mode == "max":
            if roi.shape[-1] == 1:
                out[rs:re, cs:ce] = np.maximum(roi[..., 0], tpatch)[..., None]
            else:
                out[rs:re, cs:ce] = np.maximum(roi, tpatch[..., None])
        else:
            raise ValueError("blendMode must be add|max")

    return out


def _apply_chaff(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    locations: list[tuple[float, float]],
    cloud_size: tuple[float, float],
    density_sigma_factor: float,
    noise_sigma: float,
) -> np.ndarray:
    h, w, _c = base.shape
    ch = max(1, int(float(cloud_size[0]) * h))
    cw = max(1, int(float(cloud_size[1]) * w))
    sr = max(ch * float(density_sigma_factor), 1.0)
    sc = max(cw * float(density_sigma_factor), 1.0)
    density = _gaussian_mask(ch, cw, sigma_r=sr, sigma_c=sc)

    out = base.copy()
    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)

        rs = max(0, r_center - ch // 2)
        cs = max(0, c_center - cw // 2)
        re = min(h, rs + ch)
        ce = min(w, cs + cw)
        ah, aw = re - rs, ce - cs
        if ah <= 0 or aw <= 0:
            continue

        mrs = max(0, ch // 2 - (r_center - rs))
        mcs = max(0, cw // 2 - (c_center - cs))
        m_re = mrs + ah
        m_ce = mcs + aw
        mask = density[mrs:m_re, mcs:m_ce]

        noise = rng.normal(0.0, float(noise_sigma), (ah, aw)).astype(np.float32) * mask
        roi = out[rs:re, cs:ce]
        out[rs:re, cs:ce] = _apply_scalar_field(roi, noise)

    return out


def _apply_smart_noise(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    locations: list[tuple[float, float]],
    noise_size: tuple[float, float],
    noise_sigma: float,
) -> np.ndarray:
    h, w, _c = base.shape
    nh = max(1, int(float(noise_size[0]) * h))
    nw = max(1, int(float(noise_size[1]) * w))

    out = base.copy()
    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)
        rs = max(0, r_center - nh // 2)
        cs = max(0, c_center - nw // 2)
        re = min(h, rs + nh)
        ce = min(w, cs + nw)
        ah, aw = re - rs, ce - cs
        if ah <= 0 or aw <= 0:
            continue

        noise = rng.normal(0.0, float(noise_sigma), (ah, aw)).astype(np.float32)
        roi = out[rs:re, cs:ce]
        out[rs:re, cs:ce] = _apply_scalar_field(roi, noise)

    return out


def _apply_noise_am_lines(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    line_frequency: float,
    base_intensity: float,
    noise_sigma: float,
    line_width: int,
    direction: str,
    blend_factor: float,
) -> np.ndarray:
    if float(line_frequency) <= 0:
        raise ValueError("lineFrequency must be > 0")
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be horizontal|vertical")
    if not (0.0 <= float(blend_factor) <= 1.0):
        raise ValueError("blendFactor must be within [0,1]")

    h, w, _c = base.shape
    out = base.copy()

    spacing = int(1.0 / float(line_frequency))
    hw = max(1, int(line_width) // 2)

    if direction == "horizontal":
        for r in range(0, h, spacing):
            rs = max(0, r - hw)
            re = min(h, r + hw)
            if re <= rs:
                continue
            patch_h = re - rs
            noise = rng.normal(float(base_intensity), float(noise_sigma), (patch_h, w)).astype(np.float32)
            roi = out[rs:re, :, :]
            roi_mix = roi * (1.0 - float(blend_factor))
            out[rs:re, :, :] = _apply_scalar_field(roi_mix, noise * float(blend_factor))
    else:
        for c in range(0, w, spacing):
            cs = max(0, c - hw)
            ce = min(w, c + hw)
            if ce <= cs:
                continue
            patch_w = ce - cs
            noise = rng.normal(float(base_intensity), float(noise_sigma), (h, patch_w)).astype(np.float32)
            roi = out[:, cs:ce, :]
            roi_mix = roi * (1.0 - float(blend_factor))
            out[:, cs:ce, :] = _apply_scalar_field(roi_mix, noise * float(blend_factor))

    return out


def _apply_interference(img: np.ndarray, *, itype: str, params: dict[str, Any], seed: int) -> np.ndarray:
    itype = itype.strip().lower()
    if itype not in SUPPORTED_TYPES:
        raise ValueError(f"unknown type: {itype} (supported: {sorted(SUPPORTED_TYPES)})")

    dtype = img.dtype
    base, alpha = _split_alpha(img)
    h, w, _c = base.shape
    rng = np.random.RandomState(int(seed))

    out_base = base
    if itype == "awgn":
        variance = float(params.get("noiseVariance", params.get("variance", 25.0)))
        if variance < 0:
            raise ValueError("noiseVariance must be >= 0")
        if variance > 0:
            sigma = math.sqrt(variance)
            noise = rng.normal(0.0, sigma, (h, w)).astype(np.float32)
            out_base = _apply_scalar_field(base, noise)
    elif itype == "noise_jamming":
        out_base = _apply_scalar_field(
            base,
            _noise_jamming_field(
                h,
                w,
                js_ratio_db=float(params.get("jsRatio", 10.0)),
                stripe_frequency=float(params.get("stripeFreq", 0.01)),
                stripe_amplitude=float(params.get("stripeAmplitude", 50.0)),
            ),
        )
    elif itype == "corner_reflector":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        tmpl = Path(str(params.get("templatePath", "pointTarget.png")))
        if not tmpl.is_absolute():
            tmpl = Path("tools") / tmpl
        if not tmpl.is_file():
            raise FileNotFoundError(f"template not found: {tmpl}")
        out_base = _apply_corner_reflector(
            base,
            template_path=tmpl,
            locations=locs,
            intensity=float(params.get("intensity", 1.0)),
            blend_mode=str(params.get("blendMode", "add")).strip().lower(),
        )
    elif itype == "chaff":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        cloud_size = params.get("cloudSize", [0.2, 0.3])
        if not isinstance(cloud_size, (list, tuple)) or len(cloud_size) != 2:
            raise ValueError("cloudSize must be [height_frac, width_frac]")
        out_base = _apply_chaff(
            base,
            rng=rng,
            locations=locs,
            cloud_size=(float(cloud_size[0]), float(cloud_size[1])),
            density_sigma_factor=float(params.get("densitySigmaFactor", 0.25)),
            noise_sigma=float(params.get("noiseSigma", 300.0)),
        )
    elif itype == "smart_noise_jamming":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        noise_size = params.get("noiseSize", [0.2, 0.2])
        if not isinstance(noise_size, (list, tuple)) or len(noise_size) != 2:
            raise ValueError("noiseSize must be [height_frac, width_frac]")
        out_base = _apply_smart_noise(
            base,
            rng=rng,
            locations=locs,
            noise_size=(float(noise_size[0]), float(noise_size[1])),
            noise_sigma=float(params.get("noiseSigma", 200.0)),
        )
    elif itype == "noise_am_jamming":
        out_base = _apply_noise_am_lines(
            base,
            rng=rng,
            line_frequency=float(params.get("lineFrequency", 0.05)),
            base_intensity=float(params.get("baseIntensity", 150.0)),
            noise_sigma=float(params.get("noiseSigma", 200.0)),
            line_width=int(params.get("lineWidth", 10)),
            direction=str(params.get("direction", "vertical")).strip().lower(),
            blend_factor=float(params.get("blendFactor", 0.3)),
        )

    out_base_u8 = _clip_to_dtype(out_base, dtype)
    alpha_u8 = _clip_to_dtype(alpha, dtype) if alpha is not None else None
    out = _merge_alpha(out_base_u8, alpha_u8)

    if img.ndim == 2:
        return out[..., 0]
    if img.ndim == 3 and img.shape[2] == 1 and out.ndim == 3 and out.shape[2] == 1:
        return out
    # cv2.imwrite prefers (H,W) for gray
    if out.ndim == 3 and out.shape[2] == 1:
        return out[..., 0]
    return out


@dataclass(frozen=True)
class Job:
    src: str
    dst: str
    seed: int


def _process_one(job: Job, *, itype: str, params: dict[str, Any], overwrite: bool) -> tuple[str, str]:
    try:
        src = Path(job.src)
        dst = Path(job.dst)
        if dst.is_file() and not overwrite:
            return "skip", str(dst)

        img = _read_image(src)
        out = _apply_interference(img, itype=itype, params=params, seed=int(job.seed))
        _write_image(dst, out)
        return "ok", str(dst)
    except Exception as e:
        return "fail", f"{job.src} -> {job.dst} ({e})"


def _check_diff(samples: list[Path], *, clean_root: Path, corrupt_root: Path) -> None:
    checked = 0
    identical = 0
    for src in samples:
        rel = src.relative_to(clean_root)
        dst = corrupt_root / rel
        if not dst.is_file():
            continue
        a = _read_image(src)
        b = _read_image(dst)
        if a.shape != b.shape:
            continue
        checked += 1
        if np.array_equal(a, b):
            identical += 1
    if checked == 0:
        raise SystemExit("diff check: no sample pairs checked (unexpected)")
    if identical == checked:
        raise SystemExit(f"diff check failed: all {checked} samples are byte-identical (no interference applied?)")
    _log(f"diff check: checked={checked} identical={identical}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/RSAR", help="RSAR root containing train/val/test")
    parser.add_argument("--corrupt", required=True, help="e.g. interf_jamA (writes images-interf_jamA/)")
    parser.add_argument("--type", required=True, choices=sorted(SUPPORTED_TYPES))
    parser.add_argument(
        "--params-json",
        default="",
        help="JSON string, or @path/to/params.json (keys depend on type)",
    )
    parser.add_argument("--splits", default="train,val,test", help="comma-separated splits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-replace-symlink",
        action="store_true",
        help="If images-<corrupt> is currently a symlink (placeholder), replace it with a real directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, limit number of images per split (useful for smoke).",
    )
    parser.add_argument(
        "--diff-samples",
        type=int,
        default=64,
        help="Sample N images and ensure clean != corrupt (0 disables).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    params = _parse_params(str(args.params_json))
    itype = str(args.type).strip().lower()

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for split in splits:
        split_dir = data_root / split
        clean_dir = split_dir / "images"
        if not clean_dir.is_dir():
            raise SystemExit(f"missing: {clean_dir}")

        out_dir = split_dir / f"images-{args.corrupt}"
        _ensure_real_dir(out_dir, force_replace_symlink=bool(args.force_replace_symlink))

        src_files = sorted(_iter_images(clean_dir))
        if args.max_images and int(args.max_images) > 0:
            src_files = src_files[: int(args.max_images)]

        jobs: list[Job] = []
        params_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        for src in src_files:
            rel = src.relative_to(clean_dir)
            dst = out_dir / rel
            seed = int(args.seed) ^ _stable_seed(str(src), itype, params_key, str(args.seed))
            jobs.append(Job(src=str(src), dst=str(dst), seed=seed))

        if not jobs:
            _log(f"{split}: nothing to do")
            continue

        _log(f"{split}: generate {args.corrupt} type={itype} jobs={len(jobs)} workers={args.workers}")
        ok = 0
        skipped = 0
        failed = 0
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            worker = partial(_process_one, itype=itype, params=params, overwrite=bool(args.overwrite))
            processed = 0
            total = len(jobs)
            for status, _path in ex.map(worker, jobs, chunksize=64):
                processed += 1
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    _log(f"ERROR: {_path}")
                if processed % 5000 == 0 or processed == total:
                    _log(f"{split}: progress {processed}/{total} (ok={ok} skip={skipped} fail={failed})")

        _log(f"{split}: done ok={ok} skipped={skipped} failed={failed}")
        total_ok += ok
        total_skip += skipped
        total_fail += failed

        if int(args.diff_samples) > 0:
            n = min(int(args.diff_samples), len(src_files))
            _check_diff(src_files[:n], clean_root=clean_dir, corrupt_root=out_dir)

    _log(f"all done: ok={total_ok} skipped={total_skip} failed={total_fail}")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
