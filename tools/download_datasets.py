import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


RSAR_GDRIVE_ID = "1v-HXUSmwBQCtrq0MlTOkCaBQ_vbz5_qs"


def _log(msg: str) -> None:
    print(f"[download_datasets] {msg}")


def _err(msg: str) -> None:
    print(f"[download_datasets] ERROR: {msg}", file=sys.stderr)


def _proxy_clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(k, None)
    return env


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    _log(f"run: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return int(proc.returncode)


def _looks_like_archive_path(remote_path: str) -> bool:
    p = remote_path.strip().lower()
    return p.endswith((".tar", ".tar.gz", ".tgz", ".zip"))


def _looks_like_rsar_root(path: Path) -> bool:
    # Minimal layout check.
    for split in ("train", "val", "test"):
        if not (path / split / "annfiles").is_dir():
            return False
        if not (path / split / "images").is_dir():
            return False
    return True


def _looks_like_dior_root(path: Path) -> bool:
    ann_dir = path / "Annotations"
    jpeg_dir = path / "JPEGImages"
    sets_dir = path / "ImageSets"
    if not ann_dir.is_dir() or not jpeg_dir.is_dir() or not sets_dir.is_dir():
        return False
    for name in ("train.txt", "val.txt", "test.txt"):
        if not (sets_dir / name).is_file():
            return False
    if not any(jpeg_dir.glob("*.jpg")) and not any(jpeg_dir.glob("*.JPG")):
        return False
    obb_dir = ann_dir / "Oriented Bounding Boxes"
    if obb_dir.is_dir():
        if not any(obb_dir.glob("*.xml")) and not any(obb_dir.glob("*.XML")):
            return False
    else:
        if not any(ann_dir.glob("*.xml")) and not any(ann_dir.glob("*.XML")):
            return False
    return True


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        _log(f"extract zip -> {dest_dir}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        return
    if tarfile.is_tarfile(archive_path):
        _log(f"extract tar -> {dest_dir}")
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)
        return
    raise RuntimeError(f"unknown archive format: {archive_path}")


def _maybe_flatten_single_dir(dest_dir: Path) -> None:
    # Some archives contain a single top-level folder; flatten it.
    children = [p for p in dest_dir.iterdir() if p.name not in (".DS_Store",)]
    if len(children) != 1:
        return
    if not children[0].is_dir():
        return
    inner = children[0]
    _log(f"flatten archive root: {inner} -> {dest_dir}")
    for p in inner.iterdir():
        shutil.move(str(p), str(dest_dir / p.name))
    try:
        inner.rmdir()
    except OSError:
        pass


def download_rsar(args) -> int:
    dest_dir = Path(args.dest).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _looks_like_rsar_root(dest_dir):
        _log(f"RSAR already prepared at: {dest_dir}")
        return 0

    # 1) bypy syncdown (optional)
    if args.source == "bypy":
        if shutil.which("bypy") is None:
            _err("`bypy` not found in PATH. Install via: `pip install bypy` (then run `bypy info` to login)")
            return 2
        if not args.bypy_remote:
            _err("--bypy-remote is required when --source bypy")
            return 2
        remote = str(args.bypy_remote).strip()
        if _looks_like_archive_path(remote):
            archive_path = cache_dir / Path(remote).name
            if not archive_path.exists() or archive_path.stat().st_size == 0:
                rc = _run(["bypy", "downfile", remote, str(archive_path)], env=_proxy_clean_env())
                if rc != 0:
                    return rc

            tmp_extract = cache_dir / "RSAR_extract_tmp"
            shutil.rmtree(tmp_extract, ignore_errors=True)
            tmp_extract.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive_path, tmp_extract)
            _maybe_flatten_single_dir(tmp_extract)

            if not _looks_like_rsar_root(tmp_extract):
                _err(f"downloaded from bypy but extracted content is not RSAR layout under: {tmp_extract}")
                return 3

            shutil.rmtree(dest_dir, ignore_errors=True)
            shutil.copytree(tmp_extract, dest_dir)
            _log(f"RSAR prepared at: {dest_dir}")
            return 0

        tmp_dir = cache_dir / "RSAR_bypy_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rc = _run(["bypy", "downdir", remote, str(tmp_dir)], env=_proxy_clean_env())
        if rc != 0:
            return rc
        # Try to locate extracted root inside tmp_dir
        candidates = [tmp_dir] + [p for p in tmp_dir.glob("**/") if p.is_dir()]
        for c in candidates:
            if _looks_like_rsar_root(c):
                shutil.rmtree(dest_dir, ignore_errors=True)
                shutil.copytree(c, dest_dir)
                _log(f"RSAR prepared at: {dest_dir}")
                return 0
        _err(f"downloaded from bypy but RSAR layout not found under: {tmp_dir}")
        return 3

    # 2) Google Drive download (default)
    try:
        import gdown  # type: ignore
    except Exception:
        _err("missing dependency: gdown. Install via: `pip install gdown`")
        return 2

    archive_path = cache_dir / args.archive_name
    url = f"https://drive.google.com/uc?id={args.gdrive_id}"
    _log(f"downloading RSAR from Google Drive -> {archive_path}")
    gdown.download(url, str(archive_path), quiet=False, resume=True)

    tmp_extract = cache_dir / "RSAR_extract_tmp"
    shutil.rmtree(tmp_extract, ignore_errors=True)
    tmp_extract.mkdir(parents=True, exist_ok=True)
    _extract_archive(archive_path, tmp_extract)
    _maybe_flatten_single_dir(tmp_extract)

    if not _looks_like_rsar_root(tmp_extract):
        _err(f"extracted content is not RSAR layout under: {tmp_extract}")
        return 3

    shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.copytree(tmp_extract, dest_dir)
    _log(f"RSAR prepared at: {dest_dir}")
    return 0


def download_dior(args) -> int:
    dest_dir = Path(args.dest).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _looks_like_dior_root(dest_dir):
        _log(f"DIOR already prepared at: {dest_dir}")
        return 0

    if args.source != "bypy":
        _err("DIOR downloader currently supports only: --source bypy")
        return 2
    if shutil.which("bypy") is None:
        _err("`bypy` not found in PATH. Install via: `pip install bypy` (then run `bypy info` to login)")
        return 2
    if not args.bypy_remote:
        _err("--bypy-remote is required when --source bypy")
        return 2

    tmp_dir = cache_dir / "DIOR_bypy_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(["bypy", "downdir", args.bypy_remote, str(tmp_dir)], env=_proxy_clean_env())
    if rc != 0:
        return rc

    candidates = [tmp_dir] + [p for p in tmp_dir.glob("**/") if p.is_dir()]
    for c in candidates:
        if _looks_like_dior_root(c):
            shutil.rmtree(dest_dir, ignore_errors=True)
            shutil.copytree(c, dest_dir)
            _log(f"DIOR prepared at: {dest_dir}")
            return 0

    _err(f"downloaded from bypy but DIOR layout not found under: {tmp_dir}")
    return 3


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_dior = sub.add_parser("dior", help="Download/prepare DIOR dataset")
    p_dior.add_argument("--dest", default="dataset/DIOR", help="Output dataset dir")
    p_dior.add_argument("--cache-dir", default="dataset/_downloads", help="Cache dir for downloads")
    p_dior.add_argument(
        "--source",
        choices=("bypy",),
        default="bypy",
        help="Download source (requires the dataset already stored in /apps/bypy)",
    )
    p_dior.add_argument("--bypy-remote", default="", help="Remote dir under /apps/bypy (e.g., DIOR)")
    p_dior.set_defaults(func=download_dior)

    p_rsar = sub.add_parser("rsar", help="Download/prepare RSAR dataset")
    p_rsar.add_argument("--dest", default="dataset/RSAR", help="Output dataset dir")
    p_rsar.add_argument("--cache-dir", default="dataset/_downloads", help="Cache dir for archives")
    p_rsar.add_argument(
        "--source",
        choices=("gdrive", "bypy"),
        default="gdrive",
        help="Download source (bypy requires the dataset already stored in /apps/bypy)",
    )
    p_rsar.add_argument("--gdrive-id", default=RSAR_GDRIVE_ID, help="Google Drive file id")
    p_rsar.add_argument("--archive-name", default="RSAR_download", help="Cached archive file name")
    p_rsar.add_argument(
        "--bypy-remote",
        default="",
        help="Remote dir or archive file under /apps/bypy (e.g., RSAR, RSAR.tar)",
    )
    p_rsar.set_defaults(func=download_rsar)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
