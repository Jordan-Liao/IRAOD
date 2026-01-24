import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd, *, timeout_s=120):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 1, f"[env_snapshot] failed to run {cmd!r}: {exc}\n"


def _safe_import_versions():
    versions = {}
    for name in ("torch", "mmcv", "mmdet", "mmrotate"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"import-failed: {exc}"
    try:
        import torch

        versions["torch.version.cuda"] = getattr(torch.version, "cuda", None)
        versions["torch.cuda.is_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["torch.cuda.device_count"] = torch.cuda.device_count()
            versions["torch.cuda.current_device"] = torch.cuda.current_device()
            versions["torch.cuda.get_device_name"] = torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
    except Exception as exc:
        versions["torch.cuda"] = f"probe-failed: {exc}"
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Output directory (e.g., docs)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_lock_path = out_dir / "env_lock.txt"
    system_info_path = out_dir / "system_info.md"

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # env_lock.txt
    rc, freeze = _run([sys.executable, "-m", "pip", "freeze"], timeout_s=300)
    env_lock_path.write_text(freeze, encoding="utf-8")
    if rc != 0:
        print("[env_snapshot] WARNING: `pip freeze` returned non-zero", file=sys.stderr)

    # system_info.md
    versions = _safe_import_versions()
    git_rc, git_hash = _run(["git", "rev-parse", "HEAD"])
    git_hash = git_hash.strip() if git_rc == 0 else f"unavailable ({git_hash.strip()})"
    git_status_rc, git_status = _run(["git", "status", "--porcelain"])
    dirty = "dirty" if (git_status_rc == 0 and git_status.strip()) else "clean"

    _, nvsmi = _run(["nvidia-smi"], timeout_s=20)
    _, nvsmi_l = _run(["nvidia-smi", "-L"], timeout_s=20)

    lines = []
    lines.append("# System Info\n")
    lines.append(f"- Timestamp: {now}\n")
    lines.append(f"- Python: `{sys.version.splitlines()[0]}`\n")
    lines.append(f"- Executable: `{sys.executable}`\n")
    lines.append(f"- CONDA_DEFAULT_ENV: `{os.environ.get('CONDA_DEFAULT_ENV', '')}`\n")
    lines.append(f"- Git: `{git_hash}` ({dirty})\n")
    lines.append("\n## Key Versions\n")
    for k in sorted(versions.keys()):
        lines.append(f"- {k}: `{versions[k]}`\n")
    lines.append("\n## nvidia-smi\n")
    lines.append("```text\n")
    lines.append(nvsmi.strip() + "\n")
    lines.append("```\n")
    lines.append("\n## nvidia-smi -L\n")
    lines.append("```text\n")
    lines.append(nvsmi_l.strip() + "\n")
    lines.append("```\n")

    system_info_path.write_text("".join(lines), encoding="utf-8")

    print(f"[env_snapshot] wrote {env_lock_path}")
    print(f"[env_snapshot] wrote {system_info_path}")


if __name__ == "__main__":
    raise SystemExit(main())

