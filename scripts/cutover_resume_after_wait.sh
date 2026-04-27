#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] scripts/cutover_resume_after_wait.sh is archived; forwarding to scripts/archive/cutover_resume_after_wait.sh" >&2
bash scripts/archive/cutover_resume_after_wait.sh "$@"
