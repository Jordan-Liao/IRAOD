#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] scripts/run_7corrupt_cached.sh is archived; forwarding to scripts/archive/run_7corrupt_cached.sh" >&2
bash scripts/archive/run_7corrupt_cached.sh "$@"
