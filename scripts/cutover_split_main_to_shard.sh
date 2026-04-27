#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] scripts/cutover_split_main_to_shard.sh is archived; forwarding to scripts/archive/cutover_split_main_to_shard.sh" >&2
bash scripts/archive/cutover_split_main_to_shard.sh "$@"
