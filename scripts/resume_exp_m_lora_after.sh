#!/usr/bin/env bash
set -Eeuo pipefail
echo "[deprecated] scripts/resume_exp_m_lora_after.sh is archived; forwarding to scripts/archive/resume_exp_m_lora_after.sh" >&2
bash scripts/archive/resume_exp_m_lora_after.sh "$@"
