#!/usr/bin/env bash
# Deprecated compatibility wrapper.
set -Eeuo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <corr> <cuda_visible_devices> <master_port> <work_root> <source_ckpt> [config]" >&2
  exit 2
fi

echo "[deprecated] scripts/run_single_corr_ddp.sh -> scripts/run/rsar_sfodrs_domain.sh" >&2

CORR="$1"
export CUDA_VISIBLE_DEVICES="$2"
MASTER_PORT="$3"
WORK_ROOT="$4"
SOURCE_CKPT="$5"
CONFIG="${6:-configs/current/rsar_sfodrs.py}"
NGPUS=3

CORR="${CORR}" SOURCE_CKPT="${SOURCE_CKPT}" CONFIG="${CONFIG}" \
  WORK_ROOT="${WORK_ROOT}" NGPUS="${NGPUS}" MASTER_PORT="${MASTER_PORT}" \
  RSAR_DOMAIN_MODE=full \
  bash scripts/run/rsar_sfodrs_domain.sh
