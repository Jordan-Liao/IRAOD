#!/usr/bin/env bash
# Shared helpers for IRAOD experiment launchers.

IRAOD_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRAOD_ROOT="$(cd "${IRAOD_COMMON_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CGA_MODE="${CGA_MODE:-sfodrs}"

if [[ -n "${DATA_ROOT:-}" && -z "${RSAR_DATA_ROOT:-}" ]]; then
  export RSAR_DATA_ROOT="${DATA_ROOT}"
fi

IRAOD_CORRUPTIONS=(
  chaff
  gaussian_white_noise
  point_target
  noise_suppression
  am_noise_horizontal
  smart_suppression
  am_noise_vertical
)

iraod_cd_root() {
  cd "${IRAOD_ROOT}"
}

iraod_is_dry_run() {
  [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]
}

iraod_quote() {
  local arg
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
}

iraod_log() {
  local tag="$1"
  shift
  printf '[%s] %s %s\n' "${tag}" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

iraod_log_file() {
  local log_file="$1"
  local tag="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  iraod_log "${tag}" "$*" | tee -a "${log_file}"
}

iraod_run() {
  if iraod_is_dry_run; then
    printf '[DRY_RUN] '
    iraod_quote "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

iraod_env_run() {
  local env_args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    env_args+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "[iraod] missing -- before command in iraod_env_run" >&2
    return 2
  fi
  shift

  if iraod_is_dry_run; then
    printf '[DRY_RUN] env '
    iraod_quote "${env_args[@]}"
    iraod_quote "$@"
    printf '\n'
    return 0
  fi
  env "${env_args[@]}" "$@"
}

iraod_env_run_logged() {
  local log_file="$1"
  shift
  local env_args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    env_args+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "[iraod] missing -- before command in iraod_env_run_logged" >&2
    return 2
  fi
  shift

  mkdir -p "$(dirname "${log_file}")"
  if iraod_is_dry_run; then
    printf '[DRY_RUN] env '
    iraod_quote "${env_args[@]}"
    iraod_quote "$@"
    printf '> %q 2>&1\n' "${log_file}"
    return 0
  fi
  env "${env_args[@]}" "$@" > "${log_file}" 2>&1
}

iraod_ddp_command() {
  local script="$1"
  shift
  local ngpus="${NGPUS:-1}"
  local port="${MASTER_PORT:-29501}"

  if (( ngpus > 1 )); then
    IRAOD_DDP_CMD=(
      "${PYTHON_BIN}" -m torch.distributed.launch
      "--nproc_per_node=${ngpus}"
      "--master_port=${port}"
      --use_env
      "${script}"
      "$@"
      --launcher pytorch
    )
  else
    IRAOD_DDP_CMD=("${PYTHON_BIN}" -u "${script}" "$@")
  fi
}

iraod_ddp_run() {
  iraod_ddp_command "$@"
  iraod_run "${IRAOD_DDP_CMD[@]}"
}

iraod_ddp_env_run() {
  local env_args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    env_args+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "[iraod] missing -- before script in iraod_ddp_env_run" >&2
    return 2
  fi
  shift
  iraod_ddp_command "$@"
  iraod_env_run "${env_args[@]}" -- "${IRAOD_DDP_CMD[@]}"
}

iraod_ddp_env_run_logged() {
  local log_file="$1"
  shift
  local env_args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    env_args+=("$1")
    shift
  done
  if [[ $# -eq 0 ]]; then
    echo "[iraod] missing -- before script in iraod_ddp_env_run_logged" >&2
    return 2
  fi
  shift
  iraod_ddp_command "$@"
  iraod_env_run_logged "${log_file}" "${env_args[@]}" -- "${IRAOD_DDP_CMD[@]}"
}

iraod_source_ckpt_is_null() {
  local ckpt="${1:-}"
  [[ -z "${ckpt}" || "${ckpt}" == "none" || "${ckpt}" == "null" || "${ckpt}" == "-" ]]
}

iraod_append_if_set() {
  local -n target_array="$1"
  local option="$2"
  local value="${3:-}"
  if [[ -n "${value}" ]]; then
    target_array+=("${option}" "${value}")
  fi
}
