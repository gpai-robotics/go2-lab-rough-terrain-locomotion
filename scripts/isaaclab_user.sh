#!/usr/bin/env bash

set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/opt/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"
USER_ROOT="${ISAACSIM_USER_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/isaacsim}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] IsaacLab launcher is not executable: ${ISAACLAB_SH}" >&2
    echo "Set ISAACLAB_ROOT to the IsaacLab checkout." >&2
    exit 1
fi

CACHE_DIR="${USER_ROOT}/cache"
DATA_DIR="${USER_ROOT}/data"
LOG_DIR="${USER_ROOT}/logs"
GLOBAL_CACHE_DIR="${USER_ROOT}/global-cache"
GLOBAL_DATA_DIR="${USER_ROOT}/global-data"
MPL_DIR="${USER_ROOT}/matplotlib"
TMP_DIR="${USER_ROOT}/tmp"

mkdir -p \
    "${CACHE_DIR}" \
    "${DATA_DIR}" \
    "${LOG_DIR}" \
    "${GLOBAL_CACHE_DIR}" \
    "${GLOBAL_DATA_DIR}" \
    "${MPL_DIR}" \
    "${TMP_DIR}"

KIT_PATH_ARGS=(
    "--/app/tokens/cache=${CACHE_DIR}"
    "--/app/tokens/data=${DATA_DIR}"
    "--/app/tokens/logs=${LOG_DIR}"
    "--/app/tokens/omni_cache=${CACHE_DIR}"
    "--/app/tokens/omni_data=${DATA_DIR}"
    "--/app/tokens/omni_logs=${LOG_DIR}"
    "--/app/tokens/omni_global_cache=${GLOBAL_CACHE_DIR}"
    "--/app/tokens/omni_global_data=${GLOBAL_DATA_DIR}"
)

printf -v KIT_ARGS_STRING '%s ' "${KIT_PATH_ARGS[@]}"
KIT_ARGS_STRING="${KIT_ARGS_STRING% }"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${MPL_DIR}}"
export TMPDIR="${TMPDIR:-${TMP_DIR}}"

ML_ARCHIVE="${ISAACLAB_ROOT}/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle"
if [[ -d "${ML_ARCHIVE}" ]]; then
    ML_LIB_PATHS="$(find "${ML_ARCHIVE}" -path '*/lib' -type d 2>/dev/null | paste -sd: -)"
    if [[ -n "${ML_LIB_PATHS}" ]]; then
        export LD_LIBRARY_PATH="${ML_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
    fi
fi

LAUNCH_TERM="${TERM:-xterm}"
if [[ "${LAUNCH_TERM}" == "dumb" ]]; then
    LAUNCH_TERM="xterm"
fi

exec env TERM="${LAUNCH_TERM}" "${ISAACLAB_SH}" "$@" --kit_args "${KIT_ARGS_STRING}"
