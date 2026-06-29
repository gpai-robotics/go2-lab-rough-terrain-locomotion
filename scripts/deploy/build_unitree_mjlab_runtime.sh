#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MJLAB_ROOT="${REPO_ROOT}/reference_repos/unitree_rl_mjlab"
SDK_SRC="${UNITREE_SDK2_SRC:-${REPO_ROOT}/reference_repos/unitree_sdk2}"
SDK_INSTALL="${UNITREE_SDK2_INSTALL:-${SDK_SRC}/install}"
GO2_BUILD="${MJLAB_ROOT}/deploy/robots/go2/build"
SIM_BUILD="${MJLAB_ROOT}/simulate/build"
CTRL_BIN="${GO2_BUILD}/go2_ctrl"
SIM_BIN="${SIM_BUILD}/unitree_mujoco"
JOBS="${JOBS:-$(nproc)}"

usage() {
  cat <<EOF
Usage:
  $0 [all|sdk|controller|sim|verify]

Build the Unitree RL MJLab C++ deployment binaries used by:
  scripts/deploy/run_unitree_mjlab_sim_deploy.sh

Targets:
  all         Install unitree_sdk2 locally, then build go2_ctrl and unitree_mujoco (default)
  sdk         Clone/build/install unitree_sdk2 into a repo-local prefix
  controller  Build go2_ctrl only (requires sdk)
  sim         Build unitree_mujoco only (requires sdk)
  verify      Check that expected binaries exist and resolve shared libraries

Environment overrides:
  UNITREE_SDK2_SRC      unitree_sdk2 source tree (default: reference_repos/unitree_sdk2)
  UNITREE_SDK2_INSTALL  install prefix (default: \${UNITREE_SDK2_SRC}/install)
  JOBS                  parallel build jobs (default: nproc)

Expected outputs:
  ${CTRL_BIN}
  ${SIM_BIN}

Typical recovery flow after a clean checkout or broken deploy path:
  1. git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git reference_repos/unitree_rl_mjlab
  2. cd reference_repos/unitree_rl_mjlab && git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch && cd ../..
  3. bash scripts/deploy/build_unitree_mjlab_runtime.sh all
  4. bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
  5. bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate

See docs/UNITREE_MJLAB_RUNTIME_BUILD.md for troubleshooting.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_apt_packages() {
  local missing=()
  for pkg in cmake g++ libyaml-cpp-dev libboost-all-dev libeigen3-dev libfmt-dev; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done
  if ((${#missing[@]} > 0)); then
    echo "Missing apt packages: ${missing[*]}" >&2
    echo "Install with:" >&2
    echo "  sudo apt install -y cmake g++ build-essential libyaml-cpp-dev libboost-all-dev libeigen3-dev libfmt-dev" >&2
    exit 1
  fi
}

require_mjlab_runtime() {
  if [[ ! -d "${MJLAB_ROOT}" ]]; then
    echo "Missing external runtime: ${MJLAB_ROOT}" >&2
    echo "Clone it first:" >&2
    echo "  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git reference_repos/unitree_rl_mjlab" >&2
    echo "Then apply this repo's patch:" >&2
    echo "  cd reference_repos/unitree_rl_mjlab && git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch && cd ../.." >&2
    exit 1
  fi
}

sdk_configured() {
  [[ -f "${SDK_INSTALL}/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake" ]]
}

build_sdk() {
  require_cmd git
  require_cmd cmake
  require_cmd make
  require_apt_packages

  if [[ ! -d "${SDK_SRC}/.git" ]]; then
    echo "Cloning unitree_sdk2 into ${SDK_SRC}"
    git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2.git "${SDK_SRC}"
  fi

  mkdir -p "${SDK_SRC}/build"
  cmake -S "${SDK_SRC}" -B "${SDK_SRC}/build" -DCMAKE_INSTALL_PREFIX="${SDK_INSTALL}"
  cmake --build "${SDK_SRC}/build" -j"${JOBS}"
  cmake --install "${SDK_SRC}/build"

  if ! sdk_configured; then
    echo "unitree_sdk2 install failed: missing ${SDK_INSTALL}/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake" >&2
    exit 1
  fi
  echo "unitree_sdk2 installed to: ${SDK_INSTALL}"
}

ensure_sdk() {
  if sdk_configured; then
    echo "Using unitree_sdk2 install: ${SDK_INSTALL}"
    return
  fi
  if [[ -f /opt/unitree_robotics/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake ]]; then
    SDK_INSTALL="/opt/unitree_robotics"
    echo "Using system unitree_sdk2 install: ${SDK_INSTALL}"
    return
  fi
  echo "No unitree_sdk2 install found; building repo-local SDK"
  build_sdk
}

ddscxx_include_dir() {
  if [[ -d "${SDK_SRC}/thirdparty/include/ddscxx" ]]; then
    echo "${SDK_SRC}/thirdparty/include/ddscxx"
    return
  fi
  if [[ -d "${SDK_INSTALL}/include/ddscxx" ]]; then
    echo "${SDK_INSTALL}/include/ddscxx"
    return
  fi
  echo "Could not find ddscxx headers under ${SDK_SRC} or ${SDK_INSTALL}" >&2
  exit 1
}

build_controller() {
  require_mjlab_runtime
  ensure_sdk
  require_cmd cmake
  require_cmd make
  require_apt_packages

  local ddscxx_include
  ddscxx_include="$(ddscxx_include_dir)"

  mkdir -p "${GO2_BUILD}"
  cmake -S "${MJLAB_ROOT}/deploy/robots/go2" -B "${GO2_BUILD}" \
    -DCMAKE_CXX_FLAGS="-I${SDK_INSTALL}/include -I${ddscxx_include} -I/usr/include/eigen3" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${SDK_INSTALL}/lib -Wl,-rpath,${SDK_INSTALL}/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${SDK_INSTALL}/lib -Wl,-rpath,${SDK_INSTALL}/lib"
  cmake --build "${GO2_BUILD}" -j"${JOBS}"
  echo "Built controller: ${CTRL_BIN}"
}

build_sim() {
  require_mjlab_runtime
  ensure_sdk
  require_cmd cmake
  require_cmd make
  require_apt_packages

  mkdir -p "${SIM_BUILD}"
  cmake -S "${MJLAB_ROOT}/simulate" -B "${SIM_BUILD}" \
    -DCMAKE_PREFIX_PATH="${SDK_INSTALL}"
  cmake --build "${SIM_BUILD}" -j"${JOBS}"
  echo "Built simulator: ${SIM_BIN}"
}

verify_binaries() {
  local failed=0
  for bin in "${CTRL_BIN}" "${SIM_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
      echo "[FAIL] missing executable: ${bin}" >&2
      failed=1
      continue
    fi
    echo "[OK] executable: ${bin}"
    if ldd "${bin}" | rg -q 'not found'; then
      echo "[FAIL] unresolved shared libraries for ${bin}:" >&2
      ldd "${bin}" | rg 'not found' >&2 || true
      failed=1
    else
      echo "[OK] shared libraries resolve for ${bin}"
    fi
  done
  return "${failed}"
}

main() {
  local target="${1:-all}"
  case "${target}" in
    all)
      build_sdk
      build_controller
      build_sim
      verify_binaries
      ;;
    sdk)
      build_sdk
      ;;
    controller)
      build_controller
      verify_binaries
      ;;
    sim)
      build_sim
      verify_binaries
      ;;
    verify)
      verify_binaries
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown target: ${target}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
