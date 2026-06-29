#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GO2_MUJOCO_MODEL="${GO2_MUJOCO_MODEL:-}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  bash scripts/run_mujoco_fsm.sh

This helper prints the standalone MuJoCo and hardware bring-up sequence.
EOF
  exit 0
fi

echo "[INFO] Public training repo: ${REPO_ROOT}"
cat <<EOF

Expected flow:
  1. Train the rough AsymPPO policy from this repo.
  2. Export the trained checkpoint into a local deployment bundle.
  3. Run bundle validation and parity checks.
  4. Run the local MuJoCo validation gate.
  5. Run the local MuJoCo bridge.
  6. Optionally stage the Unitree RL MJLAB C++ FSM runtime.
  7. Run a read-only DDS probe before any hardware controller.
  8. Dry-run the hardware contract.
  9. Bring up hardware over Ethernet first.

Copy-paste starting points:

  cd "${REPO_ROOT}"
  bash scripts/isaaclab_user.sh -p scripts/deploy/export_policy.py \\
    --policy-name go2_blind_rough_asymppo_mjlab_v1_candidate \\
    --checkpoint /path/to/model_1999.pt \\
    --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \\
    --phase blind-rough-mjlab-asymppo-v1 \\
    --policy-kind blind_history_policy \\
    --observation-groups policy,policy_history \\
    --format torchscript \\
    --format onnx

  cd "${REPO_ROOT}"
  python scripts/deploy/run_deployment_validation_gate.py \\
    --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \\
    --expected-policy-obs-dim 45 \\
    --expected-history-length 100 \\
    --model-path "\${GO2_MUJOCO_MODEL}"

  cd "${REPO_ROOT}"
  python scripts/deploy/run_sim2sim.py \\
    --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \\
    --model-path "\${GO2_MUJOCO_MODEL}" \\
    --execute-runtime

  # Optional closest-to-deployed C++ FSM runtime:
  cd "${REPO_ROOT}"
  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \\
    reference_repos/unitree_rl_mjlab
  cd reference_repos/unitree_rl_mjlab
  git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
  cd "${REPO_ROOT}"
  bash scripts/deploy/build_unitree_mjlab_runtime.sh all
  bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
  bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate

  cd "${REPO_ROOT}"
  python scripts/deploy/probe_go2_readonly.py \\
    --net-if "\${GO2_ETH_IF:-eth0}" \\
    --duration-s 5 \\
    --subscribe-sport \\
    --unitree-sdk-root /path/to/unitree_sdk2py

  cd "${REPO_ROOT}"
  python scripts/deploy/run_go2_hardware.py \\
    --bundle-dir artifacts/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \\
    --net-if "\${GO2_ETH_IF:-eth0}" \\
    --dry-run

Prerequisites to set yourself:
  export GO2_MUJOCO_MODEL=/path/to/unitree_go2/scene.xml
  export GO2_ETH_IF=eth0
  export UNITREE_SDK2PY_ROOT=/path/to/unitree_sdk2py

See also:
  ${REPO_ROOT}/docs/REPRODUCTION.md
  ${REPO_ROOT}/docs/RUN_COMMANDS.md
  ${REPO_ROOT}/docs/UNITREE_MJLAB_RUNTIME_BUILD.md
EOF
