#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_EXE="${MUJOCO_PYTHON:-python3}"
BUNDLE_DIR="${TERRAIN_BUNDLE:-${REPO_ROOT}/artifacts/exported/go2_terrain_locomotion_steps_v1_candidate}"
OUTPUT_DIR="${REPO_ROOT}/artifacts/mujoco_eval"

if [[ ! -d "${BUNDLE_DIR}" ]]; then
  echo "Missing bundle dir: ${BUNDLE_DIR}" >&2
  exit 1
fi

echo "[VALIDATION] Materializing Unitree MuJoCo terrain recipes."
"${PYTHON_EXE}" "${REPO_ROOT}/scripts/deploy/materialize_unitree_mujoco_terrain_recipes.py"

run_suite() {
  local suite="$1"
  local max_steps="$2"
  local rollouts="$3"
  echo "[VALIDATION] Running ${suite}: max_steps=${max_steps}, rollouts=${rollouts}"
  "${PYTHON_EXE}" "${REPO_ROOT}/scripts/deploy/run_mujoco_ood_suite.py" \
    --bundle-dir "${BUNDLE_DIR}" \
    --suite "${suite}" \
    --num-rollouts "${rollouts}" \
    --max-steps "${max_steps}" \
    --trace-steps 0 \
    --reset-preset light \
    --python-exe "${PYTHON_EXE}" \
    --output-dir "${OUTPUT_DIR}" \
    --continue-on-error
}

run_suite mujoco_nominal_v1 1500 5
run_suite mujoco_disturb_v2_moderate 1500 5
run_suite mujoco_rough_v1 1800 5
run_suite mujoco_rough_v2_hard 1800 5

cat <<EOF

[VALIDATION] MuJoCo suites complete.
Summaries:
  ${OUTPUT_DIR}/go2_terrain_locomotion_steps_v1_candidate/mujoco_nominal_v1/suite_summary.csv
  ${OUTPUT_DIR}/go2_terrain_locomotion_steps_v1_candidate/mujoco_disturb_v2_moderate/suite_summary.csv
  ${OUTPUT_DIR}/go2_terrain_locomotion_steps_v1_candidate/mujoco_rough_v1/suite_summary.csv
  ${OUTPUT_DIR}/go2_terrain_locomotion_steps_v1_candidate/mujoco_rough_v2_hard/suite_summary.csv
EOF
