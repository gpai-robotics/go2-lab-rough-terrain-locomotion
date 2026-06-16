#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
This public repo documents the MuJoCo/Unitree FSM deployment lane used by the
successful candidate. The runtime itself depends on Unitree MJLAB deployment
assets and should be installed separately.

Expected flow:
  1. Export the trained policy bundle.
  2. Copy the bundle into the Unitree MJLAB deploy policy directory.
  3. Validate in MuJoCo with the same C++ FSM used for hardware.
  4. Run hardware only after MuJoCo and read-only DDS checks pass.

See docs/DEPLOYMENT.md for concrete commands.
EOF
