# Validation

A candidate is considered deployment-ready only if it passes:

- task registration
- deploy-side Isaac rollout
- exported model parity
- MuJoCo sim2sim
- read-only hardware observation probe
- stance-only hardware bring-up
- short controlled hardware walking test

Recommended tools for those gates:

- `scripts/deploy/validate_bundle.py`
- `scripts/deploy/validate_policy_inference_parity.py`
- `scripts/deploy/play_deploy_policy.py`
- `scripts/deploy/run_deployment_validation_gate.py`
- `scripts/deploy/run_sim2sim.py`
- `scripts/deploy/probe_go2_readonly.py`
- `scripts/deploy/monitor_go2_realtime.py`
- `scripts/deploy/run_go2_hardware.py`

## Cross-Sim Checks

Record simulator parameters for every result:

- control timestep
- physics timestep
- mass
- joint damping/friction
- actuator gains
- velocity limits
- terrain type and seed
- command profile

Mismatches are not automatically fatal, but they must be visible before real
deployment.

## Hardware Logs

For hardware tests, capture low-level telemetry:

- LowState stream
- LowCmd stream
- command stream
- joint position/velocity
- torque estimate
- contact/fall events where available

Analyze mirror pairs and leg-specific weakness before increasing command range.
