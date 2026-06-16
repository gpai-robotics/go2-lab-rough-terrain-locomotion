# Validation

A candidate is considered deployment-ready only if it passes:

- task registration
- IsaacLab visual rollout
- exported model parity
- MuJoCo FSM sim2sim
- read-only hardware observation probe
- stance-only hardware bring-up
- short controlled hardware walking test

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
