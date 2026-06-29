# Deployment

## Scope

This repo defines the active deployable actor contract and the validation order
used by the reported Go2 result. It now includes local scripts for:

- bundle export,
- structural validation,
- source-vs-export parity,
- deploy-side Isaac rehearsal,
- MuJoCo sim2sim,
- read-only DDS probe and realtime monitor,
- hardware bring-up against a Unitree Python SDK checkout,
- optional Unitree RL MJLAB C++ FSM sim/controller/hardware runtime staging.

Use `docs/REPRODUCTION.md` for the full path and `docs/RUN_COMMANDS.md` for
copy-paste commands.

## Contract

The exported policy must preserve this runtime contract:

```text
policy_obs_dim          45
policy_history_dim    4500
history_length         100
control_dt            0.02
action_dim              12
```

Joint order:

```text
FL_hip, FR_hip, RL_hip, RR_hip,
FL_thigh, FR_thigh, RL_thigh, RR_thigh,
FL_calf, FR_calf, RL_calf, RR_calf
```

Default pose:

```text
[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
```

Action semantics:

```text
q_target = default_pose + 0.25 * action
```

Hardware PD:

```text
kp = 25.0
kd = 0.5
```

## Validation Order

Do not go straight from IsaacLab training to real hardware.

Use this order:

1. IsaacLab rollout/play.
2. Export parity check.
3. MuJoCo sim2sim.
4. Read-only DDS probe.
5. Stance-only hardware bring-up.
6. Slow forward hardware command.
7. Yaw/lateral commands.
8. Rough-terrain hardware tests.

In this workflow, the concrete tools are:

1. local `scripts/deploy/export_policy.py`
2. local `scripts/deploy/validate_bundle.py`
3. local `scripts/deploy/validate_policy_inference_parity.py`
4. local `scripts/deploy/play_deploy_policy.py`
5. local `scripts/deploy/run_deployment_validation_gate.py`
6. local `scripts/deploy/run_sim2sim.py`
7. local `scripts/deploy/probe_go2_readonly.py`
8. local `scripts/deploy/run_go2_hardware.py`

For the exact recovered Unitree RL MJLAB C++ FSM runtime, use:

1. `patches/unitree_rl_mjlab/go2_scripted_controller.patch`
2. `scripts/deploy/build_unitree_mjlab_runtime.sh`
3. `scripts/deploy/prepare_unitree_rl_mjlab_go2_runtime.py`
4. `scripts/deploy/validate_unitree_mjlab_go2_fsm_runtime.py`
5. `scripts/deploy/run_unitree_mjlab_sim_deploy.sh`

That path is documented in
`docs/UNITREE_MJLAB_RUNTIME_BUILD.md`.

## Expected Export Bundle

The active AsymPPO lane expects an exported bundle that contains at least:

```text
bundle_manifest.json
*.torchscript.pt
*.onnx
*.export_metadata.json
*.deploy_config.json
*.deploy.yaml
export_request.json
```

The validation gate checks the recorded tensor contract against the deployable
actor interface published by this repo:

```text
policy_obs_dim = 45
policy_history_dim = 4500
history_length = 100
action_dim = 12
```

## Public External Dependencies

MuJoCo sim2sim requires:

- Python package `mujoco`
- Python package `torch`
- a Go2 MuJoCo scene XML

Hardware bring-up requires:

- a `unitree_sdk2py` checkout or install
- a reachable robot network interface
- optionally a mode-switch helper script if you want this repo's runner to
  switch into low-level mode for you

The C++ FSM runtime additionally requires:

- an external `unitree_rl_mjlab` clone
- an external or wrapper-built `unitree_sdk2` C++ install
- the repo patch applied before building
- ONNX export in the deployment bundle

## Network

Ethernet is the recommended first deployment transport.

Wi-Fi is valid only after both checks pass:

```text
peer ping to robot Wi-Fi IP
read-only DDS LowState probe over Wi-Fi
```

A laptop being connected to a hotspot is not sufficient. The Go2-side Wi-Fi
dongle must also join the same WLAN, and the WLAN must allow multicast and
client-to-client traffic.

## Safety

Use read-only probes before any LowCmd publisher. For first hardware runs,
start with:

- robot lifted or supported when appropriate
- stance-only startup
- forward-only low speed
- operator ready to stop
- Ethernet path validated first
