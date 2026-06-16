# Deployment

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
3. MuJoCo sim2sim using the Unitree FSM runtime.
4. Read-only DDS probe.
5. Stance-only hardware bring-up.
6. Slow forward hardware command.
7. Yaw/lateral commands.
8. Rough-terrain hardware tests.

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
