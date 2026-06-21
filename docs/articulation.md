# Articulation Setup

This document covers how to define and validate the Trakr robot as an Isaac Lab articulation.

---

## 1. ArticulationCfg

Isaac Lab uses `ArticulationCfg` to describe the robot's physical structure, initial state, and actuator models. The Trakr configuration is defined in `source/trakr_rl/trakr_rl/assets/robots/trakr.py`.

### USD Reference

```python
TRAKR_CFG = ArticulationCfg(
    prim_path="/World/Origin0/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/trakr_imu.usd",
        activate_contact_sensors=True,
    ),
)
```

> **Note:** `activate_contact_sensors=True` is required for foot-contact reward terms.

---

## 2. Physical Properties

### Rigid Body

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.1,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
),
```

### Articulation Solver

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
),
```

Values for damping, max velocities, and solver iterations are adopted from the official Trakr Legged RL repository by Addverb.

---

## 3. Initial State

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.255),
    joint_pos={
        ".*_adduction": 0.0,
        ".*F_hip":      0.0,
        ".*B_hip":      0.0,
        "L[F,B]_knee":  0.0,
        "R[F,B]_knee":  0.0,
    },
    joint_vel={".*": 0.0},
),
```

The robot spawns 0.255 m above the ground in a neutral standing posture. Joint groups are matched via regular expressions.

---

## 4. Actuator Model

```python
actuators={
    "base_legs": DCMotorCfg(
        joint_names_expr=[
            ".*_adduction",
            ".*_hip",
            ".*_knee",
        ],
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
        friction=0.0,
    ),
},
```

| Parameter | Value |
|---|---|
| Effort limit | 23.5 Nm |
| Saturation effort | 23.5 Nm |
| Velocity limit | 30.0 rad/s |
| Stiffness | 25.0 |
| Damping | 0.5 |
| Friction | 0.0 |

---

## 5. Joint Naming: Trakr vs. Unitree Go2

The Trakr model uses a different joint naming scheme from the Unitree Go2. The regular expressions in the articulation config must match Trakr's convention.

| Leg | Unitree Go2 | Trakr |
|---|---|---|
| Front-left | `FL_hip_joint`, `FL_thigh_joint`, `FL_calf_joint` | `LF_adduction`, `LF_hip`, `LF_knee` |
| Front-right | `FR_hip_joint`, `FR_thigh_joint`, `FR_calf_joint` | `RF_adduction`, `RF_hip`, `RF_knee` |
| Rear-left | `RL_hip_joint`, `RL_thigh_joint`, `RL_calf_joint` | `LB_adduction`, `LB_hip`, `LB_knee` |
| Rear-right | `RR_hip_joint`, `RR_thigh_joint`, `RR_calf_joint` | `RB_adduction`, `RB_hip`, `RB_knee` |

The SDK joint ordering used for deployment is:
```python
joint_sdk_names=[
    "LB_adduction", "LB_hip", "LB_knee",
    "LF_adduction", "LF_hip", "LF_knee",
    "RB_adduction", "RB_hip", "RB_knee",
    "RF_adduction", "RF_hip", "RF_knee",
]
```

---

## 6. Standalone Validation Script

A standalone controller was developed to verify the articulation before integrating RL. It confirms the robot loads correctly and tracks joint-level position commands.
# Articulation Setup

This document covers how to define and validate the Trakr robot as an Isaac Lab articulation.

---

## 1. ArticulationCfg

Isaac Lab uses `ArticulationCfg` to describe the robot's physical structure, initial state, and actuator models. The Trakr configuration is defined in `assets/robots/trakr.py`.

### USD Reference

```python
TRAKR_CFG = ArticulationCfg(
    prim_path="/World/Origin0/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/trakr_imu.usd",
        activate_contact_sensors=True,
    ),
)
```

> **Note:** `activate_contact_sensors=True` is required for foot-contact reward terms.

---

## 2. Physical Properties

### Rigid Body

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.1,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
),
```

### Articulation Solver

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
),
```

Values for damping, max velocities, and solver iterations are adopted from the official Trakr Legged RL repository by Addverb.

---

## 3. Initial State

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.255),
    joint_pos={
        ".*_adduction": 0.0,
        ".*F_hip":      0.0,
        ".*B_hip":      0.0,
        "L[F,B]_knee":  0.0,
        "R[F,B]_knee":  0.0,
    },
    joint_vel={".*": 0.0},
),
```

The robot spawns 0.255 m above the ground in a neutral standing posture. Joint groups are matched via regular expressions.

---

## 4. Actuator Model

```python
actuators={
    "base_legs": DCMotorCfg(
        joint_names_expr=[
            ".*_adduction",
            ".*_hip",
            ".*_knee",
        ],
        effort_limit=23.5,
        saturation_effort=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
        friction=0.0,
    ),
},
```

| Parameter | Value |
|---|---|
| Effort limit | 23.5 Nm |
| Saturation effort | 23.5 Nm |
| Velocity limit | 30.0 rad/s |
| Stiffness | 25.0 |
| Damping | 0.5 |
| Friction | 0.0 |

---

## 5. Joint Naming: Trakr vs. Unitree Go2

The Trakr model uses a different joint naming scheme from the Unitree Go2. The regular expressions in the articulation config must match Trakr's convention.

| Leg | Unitree Go2 | Trakr |
|---|---|---|
| Front-left | `FL_hip_joint`, `FL_thigh_joint`, `FL_calf_joint` | `LF_adduction`, `LF_hip`, `LF_knee` |
| Front-right | `FR_hip_joint`, `FR_thigh_joint`, `FR_calf_joint` | `RF_adduction`, `RF_hip`, `RF_knee` |
| Rear-left | `RL_hip_joint`, `RL_thigh_joint`, `RL_calf_joint` | `LB_adduction`, `LB_hip`, `LB_knee` |
| Rear-right | `RR_hip_joint`, `RR_thigh_joint`, `RR_calf_joint` | `RB_adduction`, `RB_hip`, `RB_knee` |

The SDK joint ordering used for deployment is:
```python
joint_sdk_names=[
    "LB_adduction", "LB_hip", "LB_knee",
    "LF_adduction", "LF_hip", "LF_knee",
    "RB_adduction", "RB_hip", "RB_knee",
    "RF_adduction", "RF_hip", "RF_knee",
]
```