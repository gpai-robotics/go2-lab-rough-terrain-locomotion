"""Shared deployment-contract constants for the Go2 rough policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from history_layout import ISAACLAB_TERM_MAJOR


GO2_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

GO2_ACTUATOR_NAMES = [
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
]

GO2_DEFAULT_JOINT_POS = [
    0.1,
    -0.1,
    0.1,
    -0.1,
    0.8,
    0.8,
    1.0,
    1.0,
    -1.5,
    -1.5,
    -1.5,
    -1.5,
]

GO2_BASE_INIT_POS = [0.0, 0.0, 0.4]
GO2_BASE_INIT_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]
GO2_JOINT_STIFFNESS = [25.0] * len(GO2_JOINT_NAMES)
GO2_JOINT_DAMPING = [0.5] * len(GO2_JOINT_NAMES)
GO2_EFFORT_LIMIT = [23.5] * len(GO2_JOINT_NAMES)
GO2_VELOCITY_LIMIT = [30.0] * len(GO2_JOINT_NAMES)
GO2_ACTION_SCALE = [0.25] * len(GO2_JOINT_NAMES)
GO2_ACTION_CLIP = [[-100.0, 100.0]] * len(GO2_JOINT_NAMES)

BLIND_HISTORY_POLICY_KIND = "blind_history_policy"
BLIND_HISTORY_OBSERVATION_GROUPS = ["policy", "policy_history"]
CONTROL_DT = 0.02
CONTROL_RATE_HZ = 1.0 / CONTROL_DT
ACTION_DIM = len(GO2_JOINT_NAMES)
HISTORY_LAYOUT = ISAACLAB_TERM_MAJOR

POLICY_ORDER = [
    {"name": "base_ang_vel", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
    {"name": "projected_gravity", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
    {"name": "velocity_commands", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
    {"name": "joint_pos_rel", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
    {"name": "joint_vel_rel", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
    {"name": "last_action", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bundle_dir(policy_name: str) -> Path:
    return repo_root() / "artifacts" / "exported" / policy_name


def policy_obs_dim() -> int:
    return sum(int(term["dim"]) for term in POLICY_ORDER)


def build_deploy_config(policy_history_length: int) -> dict[str, Any]:
    policy_dim = policy_obs_dim()
    return {
        "robot": {
            "joint_names": GO2_JOINT_NAMES,
            "actuator_names": GO2_ACTUATOR_NAMES,
            "base_init_pos": GO2_BASE_INIT_POS,
            "base_init_quat_wxyz": GO2_BASE_INIT_QUAT_WXYZ,
            "default_joint_pos": GO2_DEFAULT_JOINT_POS.copy(),
            "joint_stiffness": GO2_JOINT_STIFFNESS,
            "joint_damping": GO2_JOINT_DAMPING,
            "effort_limit": GO2_EFFORT_LIMIT,
            "velocity_limit": GO2_VELOCITY_LIMIT,
        },
        "actions": {
            "type": "JointPositionAction",
            "joint_names": GO2_JOINT_NAMES,
            "joint_ids": list(range(len(GO2_JOINT_NAMES))),
            "scale": GO2_ACTION_SCALE,
            "offset": GO2_DEFAULT_JOINT_POS.copy(),
            "clip": GO2_ACTION_CLIP,
            "use_default_offset": True,
        },
        "observations": {
            "policy_order": POLICY_ORDER,
            "policy_dim": policy_dim,
            "policy_history_length": int(policy_history_length),
            "policy_history_dim": policy_dim * int(policy_history_length),
            "policy_kind": BLIND_HISTORY_POLICY_KIND,
            "history_layout": HISTORY_LAYOUT,
            "use_gym_history": int(policy_history_length) > 0,
        },
        "commands": {
            "base_velocity": {
                "default": [0.5, 0.0, 0.0],
            }
        },
        "control": {
            "step_dt": CONTROL_DT,
            "physics_dt": 0.005,
            "decimation": 4,
        },
    }


def build_manifest(
    *,
    policy_name: str,
    source_checkpoint: str,
    task: str,
    phase: str,
    freeze_note: str = "",
    exported_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "policy_name": policy_name,
        "source_checkpoint": source_checkpoint,
        "task": task,
        "phase": phase,
        "policy_kind": BLIND_HISTORY_POLICY_KIND,
        "deployable_observation_groups": BLIND_HISTORY_OBSERVATION_GROUPS,
        "control_rate_hz": CONTROL_RATE_HZ,
        "history_layout": HISTORY_LAYOUT,
        "freeze_note": freeze_note,
        "exported_artifacts": list(exported_artifacts or []),
    }
