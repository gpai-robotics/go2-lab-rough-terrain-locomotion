"""Shared joint-order schema helpers for Go2 hardware telemetry tools."""

from __future__ import annotations

import numpy as np

SCHEMA_NAME = "go2_policy_order_monitor"
SCHEMA_VERSION = 1

POLICY_JOINT_NAMES = [
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

# Unitree low-level SDK order and policy order differ.
# These index arrays keep conversions explicit and shared across tools.
SDK_TO_POLICY = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
POLICY_TO_SDK = np.zeros_like(SDK_TO_POLICY)
POLICY_TO_SDK[SDK_TO_POLICY] = np.arange(12, dtype=np.int64)
