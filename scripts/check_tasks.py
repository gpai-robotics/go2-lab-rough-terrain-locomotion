#!/usr/bin/env python3
"""Sanity-check that the Gym tasks register correctly."""

from __future__ import annotations

import sys

try:
    import gymnasium as gym
except ModuleNotFoundError as exc:
    if exc.name == "gymnasium":
        print("gymnasium is not available in this Python environment.", file=sys.stderr)
        print(
            "Run this script through IsaacLab, for example:",
            file=sys.stderr,
        )
        print(
            "  $ISAACLAB_ROOT/isaaclab.sh -p scripts/check_tasks.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise

import go2_rough  # noqa: F401


def main() -> None:
    task_ids = [
        "RMA-Go2-PrivilegedTeacher-Rough-StageA",
        "RMA-Go2-BlindHistory-Rough-StageA",
    ]

    for task_id in task_ids:
        print(gym.spec(task_id).id)


if __name__ == "__main__":
    main()
