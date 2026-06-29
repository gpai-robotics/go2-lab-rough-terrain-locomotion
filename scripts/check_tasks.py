#!/usr/bin/env python3
"""Sanity-check that the Gym tasks register correctly."""

from __future__ import annotations

import gymnasium as gym

import go2_rough  # noqa: F401


def main() -> None:
    for task_id in ["Go2-Flat-MJLAB-Prior-V1", "Go2-Blind-Rough-MJLAB-AsymPPO-V1"]:
        print(gym.spec(task_id).id)


if __name__ == "__main__":
    main()
