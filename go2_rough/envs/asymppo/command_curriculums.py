"""Curriculum helpers for blind locomotion variants."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    delta: float = 0.1,
):
    """Expand linear command ranges when XY tracking is strong enough.

    Mirrors the reference IsaacLab/Unitree pattern:
    start with narrow command ranges and widen them only after the policy shows
    consistent tracking on the current regime.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    current_window = int(env.common_step_counter // env.max_episode_length)
    last_update_window = getattr(command_term.cfg, "_lin_vel_last_update_window", -1)

    if current_window > last_update_window and reward > reward_term.weight * 0.8:
        delta_command = torch.tensor([-delta, delta], device=env.device)
        ranges.lin_vel_x = torch.clamp(
            torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
            limit_ranges.lin_vel_x[0],
            limit_ranges.lin_vel_x[1],
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
            limit_ranges.lin_vel_y[0],
            limit_ranges.lin_vel_y[1],
        ).tolist()
        command_term.cfg._lin_vel_last_update_window = current_window

    return {
        "lin_vel_x_min": ranges.lin_vel_x[0],
        "lin_vel_x_max": ranges.lin_vel_x[1],
        "lin_vel_y_min": ranges.lin_vel_y[0],
        "lin_vel_y_max": ranges.lin_vel_y[1],
    }


def ang_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
    delta: float = 0.1,
):
    """Expand yaw-rate command range when yaw tracking is strong enough."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    current_window = int(env.common_step_counter // env.max_episode_length)
    last_update_window = getattr(command_term.cfg, "_ang_vel_last_update_window", -1)

    if current_window > last_update_window and reward > reward_term.weight * 0.8:
        delta_command = torch.tensor([-delta, delta], device=env.device)
        ranges.ang_vel_z = torch.clamp(
            torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
            limit_ranges.ang_vel_z[0],
            limit_ranges.ang_vel_z[1],
        ).tolist()
        command_term.cfg._ang_vel_last_update_window = current_window

    return {
        "ang_vel_z_min": ranges.ang_vel_z[0],
        "ang_vel_z_max": ranges.ang_vel_z[1],
    }
