"""Blind proprioceptive rough-terrain AsymPPO baseline.

This configuration is the active rough blind baseline.

It keeps the rough-terrain command and randomization family used by the frozen
blind ladder, but strips the task down to a clean deployable blind baseline:

- proprio-only actor and critic observations
- mixed rough terrain instead of the Stage D stair-specialized curriculum
- stronger early-failure terminations
- reward balance tuned for blind locomotion rather than terrain-aware shaping
"""

from __future__ import annotations

import os
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from go2_rough.envs.asset_contract import base_body_name, foot_body_regex, print_asset_contract


def stand_still_foot_motion_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    velocity_threshold: float = 0.2,
):
    """Penalize foot motion when the robot should be essentially standing still."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # For omni tasks, yaw commands are locomotion too. Treating pure-yaw as
    # standstill would incorrectly punish turning behavior.
    cmd_is_small = torch.linalg.norm(command[:, :3], dim=1) < command_threshold
    body_is_slow = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < velocity_threshold
    standstill = torch.logical_and(cmd_is_small, body_is_slow).unsqueeze(-1)
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    return torch.sum(foot_speed * standstill, dim=1)


def air_time_variance_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    min_recorded_air_time: float = 0.05,
    clip_max_air_time: float = 0.5,
):
    """Penalize large variance in completed foot swing times during commanded motion.

    This is intentionally mild structure, not a hard gait template. It nudges the
    policy away from messy asymmetric swing timing once a meaningful SE(2)
    locomotion command is present.
    """
    sensor = env.scene[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.norm(command[:, :3], dim=1) >= command_threshold

    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_air_time = torch.clamp(last_air_time, min=0.0, max=clip_max_air_time)
    has_recorded_swing = torch.amax(last_air_time, dim=1) >= min_recorded_air_time

    variance = torch.var(last_air_time, dim=1, correction=0)
    return variance * torch.logical_and(moving, has_recorded_swing).float()


def root_height_above_foot_plane_below(
    env,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
):
    """Terminate when the base collapses too close to the local foot support plane.

    Using env origin as the height reference is too blunt on descending stairs:
    the robot can pitch forward and move down a valid step while its root height
    relative to the patch origin drops sharply. Using the mean foot height keeps
    the collapse test tied to local support instead of global patch geometry.
    """
    asset = env.scene[asset_cfg.name]
    feet = env.scene[foot_cfg.name]
    foot_heights = feet.data.body_pos_w[:, foot_cfg.body_ids, 2]
    support_plane_height = foot_heights.mean(dim=1)
    clearance = asset.data.root_pos_w[:, 2] - support_plane_height
    return clearance < minimum_clearance


def low_progress_termination(
    env,
    min_command: float = 0.3,
    min_displacement: float = 0.2,
    min_planar_speed: float = 0.05,
    grace_period_s: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terminate episodes that are clearly stuck despite non-trivial commands."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    command_xy = torch.linalg.norm(command[:, :2], dim=1)
    planar_displacement = torch.linalg.norm(
        asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2],
        dim=1,
    )
    planar_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    grace_steps = int(grace_period_s / env.step_dt)

    return (
        (env.episode_length_buf > grace_steps)
        & (command_xy > min_command)
        & (planar_displacement < min_displacement)
        & (planar_speed < min_planar_speed)
    )

@configclass
class Go2AsymPpoRoughBaseEnvCfg(UnitreeGo2RoughEnvCfg):
    """Pure proprioceptive locomotion baseline with strong randomization."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== GO2 BLIND ROUGH ASYMPPO BASE ==========\n")
        print_asset_contract()

        if go2_usd_path := os.environ.get("GO2_USD_PATH"):
            self.scene.robot.spawn.usd_path = go2_usd_path
        self.scene.num_envs = 4096

        # Make playback/recording open in a robot-following view by default.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.8, 2.8, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.4)

        # This baseline stays purely proprioceptive.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.privileged = None

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.ranges.lin_vel_x = (0.0, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        cmd.rel_heading_envs = 0.0

        # Shared rough randomization family retained so B1/B2/B3 remain
        # comparable to the frozen B2 definition.
        self.events.physics_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2.0)
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)
            self.events.add_base_mass.params["asset_cfg"].body_names = base_body_name()
        self.events.motor_strength = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (0.6, 1.4),
                "damping_distribution_params": (0.6, 1.4),
                "operation": "scale",
            },
        )
        self.events.base_external_force_torque = None

        # General rough-terrain curriculum instead of the old stair specialist mix.
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel

        terrain_gen = self.scene.terrain.terrain_generator
        terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.0
        terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        terrain_gen.sub_terrains["boxes"].proportion = 0.0
        terrain_gen.sub_terrains["random_rough"].proportion = 0.2
        terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1

        # Stable blind-policy terminations.
        self.terminations.base_contact.params["sensor_cfg"].body_names = base_body_name()
        self.terminations.base_contact.params["threshold"] = 1.0
        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "limit_angle": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.terminations.base_height = DoneTerm(
            func=root_height_above_foot_plane_below,
            params={
                "minimum_clearance": 0.08,
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_cfg": SceneEntityCfg("robot", body_names=foot_body_regex()),
            },
        )
        self.terminations.low_progress = DoneTerm(
            func=low_progress_termination,
            params={
                "min_command": 0.3,
                "min_displacement": 0.25,
                "min_planar_speed": 0.1,
                "grace_period_s": 3.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Blind-policy reward balance: tracking dominates, regularization stays
        # present but intentionally weak.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.075

        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_pos_limits.weight = -0.05

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = foot_body_regex()
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_body_regex()),
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_body_regex()),
            },
        )
        self.rewards.foot_clearance = None
        self.rewards.base_height = None

        self.rewards.stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.2,
            params={"command_name": "base_velocity", "command_threshold": 0.15},
        )
        self.rewards.stand_still_foot_motion = RewTerm(
            func=stand_still_foot_motion_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_body_regex()),
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "velocity_threshold": 0.2,
            },
        )
        self.rewards.hip_joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # Reduce terrain-coupled and rough-terrain collision shaping.
        self.rewards.undesired_contacts = None
