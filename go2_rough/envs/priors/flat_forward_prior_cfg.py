"""Go2 flat prior backbone used to warm-start the rough AsymPPO actor."""

import os
import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from go2_rough.envs.asset_contract import foot_body_regex, print_asset_contract


def stand_still_foot_motion_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    velocity_threshold: float = 0.2,
):
    """Penalize foot motion when the robot is commanded to remain essentially still."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_is_small = torch.linalg.norm(command[:, :2], dim=1) < command_threshold
    body_is_slow = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) < velocity_threshold
    standstill = torch.logical_and(cmd_is_small, body_is_slow).unsqueeze(-1)
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    return torch.sum(foot_speed * standstill, dim=1)


def root_height_below_env_origin(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terminate when the base collapses too close to the local terrain frame."""
    asset = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return relative_height < minimum_height


@configclass
class Go2FlatForwardPriorEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        print("\n========== GO2 FLAT PRIOR BACKBONE ==========\n")
        print_asset_contract()

        if go2_usd_path := os.environ.get("GO2_USD_PATH"):
            self.scene.robot.spawn.usd_path = go2_usd_path
        self.scene.num_envs = 4096
        self.episode_length_s = 20.0

        # Make playback/recording open in a robot-following view by default.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, 2.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.35)

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.observations.privileged = None

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.0, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        if self.events.physics_material is not None:
            self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
            self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.9)
        self.events.add_base_mass = None
        self.events.base_com = None

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.3,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_body_regex()),
                "threshold": 0.5,
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=foot_body_regex()),
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_body_regex()),
            },
        )
        self.rewards.stand_still_joint_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.35,
            params={"command_name": "base_velocity", "command_threshold": 0.15},
        )
        self.rewards.stand_still_foot_motion = RewTerm(
            func=stand_still_foot_motion_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=foot_body_regex()),
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "velocity_threshold": 0.2,
            },
        )
        self.rewards.hip_joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.08,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )
        self.rewards.joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip|thigh|calf)_joint")},
        )

        # Match the old flat backbone recipe: explicit posture shaping stays
        # off and the controller is encouraged to discover nominal gait without
        # a base-height target term.
        self.rewards.base_height = None
        self.rewards.dof_pos_limits.weight = 0.0
