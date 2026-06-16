"""Usable-range rough-terrain omni branch for the public AsymPPO path."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from go2_rough.envs.asymppo import command_curriculums
from go2_rough.envs.asymppo.rough_base_cfg import air_time_variance_penalty
from go2_rough.envs.asymppo.rough_privileged_history_cfg import Go2AsymPpoPrivilegedHistoryEnvCfg


@configclass
class Go2AsymPpoRoughOmniEnvCfg(Go2AsymPpoPrivilegedHistoryEnvCfg):
    """Rough omni candidate optimized for a strong deployable range."""

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.05
        cmd.ranges.lin_vel_x = (-0.1, 0.1)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.1, 0.1)
        cmd.ranges.heading = None
        cmd.limit_ranges = cmd.ranges.__class__(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
            heading=None,
        )

        self.curriculum.lin_vel_command_levels = CurrTerm(
            func=command_curriculums.lin_vel_cmd_levels,
            params={"reward_term_name": "track_lin_vel_xy_exp", "delta": 0.1},
        )
        self.curriculum.ang_vel_command_levels = CurrTerm(
            func=command_curriculums.ang_vel_cmd_levels,
            params={"reward_term_name": "track_ang_vel_z_exp", "delta": 0.1},
        )

        self.terminations.low_progress.params["min_displacement"] = 0.2
        self.terminations.low_progress.params["grace_period_s"] = 4.0

        self.rewards.stand_still_joint_deviation.params["command_threshold"] = 0.2
        self.rewards.stand_still_foot_motion.params["command_threshold"] = 0.2
        self.rewards.air_time_variance = RewTerm(
            func=air_time_variance_penalty,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "command_threshold": 0.2,
                "min_recorded_air_time": 0.05,
                "clip_max_air_time": 0.5,
            },
        )

        print("\n========== GO2 BLIND ROUGH ASYMPPO OMNI V1 ==========\n")
