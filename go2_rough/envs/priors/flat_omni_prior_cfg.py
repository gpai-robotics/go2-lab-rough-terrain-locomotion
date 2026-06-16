"""Omnidirectional flat prior used to seed the rough AsymPPO actor."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from go2_rough.envs.asymppo import command_curriculums
from go2_rough.envs.priors.flat_forward_prior_cfg import Go2FlatForwardPriorEnvCfg


@configclass
class Go2FlatOmniPriorEnvCfg(Go2FlatForwardPriorEnvCfg):
    """Flat omni prior with curriculum-expanded planar and yaw commands."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== GO2 FLAT OMNI PRIOR ==========\n")

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (-0.1, 0.1)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.1, 0.1)
        cmd.ranges.heading = None
        cmd.limit_ranges = cmd.ranges.__class__(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
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
