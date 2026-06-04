"""Privileged rough-teacher environment."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from go2_rough.envs.blind_rough_cfg import Go2BlindBaselineRoughEnvCfg
from go2_rough.envs.teacher_rough_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TeacherTerrainPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


def moving_base_height_l2(
    env,
    target_height: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
):
    """Penalize base height error only when the commanded motion is non-trivial."""
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.linalg.norm(command[:, :2], dim=1) > command_threshold).float()
    height_penalty = base_mdp.base_height_l2(
        env,
        target_height=target_height,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return moving_mask * height_penalty


@configclass
class Go2PrivilegedTeacherRoughEnvCfg(Go2BlindBaselineRoughEnvCfg):
    """Privileged teacher env with terrain and hidden-dynamics groups."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.policy.height_scan = None
        self.observations.terrain_privileged = TeacherTerrainPrivilegedObsCfg()
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()

        self.rewards.base_height = RewTerm(
            func=moving_base_height_l2,
            weight=-5.0,
            params={
                "target_height": 0.33,
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
