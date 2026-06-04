"""Blind-history rough env with teacher-only terrain+dynamics privilege."""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from go2_rough.envs.blind_history_cfg import Go2BlindBaselineHistoryRoughEnvCfg
from go2_rough.envs.teacher_rough_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TeacherTerrainPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


@configclass
class Go2BlindHistoryRoughStudentEnvCfg(Go2BlindBaselineHistoryRoughEnvCfg):
    """Blind history student env with teacher-only terrain+dynamics privilege."""

    policy_history_length: int = 100

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

        self.observations.policy.height_scan = None
        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.terrain_privileged = TeacherTerrainPrivilegedObsCfg()
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()
