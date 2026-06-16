"""AsymPPO blind history env with critic-only privileged groups."""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from go2_rough.envs.asymppo.rough_history_base_cfg import Go2AsymPpoHistoryBaseEnvCfg
from go2_rough.envs.privileged_obs import (
    DynamicsPrivilegedObsCfg,
    TerrainPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


@configclass
class Go2AsymPpoPrivilegedHistoryEnvCfg(Go2AsymPpoHistoryBaseEnvCfg):
    """Blind-history actor env with terrain and dynamics privilege for the critic."""

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

        # Keep actor observations blind while exposing terrain only through
        # critic-side privileged groups.
        self.observations.policy.height_scan = None
        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.terrain_privileged = TerrainPrivilegedObsCfg()
        self.observations.dynamics_privileged = DynamicsPrivilegedObsCfg()

        print("\n========== GO2 ASYMPPO BLIND HISTORY ==========\n")
