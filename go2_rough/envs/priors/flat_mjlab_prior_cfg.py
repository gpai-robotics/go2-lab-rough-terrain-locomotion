"""Flat omni prior for the mjlab-style branch."""

from __future__ import annotations

from isaaclab.utils import configclass

from go2_rough.envs.mjlab_contract import MjlabCriticPrivilegedObsCfg, apply_mjlab_policy_contract
from go2_rough.envs.priors.flat_omni_prior_cfg import Go2FlatOmniPriorEnvCfg


@configclass
class Go2FlatMjlabPriorEnvCfg(Go2FlatOmniPriorEnvCfg):
    """Flat omni prior under the deploy-honest mjlab-style actor contract."""

    mjlab_use_gait_phase: bool = False
    mjlab_encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    mjlab_obs_delay_min_lag: int = 0
    mjlab_obs_delay_max_lag: int = 1
    mjlab_obs_delay_hold_prob: float = 0.85
    mjlab_obs_delay_update_period: int = 4

    def __post_init__(self):
        super().__post_init__()

        apply_mjlab_policy_contract(
            self.observations.policy,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        self.observations.critic_privileged = MjlabCriticPrivilegedObsCfg()

        self.commands.base_velocity.resampling_time_range = (2.0, 5.0)

        if self.events.physics_material is not None:
            self.events.physics_material.params["static_friction_range"] = (0.5, 1.1)
            self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.encoder_bias = None
        self.events.motor_strength = None
        self.events.motor_strength_hip_thigh = None
        self.events.motor_strength_calf = None

        print("\n========== GO2 FLAT MJLAB-CONTRACT PRIOR ==========\n")
