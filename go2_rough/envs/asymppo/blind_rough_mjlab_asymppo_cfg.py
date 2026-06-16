"""MJLAB-contract asymmetric PPO rough blind policy config.

The actor is deployable and blind. The critic is privileged during asymmetric
PPO training.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from go2_rough.envs.asymppo.rough_omni_cfg import Go2AsymPpoRoughOmniEnvCfg
from go2_rough.envs.mjlab_contract import MjlabCriticPrivilegedObsCfg, apply_mjlab_policy_contract


@configclass
class Go2BlindRoughMjlabAsymPpoEnvCfg(Go2AsymPpoRoughOmniEnvCfg):
    """Rough omni baseline with a deploy-honest MJLAB actor contract."""

    mjlab_use_gait_phase: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Keep the same robot/actuator model as the frozen flat prior. The
        # mjlab actuator prior is a separate ablation and needs its own flat.
        apply_mjlab_policy_contract(
            self.observations.policy,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        apply_mjlab_policy_contract(
            self.observations.policy_history,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        self.observations.critic_privileged = MjlabCriticPrivilegedObsCfg()

        # Keep the proven deployment dynamics envelope for the clean AsymPPO diagnosis.
        # The wider kp/kd and COM ablations both suppressed terrain progression.
        self.events.motor_strength.params["stiffness_distribution_params"] = (0.6, 1.4)
        self.events.motor_strength.params["damping_distribution_params"] = (0.6, 1.4)

        # Keep the recovery and COM disturbance pressure from the previous run,
        # but remove the wide-gain ablation so we isolate gain randomization.
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.35, 0.35),
                    "yaw": (-0.4, 0.4),
                },
            },
        )
        self.events.base_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (-0.01, 0.01),
                },
            },
        )

        print("\n========== GO2 BLIND ROUGH MJLAB ASYMMETRIC PPO ==========\n")
