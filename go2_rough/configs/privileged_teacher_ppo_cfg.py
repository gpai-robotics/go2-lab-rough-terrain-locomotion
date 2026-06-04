"""Minimal PPO config for the privileged rough teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class PrivilegedTeacherPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "PrivilegedTeacherActorCritic"
    terrain_latent_dim: int = 8
    terrain_encoder_hidden_dims: list[int] = [64, 32]
    terrain_target_dim: int = 13
    terrain_target_hidden_dims: list[int] = [64]
    privileged_group_name: str = "terrain_privileged"
    warm_start_checkpoint_path: str | None = None


@configclass
class TerrainSupervisedTeacherAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "TerrainSupervisedTeacherPPO"
    terrain_regression_coef_stage0: float = 0.5
    terrain_regression_coef_stage1: float = 0.2
    terrain_regression_coef_stage2: float = 0.05
    terrain_stage0_end: int = 300
    terrain_stage1_end: int = 800


@configclass
class Go2PrivilegedTeacherRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_privileged_teacher_rough_v1"

    obs_groups = {
        "policy": ["policy", "dynamics_privileged", "terrain_privileged"],
        "critic": ["policy", "dynamics_privileged", "terrain_privileged"],
    }

    policy = PrivilegedTeacherPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        terrain_latent_dim=8,
        terrain_encoder_hidden_dims=[64, 32],
        terrain_target_dim=13,
        terrain_target_hidden_dims=[64],
        privileged_group_name="terrain_privileged",
        warm_start_checkpoint_path=None,
    )

    algorithm = TerrainSupervisedTeacherAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        terrain_regression_coef_stage0=0.5,
        terrain_regression_coef_stage1=0.2,
        terrain_regression_coef_stage2=0.05,
        terrain_stage0_end=300,
        terrain_stage1_end=800,
    )
