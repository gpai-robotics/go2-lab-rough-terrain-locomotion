"""Minimal PPO config for the blind-history rough baseline."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class BlindHistoryPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TemporalBlindActorCritic"
    actor_init_path: str | None = None
    history_group_name: str = "policy_history"
    temporal_channels: list[int] = [64, 64]
    temporal_kernel_size: int = 3
    history_feature_dim: int = 64
    history_target_dim: int = 128
    history_target_hidden_dims: list[int] = [128]


@configclass
class BlindTeacherImitationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "TeacherGuidedBlindHistoryPPO"
    teacher_expert_path: str | None = None
    latent_command_threshold: float = 0.1
    latent_regression_coef_stage0: float = 0.5
    latent_regression_coef_stage1: float = 0.2
    latent_regression_coef_stage2: float = 0.05
    latent_stage0_end: int = 300
    latent_stage1_end: int = 800
    imitation_command_threshold: float = 0.1
    imitation_coef_stage0: float = 0.1
    imitation_coef_stage1: float = 0.02
    imitation_stage0_end: int = 300
    imitation_stage1_end: int = 800


@configclass
class Go2BlindHistoryRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_history_rough_v1"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=None,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )

    algorithm = BlindTeacherImitationAlgorithmCfg(
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
        teacher_expert_path=None,
        latent_command_threshold=0.1,
        latent_regression_coef_stage0=0.5,
        latent_regression_coef_stage1=0.2,
        latent_regression_coef_stage2=0.05,
        latent_stage0_end=300,
        latent_stage1_end=800,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.1,
        imitation_coef_stage1=0.02,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )
