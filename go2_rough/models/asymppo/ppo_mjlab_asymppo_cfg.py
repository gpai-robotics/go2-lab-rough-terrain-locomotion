"""Runner config for the MJLAB-style asymmetric PPO rough blind branch."""

import os
from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner
from go2_rough.models.asymppo.history_actor_critic import TemporalBlindActorCritic
from go2_rough.models.asymppo.policy_cfg import AsymPpoHistoryPolicyCfg


_rsl_on_policy_runner.TemporalBlindActorCritic = TemporalBlindActorCritic

MJLAB_FLAT_PRIOR_CKPT = os.environ.get("GO2_FLAT_PRIOR_CKPT", "")


@configclass
class Go2BlindRoughMjlabAsymPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_rough_asymppo_mjlab_v1"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history", "critic_privileged", "dynamics_privileged", "terrain_privileged"],
    }

    policy = AsymPpoHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=MJLAB_FLAT_PRIOR_CKPT if MJLAB_FLAT_PRIOR_CKPT and Path(MJLAB_FLAT_PRIOR_CKPT).exists() else None,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )
