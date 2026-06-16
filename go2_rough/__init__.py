"""Public task registry for the Go2 rough-terrain AsymPPO path."""

from __future__ import annotations

import gymnasium as gym
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

from go2_rough.models.asymppo.history_actor_critic import TemporalBlindActorCritic


_rsl_on_policy_runner.TemporalBlindActorCritic = TemporalBlindActorCritic


def _register_task(task_id: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )


_register_task(
    "Go2-Flat-MJLAB-Prior-V1",
    "go2_rough.envs.priors.flat_mjlab_prior_cfg:Go2FlatMjlabPriorEnvCfg",
    "go2_rough.models.priors.flat_mjlab_prior_runner_cfg:Go2FlatMjlabPriorPPORunnerCfg",
)

_register_task(
    "Go2-Blind-Rough-MJLAB-AsymPPO-V1",
    "go2_rough.envs.asymppo.blind_rough_mjlab_asymppo_cfg:Go2BlindRoughMjlabAsymPpoEnvCfg",
    "go2_rough.models.asymppo.ppo_mjlab_asymppo_cfg:Go2BlindRoughMjlabAsymPpoRunnerCfg",
)
