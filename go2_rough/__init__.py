import gymnasium as gym

from go2_rough.models.teacher_guided_blind_history_ppo import TeacherGuidedBlindHistoryPPO
from go2_rough.models.history_actor_critic import TemporalBlindActorCritic
from go2_rough.models.privileged_teacher_actor_critic import PrivilegedTeacherActorCritic
from go2_rough.models.terrain_supervised_teacher_ppo import TerrainSupervisedTeacherPPO
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

# Register the small set of custom classes needed by the extracted C1 line.
_rsl_on_policy_runner.TemporalBlindActorCritic = TemporalBlindActorCritic
_rsl_on_policy_runner.TeacherGuidedBlindHistoryPPO = TeacherGuidedBlindHistoryPPO
_rsl_on_policy_runner.PrivilegedTeacherActorCritic = PrivilegedTeacherActorCritic
_rsl_on_policy_runner.TerrainSupervisedTeacherPPO = TerrainSupervisedTeacherPPO


def _register_task(task_id: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str, *,
                   entry_point: str = "isaaclab.envs:ManagerBasedRLEnv") -> None:
    gym.register(
        id=task_id,
        entry_point=entry_point,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )


_register_task(
    "RMA-Go2-BlindHistory-Rough-StageA",
    "go2_rough.envs.blind_history_rough_cfg:Go2BlindHistoryRoughStudentEnvCfg",
    "go2_rough.configs.blind_history_ppo_cfg:Go2BlindHistoryRoughPPORunnerCfg",
)
_register_task(
    "RMA-Go2-PrivilegedTeacher-Rough-StageA",
    "go2_rough.envs.privileged_teacher_rough_cfg:Go2PrivilegedTeacherRoughEnvCfg",
    "go2_rough.configs.privileged_teacher_ppo_cfg:Go2PrivilegedTeacherRoughPPORunnerCfg",
)
