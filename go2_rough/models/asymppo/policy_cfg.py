"""Policy configuration for the active temporal AsymPPO actor."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class AsymPpoHistoryPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TemporalBlindActorCritic"
    actor_init_path: str | None = None
    history_group_name: str = "policy_history"
    temporal_channels: list[int] = [64, 64]
    temporal_kernel_size: int = 3
    history_feature_dim: int = 64
    history_target_dim: int = 128
    history_target_hidden_dims: list[int] = [128]
