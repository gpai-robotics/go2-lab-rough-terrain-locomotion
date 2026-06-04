from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


class TemporalBlindActorCritic(ActorCritic):
    """Direct blind history-to-action policy with a small temporal encoder."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        history_group_name: str = "policy_history",
        temporal_channels: tuple[int] | list[int] = (64, 64),
        temporal_kernel_size: int = 3,
        history_feature_dim: int = 64,
        history_target_dim: int = 128,
        history_target_hidden_dims: tuple[int] | list[int] = (128,),
        actor_init_path: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        if kwargs:
            print(
                "TemporalBlindActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        self.history_group_name = history_group_name
        self.temporal_kernel_size = int(temporal_kernel_size)

        self._actor_has_history = history_group_name in obs_groups["policy"]
        self._critic_has_history = history_group_name in obs_groups["critic"]
        if not (self._actor_has_history or self._critic_has_history):
            raise ValueError(
                f"TemporalBlindActorCritic expects observation group '{history_group_name}' "
                "in either policy or critic mappings."
            )

        policy_obs = obs["policy"]
        history_obs = obs[history_group_name]
        if len(policy_obs.shape) != 2 or len(history_obs.shape) != 2:
            raise ValueError("TemporalBlindActorCritic only supports flattened batched observations.")
        self.policy_obs_dim = int(policy_obs.shape[-1])
        self.history_dim = int(history_obs.shape[-1])
        if self.history_dim % self.policy_obs_dim != 0:
            raise ValueError(
                f"History dim {self.history_dim} is not divisible by policy dim {self.policy_obs_dim}."
            )
        self.history_length = self.history_dim // self.policy_obs_dim
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization

        conv_layers: list[nn.Module] = []
        in_channels = self.policy_obs_dim
        for layer_idx, out_channels in enumerate(temporal_channels):
            dilation = 2**layer_idx
            padding = dilation * (self.temporal_kernel_size - 1) // 2
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    kernel_size=self.temporal_kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            conv_layers.append(nn.ELU())
            in_channels = int(out_channels)
        self.temporal_encoder = nn.Sequential(*conv_layers)
        self.history_projection = nn.Sequential(
            nn.Linear(in_channels * 2, int(history_feature_dim)),
            nn.ELU(),
        )
        self.history_target_head = MLP(
            int(history_feature_dim),
            int(history_target_dim),
            history_target_hidden_dims,
            activation,
        )

        actor_non_history_dim = 0
        for obs_group in obs_groups["policy"]:
            if obs_group != history_group_name:
                actor_non_history_dim += obs[obs_group].shape[-1]

        critic_non_history_dim = 0
        for obs_group in obs_groups["critic"]:
            if obs_group != history_group_name:
                critic_non_history_dim += obs[obs_group].shape[-1]

        num_actor_obs = actor_non_history_dim + (history_feature_dim if self._actor_has_history else 0)
        num_critic_obs = critic_non_history_dim + (history_feature_dim if self._critic_has_history else 0)

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if self.actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if self.critic_obs_normalization else nn.Identity()
        )

        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        if actor_init_path:
            self.load_actor_only(actor_init_path)

    def encode_history_feature(self, obs: TensorDict) -> torch.Tensor:
        history = obs[self.history_group_name].view(-1, self.history_length, self.policy_obs_dim)
        history = history.transpose(1, 2)
        temporal = self.temporal_encoder(history)
        pooled = temporal.mean(dim=-1)
        latest = temporal[:, :, -1]
        return self.history_projection(torch.cat([latest, pooled], dim=-1))

    def _encode_obs_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
        obs_list = []
        for group_name in group_names:
            if group_name == self.history_group_name:
                obs_list.append(self.encode_history_feature(obs))
            else:
                obs_list.append(obs[group_name])
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["critic"])

    def encode_history_target(self, obs: TensorDict) -> torch.Tensor:
        return self.history_target_head(self.encode_history_feature(obs))

    def load_actor_only(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        actor_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key[len("actor.") :]] = value
        if not actor_state:
            raise RuntimeError(f"No actor weights found in checkpoint: {checkpoint_path}")

        self._small_init_module(self.temporal_encoder)
        self._small_init_module(self.history_projection)
        self._small_init_module(self.history_target_head)
        self._partial_copy_mlp(self.actor, actor_state)

    @staticmethod
    def _small_init_module(module: nn.Module, gain: float = 0.1) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.xavier_uniform_(submodule.weight, gain=gain)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, nn.Conv1d):
                nn.init.xavier_uniform_(submodule.weight, gain=gain)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

    @staticmethod
    def _copy_if_present(target_tensor: torch.Tensor, source_state: dict[str, torch.Tensor], key: str) -> bool:
        if key not in source_state:
            return False
        source_tensor = source_state[key]
        if target_tensor.shape == source_tensor.shape:
            with torch.no_grad():
                target_tensor.copy_(source_tensor)
            return True
        return False

    def _partial_copy_mlp(self, module: nn.Module, source_state: dict[str, torch.Tensor]) -> None:
        if "0.weight" in source_state:
            source_weight = source_state["0.weight"]
            target_weight = module[0].weight
            with torch.no_grad():
                nn.init.xavier_uniform_(target_weight, gain=0.1)
                shared_in = min(source_weight.shape[1], target_weight.shape[1])
                shared_out = min(source_weight.shape[0], target_weight.shape[0])
                target_weight[:shared_out, :shared_in].copy_(source_weight[:shared_out, :shared_in])
        if "0.bias" in source_state:
            source_bias = source_state["0.bias"]
            target_bias = module[0].bias
            shared = min(source_bias.shape[0], target_bias.shape[0])
            with torch.no_grad():
                if target_bias is not None:
                    target_bias.zero_()
                target_bias[:shared].copy_(source_bias[:shared])

        for idx, layer in enumerate(module):
            if idx == 0 or not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, source_state, f"{idx}.weight")
            self._copy_if_present(layer.bias, source_state, f"{idx}.bias")
