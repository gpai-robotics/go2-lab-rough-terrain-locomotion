from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


class PrivilegedTeacherActorCritic(ActorCritic):
    """Teacher actor-critic with a dedicated encoder for privileged terrain scans."""

    is_recurrent: bool = False
    min_std: float = 1.0e-6

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        terrain_latent_dim: int = 32,
        terrain_encoder_hidden_dims: tuple[int] | list[int] = (128, 64),
        terrain_target_dim: int | None = None,
        terrain_target_hidden_dims: tuple[int] | list[int] | None = None,
        privileged_group_name: str = "privileged",
        warm_start_checkpoint_path: str | None = None,
        min_std: float = 0.05,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        self.privileged_group_name = privileged_group_name
        self.min_std = float(min_std)

        privileged_obs = obs[privileged_group_name]
        privileged_dim = privileged_obs.shape[-1]

        self.terrain_encoder = MLP(privileged_dim, terrain_latent_dim, terrain_encoder_hidden_dims, activation)
        self.terrain_target_dim = int(terrain_target_dim) if terrain_target_dim is not None else None
        self.terrain_target_head = None
        if self.terrain_target_dim is not None:
            hidden_dims = list(terrain_target_hidden_dims or [64])
            self.terrain_target_head = MLP(terrain_latent_dim, self.terrain_target_dim, hidden_dims, activation)

        actor_non_privileged_dim = 0
        for obs_group in obs_groups["policy"]:
            if obs_group != privileged_group_name:
                actor_non_privileged_dim += obs[obs_group].shape[-1]
        critic_non_privileged_dim = 0
        for obs_group in obs_groups["critic"]:
            if obs_group != privileged_group_name:
                critic_non_privileged_dim += obs[obs_group].shape[-1]

        num_actor_obs = actor_non_privileged_dim + terrain_latent_dim
        num_critic_obs = critic_non_privileged_dim + terrain_latent_dim

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            else:
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            else:
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self.distribution = None
        Normal.set_default_validate_args(False)

        if warm_start_checkpoint_path:
            self.load_from_blind_checkpoint(warm_start_checkpoint_path)

    def _encode_obs_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
        obs_list = []
        for group_name in group_names:
            if group_name == self.privileged_group_name:
                obs_list.append(self.terrain_encoder(obs[group_name]))
            else:
                obs_list.append(obs[group_name])
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["critic"])

    def encode_terrain_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.terrain_encoder(obs[self.privileged_group_name])

    def predict_terrain_target(self, obs: TensorDict) -> torch.Tensor:
        if self.terrain_target_head is None:
            raise RuntimeError("Terrain target head is not enabled for this policy.")
        return self.terrain_target_head(self.encode_terrain_latent(obs))

    def load_from_blind_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        self._small_init_module(self.terrain_encoder)
        if self.terrain_target_head is not None:
            self._small_init_module(self.terrain_target_head)
        self._partial_copy_mlp(self.actor, state_dict, "actor")
        self._partial_copy_mlp(self.critic, state_dict, "critic")

    @staticmethod
    def _small_init_module(module: nn.Module, gain: float = 0.1) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
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

    def _partial_copy_mlp(self, module: nn.Module, source_state: dict[str, torch.Tensor], prefix: str) -> None:
        first_weight_key = f"{prefix}.0.weight"
        first_bias_key = f"{prefix}.0.bias"
        if first_weight_key in source_state:
            source_weight = source_state[first_weight_key]
            target_weight = module[0].weight
            with torch.no_grad():
                nn.init.xavier_uniform_(target_weight, gain=0.1)
                shared_in = min(source_weight.shape[1], target_weight.shape[1])
                shared_out = min(source_weight.shape[0], target_weight.shape[0])
                target_weight[:shared_out, :shared_in].copy_(source_weight[:shared_out, :shared_in])
        if first_bias_key in source_state:
            source_bias = source_state[first_bias_key]
            target_bias = module[0].bias
            shared = min(source_bias.shape[0], target_bias.shape[0])
            with torch.no_grad():
                if target_bias is not None:
                    target_bias.zero_()
                target_bias[:shared].copy_(source_bias[:shared])

        for idx, layer in enumerate(module):
            if idx == 0 or not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, source_state, f"{prefix}.{idx}.weight")
            self._copy_if_present(layer.bias, source_state, f"{prefix}.{idx}.bias")
