from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from go2_rough.models.privileged_teacher_actor_critic import PrivilegedTeacherActorCritic


class FrozenTeacherPolicy(nn.Module):
    """Read-only privileged teacher rebuilt from a frozen teacher checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        super().__init__()

        dummy_obs = TensorDict(
            {
                "policy": torch.zeros(1, 48),
                "dynamics_privileged": torch.zeros(1, 27),
                "terrain_privileged": torch.zeros(1, 187),
            },
            batch_size=[1],
        )
        obs_groups = {
            "policy": ["policy", "dynamics_privileged", "terrain_privileged"],
            "critic": ["policy", "dynamics_privileged", "terrain_privileged"],
        }
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        has_terrain_target_head = any(key.startswith("terrain_target_head.") for key in state_dict.keys())
        self.policy = PrivilegedTeacherActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=12,
            init_noise_std=0.35,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            terrain_latent_dim=8,
            terrain_encoder_hidden_dims=[64, 32],
            terrain_target_dim=13 if has_terrain_target_head else None,
            terrain_target_hidden_dims=[64] if has_terrain_target_head else None,
            privileged_group_name="terrain_privileged",
        )
        self.policy.load_state_dict(state_dict, strict=True)
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs)

    def get_latent_target(self, obs: TensorDict) -> torch.Tensor:
        x = self.policy.get_actor_obs(obs)
        x = self.policy.actor_obs_normalizer(x)
        actor_layers = list(self.policy.actor.children())
        for layer in actor_layers[:-1]:
            x = layer(x)
        return x
