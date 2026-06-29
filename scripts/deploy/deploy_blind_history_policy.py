"""Shared deployable blind-history policy reconstruction helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to CLI
        raise SystemExit(
            "PyTorch is not available in this Python environment. "
            "Run this script with the IsaacLab Python environment."
        ) from exc
    return torch, nn


def collect_prefix(state_dict: OrderedDict[str, object], prefix: str) -> OrderedDict[str, object]:
    collected = OrderedDict()
    prefix_with_dot = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            collected[key[len(prefix_with_dot) :]] = value
    if not collected:
        raise RuntimeError(f"Missing state dict prefix '{prefix_with_dot}'.")
    return collected


def infer_mlp_dims(module_state: OrderedDict[str, object]) -> tuple[int, list[int], int]:
    linear_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    weight_shapes = [tuple(module_state[f"{idx}.weight"].shape) for idx in linear_indices]
    input_dim = int(weight_shapes[0][1])
    hidden_dims = [int(shape[0]) for shape in weight_shapes[:-1]]
    output_dim = int(weight_shapes[-1][0])
    return input_dim, hidden_dims, output_dim


def build_mlp(nn, input_dim: int, hidden_dims: list[int], output_dim: int):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def build_history_projection(nn, input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ELU(),
    )


def infer_conv1d_layers(module_state: OrderedDict[str, object]) -> list[tuple[int, int, int, int, int, int]]:
    conv_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    layers = []
    for order_idx, module_idx in enumerate(conv_indices):
        weight = module_state[f"{module_idx}.weight"]
        out_channels, in_channels, kernel_size = weight.shape
        dilation = 2**order_idx
        padding = dilation * (int(kernel_size) - 1) // 2
        layers.append((module_idx, int(in_channels), int(out_channels), int(kernel_size), int(dilation), int(padding)))
    return layers


def load_checkpoint_state(checkpoint_path: Path):
    torch, _ = import_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")
    return checkpoint["model_state_dict"]


def build_deployable_module(
    *,
    state_dict: OrderedDict[str, object],
    policy_obs_dim: int,
    policy_history_length: int,
):
    torch, nn = import_torch()

    class DeployableBlindHistoryPolicy(nn.Module):
        def __init__(self, state_dict: OrderedDict[str, object], policy_history_length: int) -> None:
            super().__init__()
            temporal_state = collect_prefix(state_dict, "temporal_encoder")
            projection_state = collect_prefix(state_dict, "history_projection")
            actor_state = collect_prefix(state_dict, "actor")

            conv_specs = infer_conv1d_layers(temporal_state)
            conv_layers: list[nn.Module] = []
            for module_idx, in_channels, out_channels, kernel_size, dilation, padding in conv_specs:
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                conv.weight.data.copy_(temporal_state[f"{module_idx}.weight"])
                conv.bias.data.copy_(temporal_state[f"{module_idx}.bias"])
                conv_layers.append(conv)
                conv_layers.append(nn.ELU())
            self.temporal_encoder = nn.Sequential(*conv_layers)

            projection_input_dim, projection_hidden_dims, history_feature_dim = infer_mlp_dims(projection_state)
            actor_input_dim, actor_hidden_dims, action_dim = infer_mlp_dims(actor_state)
            inferred_policy_obs_dim = actor_input_dim - history_feature_dim
            if inferred_policy_obs_dim <= 0:
                raise RuntimeError(
                    "Invalid exported history-policy actor input contract. "
                    f"Expected positive policy_obs_dim, got {inferred_policy_obs_dim}."
                )
            if inferred_policy_obs_dim != policy_obs_dim:
                raise RuntimeError(
                    "Checkpoint actor contract does not match this repo's deploy contract. "
                    f"Expected policy_obs_dim={policy_obs_dim}, got {inferred_policy_obs_dim}."
                )
            if projection_hidden_dims:
                raise RuntimeError(
                    "Unexpected blind-history projection architecture during export. "
                    f"Expected a single linear layer + ELU, got hidden dims {projection_hidden_dims}."
                )

            self.policy_obs_dim = inferred_policy_obs_dim
            self.history_dim = inferred_policy_obs_dim * int(policy_history_length)
            self.action_dim = action_dim
            self.policy_history_length = int(policy_history_length)
            self.history_projection = build_history_projection(nn, projection_input_dim, history_feature_dim)
            self.actor = build_mlp(nn, actor_input_dim, actor_hidden_dims, action_dim)
            self.history_projection.load_state_dict(projection_state, strict=True)
            self.actor.load_state_dict(actor_state, strict=True)

        def encode_history_feature(self, policy_history):
            history = policy_history.view(-1, self.policy_history_length, self.policy_obs_dim).transpose(1, 2)
            temporal = self.temporal_encoder(history)
            pooled = temporal.mean(dim=-1)
            latest = temporal[:, :, -1]
            return self.history_projection(torch.cat([latest, pooled], dim=-1))

        def forward(self, policy_obs, policy_history):
            history_feature = self.encode_history_feature(policy_history)
            actor_obs = torch.cat([policy_obs, history_feature], dim=-1)
            return self.actor(actor_obs)

    return DeployableBlindHistoryPolicy(state_dict, policy_history_length).eval()
