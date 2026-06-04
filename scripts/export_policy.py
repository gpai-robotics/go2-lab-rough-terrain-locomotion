#!/usr/bin/env python3
"""Export the blind-history student into a simple deployment bundle."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="RMA-Go2-BlindHistory-Rough-StageA")
    parser.add_argument("--phase", default="blind-history-stagea")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--format", action="append", dest="formats", default=[])
    return parser.parse_args()


def _import_task_cfg_loader():
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import isaaclab_tasks  # noqa: F401
    import go2_rough  # noqa: F401
    return load_cfg_from_registry


def _collect_prefix(state_dict: OrderedDict[str, object], prefix: str) -> OrderedDict[str, object]:
    collected = OrderedDict()
    prefix_with_dot = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            collected[key[len(prefix_with_dot) :]] = value
    if not collected:
        raise RuntimeError(f"Missing state dict prefix '{prefix_with_dot}'.")
    return collected


def _infer_mlp_dims(module_state: OrderedDict[str, object]) -> tuple[int, list[int], int]:
    linear_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    weight_shapes = [tuple(module_state[f"{idx}.weight"].shape) for idx in linear_indices]
    input_dim = weight_shapes[0][1]
    hidden_dims = [shape[0] for shape in weight_shapes[:-1]]
    output_dim = weight_shapes[-1][0]
    return input_dim, hidden_dims, output_dim


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _infer_conv1d_layers(module_state: OrderedDict[str, object]) -> list[tuple[int, int, int, int, int]]:
    conv_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    layers = []
    for order_idx, module_idx in enumerate(conv_indices):
        weight = module_state[f"{module_idx}.weight"]
        out_channels, in_channels, kernel_size = weight.shape
        dilation = 2**order_idx
        padding = dilation * (int(kernel_size) - 1) // 2
        layers.append((module_idx, int(in_channels), int(out_channels), int(kernel_size), int(dilation), int(padding)))
    return layers


class DeployableBlindHistoryPolicy(nn.Module):
    def __init__(self, state_dict: OrderedDict[str, object], history_length: int) -> None:
        super().__init__()
        temporal_state = _collect_prefix(state_dict, "temporal_encoder")
        projection_state = _collect_prefix(state_dict, "history_projection")
        actor_state = _collect_prefix(state_dict, "actor")

        conv_specs = _infer_conv1d_layers(temporal_state)
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

        proj_input_dim, proj_hidden_dims, proj_output_dim = _infer_mlp_dims(projection_state)
        if proj_hidden_dims != [proj_output_dim]:
            pass
        self.history_projection = nn.Sequential(
            nn.Linear(proj_input_dim, proj_output_dim),
            nn.ELU(),
        )
        self.history_projection[0].weight.data.copy_(projection_state["0.weight"])
        self.history_projection[0].bias.data.copy_(projection_state["0.bias"])

        actor_input_dim, actor_hidden_dims, action_dim = _infer_mlp_dims(actor_state)
        self.actor = _build_mlp(actor_input_dim, actor_hidden_dims, action_dim)
        self.actor.load_state_dict(actor_state, strict=True)

        self.policy_obs_dim = int(actor_input_dim - proj_output_dim)
        self.history_length = int(history_length)
        self.history_dim = int(self.policy_obs_dim * self.history_length)
        self.action_dim = int(action_dim)

    def forward(self, policy_obs, policy_history):
        history = policy_history.view(-1, self.history_length, self.policy_obs_dim).transpose(1, 2)
        temporal = self.temporal_encoder(history)
        pooled = temporal.mean(dim=-1)
        latest = temporal[:, :, -1]
        history_feature = self.history_projection(torch.cat([latest, pooled], dim=-1))
        actor_obs = torch.cat([policy_obs, history_feature], dim=-1)
        return self.actor(actor_obs)


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    formats = args.formats or ["torchscript"]

    load_cfg_from_registry = _import_task_cfg_loader()
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    history_length = int(getattr(env_cfg, "policy_history_length", 100))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    module = DeployableBlindHistoryPolicy(state_dict, history_length)
    module.eval()

    artifact_names: list[str] = []
    dummy_policy = torch.zeros(1, module.policy_obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, module.history_dim, dtype=torch.float32)

    if "torchscript" in formats:
        ts_name = f"{args.policy_name}.torchscript.pt"
        traced = torch.jit.trace(module, (dummy_policy, dummy_history))
        traced.save(str(bundle_dir / ts_name))
        artifact_names.append(ts_name)

    if "onnx" in formats:
        onnx_name = f"{args.policy_name}.onnx"
        torch.onnx.export(
            module,
            (dummy_policy, dummy_history),
            str(bundle_dir / onnx_name),
            input_names=["policy_obs", "policy_history"],
            output_names=["action"],
            dynamic_axes={"policy_obs": {0: "batch"}, "policy_history": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        artifact_names.append(onnx_name)

    metadata_name = f"{args.policy_name}.export_metadata.json"
    metadata = {
        "policy_name": args.policy_name,
        "source_checkpoint": str(checkpoint_path),
        "task": args.task,
        "phase": args.phase,
        "policy_kind": "blind_history_policy",
        "policy_obs_dim": module.policy_obs_dim,
        "policy_history_length": history_length,
        "policy_history_dim": module.history_dim,
        "action_dim": module.action_dim,
    }
    (bundle_dir / metadata_name).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    artifact_names.append(metadata_name)

    deploy_name = f"{args.policy_name}.deploy_config.json"
    deploy_payload = {
        "observations": {
            "policy_dim": module.policy_obs_dim,
            "policy_history_length": history_length,
            "policy_history_dim": module.history_dim,
        },
        "control": {
            "step_dt": 0.02,
            "physics_dt": 0.005,
            "decimation": 4,
            "control_rate_hz": 50,
        },
    }
    (bundle_dir / deploy_name).write_text(json.dumps(deploy_payload, indent=2, sort_keys=True) + "\n")
    artifact_names.append(deploy_name)

    manifest = {
        "policy_name": args.policy_name,
        "source_checkpoint": str(checkpoint_path),
        "task": args.task,
        "phase": args.phase,
        "policy_kind": "blind_history_policy",
        "deployable_observation_groups": ["policy", "policy_history"],
        "control_rate_hz": 50,
        "exported_artifacts": artifact_names,
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (bundle_dir / "export_request.json").write_text(
        json.dumps(
            {
                "policy_name": args.policy_name,
                "checkpoint": str(checkpoint_path),
                "task": args.task,
                "phase": args.phase,
                "requested_formats": formats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
