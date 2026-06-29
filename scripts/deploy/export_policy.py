#!/usr/bin/env python3
"""Export the blind-history Go2 policy into local deployment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from deploy_blind_history_policy import build_deployable_module, import_torch, load_checkpoint_state
from go2_deploy_contract import (
    BLIND_HISTORY_OBSERVATION_GROUPS,
    BLIND_HISTORY_POLICY_KIND,
    build_deploy_config,
    build_manifest,
    default_bundle_dir,
    policy_obs_dim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True, help="Bundle policy name.")
    parser.add_argument("--checkpoint", required=True, help="Frozen training checkpoint to export.")
    parser.add_argument("--task", required=True, help="Registered training task name.")
    parser.add_argument("--phase", required=True, help="Training lineage label.")
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Output bundle directory. Defaults to artifacts/exported/<policy-name> inside this repo.",
    )
    parser.add_argument(
        "--policy-kind",
        default=BLIND_HISTORY_POLICY_KIND,
        choices=[BLIND_HISTORY_POLICY_KIND],
        help="Only the active blind-history runtime contract is supported here.",
    )
    parser.add_argument(
        "--observation-groups",
        default="policy,policy_history",
        help="Comma-separated deployment observation groups.",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=[],
        help="Requested export format. Repeat for multiple formats. Supported: torchscript, onnx.",
    )
    parser.add_argument(
        "--policy-history-length",
        type=int,
        default=0,
        help="Deployment history length. Use 0 to infer from the task env config.",
    )
    parser.add_argument("--freeze-note", default="", help="Optional note persisted in the bundle manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned export request and exit.")
    return parser.parse_args()


def _import_task_cfg_loader():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        import isaaclab_tasks  # noqa: F401
        import go2_rough  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "IsaacLab task config utilities are not available in this Python environment. "
            "Run this script with the IsaacLab Python environment."
        ) from exc
    return load_cfg_from_registry


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _infer_history_length_from_task(task: str) -> int:
    try:
        load_cfg_from_registry = _import_task_cfg_loader()
        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    except Exception as exc:
        raise SystemExit(
            f"Could not load task config for '{task}' to infer policy history length. "
            "Pass --policy-history-length explicitly."
        ) from exc

    for attr_name in ("policy_history_length", "adaptation_history_length"):
        value = getattr(env_cfg, attr_name, None)
        if isinstance(value, int) and value > 0:
            return value

    obs_cfg = getattr(env_cfg, "observations", None)
    policy_history_cfg = getattr(obs_cfg, "policy_history", None) if obs_cfg is not None else None
    value = getattr(policy_history_cfg, "history_length", None) if policy_history_cfg is not None else None
    if isinstance(value, int) and value > 0:
        return value

    raise SystemExit(
        f"Could not infer policy history length for task '{task}'. Pass --policy-history-length explicitly."
    )


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "[", "]", "{", "}", ",", "'"]):
            return json.dumps(value)
        return value
    return str(value)


def _yaml_dump(value, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text("\n".join(_yaml_dump(payload)) + "\n")


def _export_torchscript(torch, module, bundle_dir: Path, policy_name: str) -> str:
    output_name = f"{policy_name}.torchscript.pt"
    output_path = bundle_dir / output_name
    dummy_policy = torch.zeros(1, module.policy_obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, module.history_dim, dtype=torch.float32)
    traced = torch.jit.trace(module, (dummy_policy, dummy_history))
    traced.save(str(output_path))
    return output_name


def _export_onnx(torch, module, bundle_dir: Path, policy_name: str) -> str:
    output_name = f"{policy_name}.onnx"
    output_path = bundle_dir / output_name
    dummy_policy = torch.zeros(1, module.policy_obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, module.history_dim, dtype=torch.float32)
    torch.onnx.export(
        module,
        (dummy_policy, dummy_history),
        str(output_path),
        input_names=["policy_obs", "policy_history"],
        output_names=["action"],
        dynamic_axes={
            "policy_obs": {0: "batch"},
            "policy_history": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )
    return output_name


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else default_bundle_dir(args.policy_name)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    formats = args.formats or ["torchscript"]
    deployable_observation_groups = _parse_csv(args.observation_groups)

    if deployable_observation_groups != BLIND_HISTORY_OBSERVATION_GROUPS:
        raise SystemExit(
            f"Blind-history export requires observation groups {BLIND_HISTORY_OBSERVATION_GROUPS}, "
            f"got {deployable_observation_groups}."
        )

    supported_formats = {"torchscript", "onnx"}
    unsupported = [fmt for fmt in formats if fmt not in supported_formats]
    if unsupported:
        raise SystemExit(
            f"Unsupported export format(s): {', '.join(unsupported)}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    resolved_history_length = (
        args.policy_history_length if args.policy_history_length > 0 else _infer_history_length_from_task(args.task)
    )

    request_payload = {
        "policy_name": args.policy_name,
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "phase": args.phase,
        "policy_kind": BLIND_HISTORY_POLICY_KIND,
        "requested_formats": formats,
        "resolved_policy_history_length": resolved_history_length,
    }
    if args.dry_run:
        request_payload["status"] = "planned"
        request_payload["bundle_dir"] = str(bundle_dir)
        print(json.dumps(request_payload, indent=2, sort_keys=True))
        return 0

    torch, _ = import_torch()

    bundle_dir.mkdir(parents=True, exist_ok=True)
    module = build_deployable_module(
        state_dict=load_checkpoint_state(checkpoint_path),
        policy_obs_dim=policy_obs_dim(),
        policy_history_length=resolved_history_length,
    )

    artifact_names: list[str] = []
    if "torchscript" in formats:
        artifact_names.append(_export_torchscript(torch, module, bundle_dir, args.policy_name))
    if "onnx" in formats:
        artifact_names.append(_export_onnx(torch, module, bundle_dir, args.policy_name))

    metadata_name = f"{args.policy_name}.export_metadata.json"
    _write_json(
        bundle_dir / metadata_name,
        {
            "policy_name": args.policy_name,
            "source_checkpoint": str(checkpoint_path),
            "task": args.task,
            "phase": args.phase,
            "runtime_contract": {
                "policy_kind": BLIND_HISTORY_POLICY_KIND,
                "deployable_observation_groups": BLIND_HISTORY_OBSERVATION_GROUPS,
            },
            "tensor_contract": {
                "policy_obs_dim": module.policy_obs_dim,
                "policy_history_dim": module.history_dim,
                "history_length": module.policy_history_length,
                "action_dim": module.action_dim,
                "forward_signature": {
                    "inputs": {
                        "policy_obs": ["batch", module.policy_obs_dim],
                        "policy_history": ["batch", module.history_dim],
                    },
                    "outputs": {
                        "action": ["batch", module.action_dim],
                    },
                },
            },
        },
    )
    artifact_names.append(metadata_name)

    deploy_config_name = f"{args.policy_name}.deploy_config.json"
    deploy_config = build_deploy_config(module.policy_history_length)
    _write_json(bundle_dir / deploy_config_name, deploy_config)
    artifact_names.append(deploy_config_name)

    deploy_yaml_name = f"{args.policy_name}.deploy.yaml"
    _write_yaml(bundle_dir / deploy_yaml_name, deploy_config)
    artifact_names.append(deploy_yaml_name)

    manifest = build_manifest(
        policy_name=args.policy_name,
        source_checkpoint=str(checkpoint_path),
        task=args.task,
        phase=args.phase,
        freeze_note=args.freeze_note,
        exported_artifacts=artifact_names,
    )
    _write_json(bundle_dir / "bundle_manifest.json", manifest)

    request_payload["status"] = "completed"
    request_payload["generated_artifacts"] = artifact_names
    request_payload["bundle_dir"] = str(bundle_dir)
    request_payload["note"] = "Export completed for the blind-history runtime contract."
    _write_json(bundle_dir / "export_request.json", request_payload)

    print(f"Exported {args.policy_name} into {bundle_dir}")
    for artifact_name in ["bundle_manifest.json", *artifact_names, "export_request.json"]:
        print(f"- {bundle_dir / artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
