#!/usr/bin/env python3
"""Materialize a frozen bundle into a unitree_rl_lab-style deploy layout.

Output layout:

<output-dir>/
  exported/
    policy.onnx
  params/
    deploy.yaml
    bundle_manifest.json
    export_metadata.json
    deploy_config.json

This is a compatibility bridge only. The bundle remains the source of truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


GO2_JOINT_SDK_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--robot",
        default="go2",
        choices=["go2"],
        help="Robot family for runtime layout conversion.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def _find_artifact(bundle_dir: Path, manifest: dict, suffix: str) -> Path | None:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(suffix):
            path = bundle_dir / artifact
            if path.exists():
                return path
    return None


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "[", "]", "{", "}", ",", "'"]):
            return json.dumps(value)
        return value
    return str(value)


def _yaml_dump(value: object, indent: int = 0) -> list[str]:
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


def _go2_joint_ids_map(policy_joint_names: list[str]) -> list[int]:
    sdk_index = {name: idx for idx, name in enumerate(GO2_JOINT_SDK_NAMES)}
    try:
        return [sdk_index[name] for name in policy_joint_names]
    except KeyError as exc:
        raise SystemExit(f"Unsupported Go2 joint name in policy bundle: {exc.args[0]}") from exc


def _convert_deploy_config_to_runtime_yaml(deploy_cfg: dict, robot: str) -> dict:
    if robot != "go2":
        raise SystemExit(f"Unsupported robot for runtime deploy conversion: {robot}")

    actions = deploy_cfg["actions"]
    observations = deploy_cfg["observations"]
    robot_cfg = deploy_cfg["robot"]

    policy_order = observations["policy_order"]
    history_length = int(observations.get("policy_history_length", 0))
    use_gym_history = bool(observations.get("use_gym_history", False))

    # Our frozen Go2 bundles store policy joint order, while unitree_rl_lab expects
    # `joint_ids_map` to point into Unitree SDK motor order.
    joint_ids_map = _go2_joint_ids_map(actions["joint_names"])

    def make_obs_group(term_history_length: int, use_gym_history: bool = False) -> dict:
        group: dict[str, object] = {}
        if use_gym_history:
            group["use_gym_history"] = True
        for term in policy_order:
            group[term["name"]] = {
                "params": {},
                "clip": None,
                "scale": term["scale"],
                "history_length": term_history_length,
            }
        return group

    runtime_cfg = {
        "joint_ids_map": joint_ids_map,
        "step_dt": float(deploy_cfg["control"]["step_dt"]),
        "stiffness": robot_cfg["joint_stiffness"],
        "damping": robot_cfg["joint_damping"],
        "default_joint_pos": robot_cfg["default_joint_pos"],
        "commands": {
            "base_velocity": {
                "default": deploy_cfg["commands"]["base_velocity"]["default"],
                "ranges": deploy_cfg["commands"]["base_velocity"].get(
                    "ranges",
                    {
                        "lin_vel_x": [0.0, 1.0],
                        "lin_vel_y": [0.0, 0.0],
                        "ang_vel_z": [0.0, 0.0],
                    },
                ),
            }
        },
        "actions": {
            actions["type"]: {
                "clip": actions["clip"],
                "joint_names": actions["joint_names"],
                "joint_ids": actions["joint_ids"],
                "scale": actions["scale"],
                "offset": actions["offset"],
            }
        },
        "observations": {
            "policy_obs": make_obs_group(term_history_length=1),
            "policy_history": make_obs_group(
                term_history_length=history_length,
                use_gym_history=use_gym_history,
            ),
        },
    }
    return runtime_cfg


def materialize_bundle(bundle_dir: Path, output_dir: Path, robot: str = "go2", force: bool = False) -> None:
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    onnx_path = _find_artifact(bundle_dir, manifest, ".onnx")
    yaml_path = _find_artifact(bundle_dir, manifest, ".deploy.yaml")
    deploy_cfg_path = _find_artifact(bundle_dir, manifest, ".deploy_config.json")
    export_metadata_path = _find_artifact(bundle_dir, manifest, ".export_metadata.json")

    if onnx_path is None:
        raise SystemExit(
            "Bundle does not contain an ONNX artifact. Re-export with:\n"
            "  --format torchscript --format onnx"
        )
    if yaml_path is None and deploy_cfg_path is None:
        raise SystemExit("Bundle does not contain a unitree_rl_lab-compatible deploy artifact.")

    if output_dir.exists():
        if not force:
            raise SystemExit(f"Output directory already exists: {output_dir}\nUse --force to replace it.")
        shutil.rmtree(output_dir)

    exported_dir = output_dir / "exported"
    params_dir = output_dir / "params"
    exported_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, exported_dir / "policy.onnx")
    if deploy_cfg_path is not None:
        deploy_cfg = json.loads(deploy_cfg_path.read_text())
        runtime_cfg = _convert_deploy_config_to_runtime_yaml(deploy_cfg, robot=robot)
        (params_dir / "deploy.yaml").write_text("\n".join(_yaml_dump(runtime_cfg)) + "\n")
    else:
        shutil.copy2(yaml_path, params_dir / "deploy.yaml")
    shutil.copy2(manifest_path, params_dir / "bundle_manifest.json")
    if yaml_path is not None:
        shutil.copy2(yaml_path, params_dir / "bundle_compat.deploy.yaml")
    if deploy_cfg_path is not None:
        shutil.copy2(deploy_cfg_path, params_dir / "deploy_config.json")
    if export_metadata_path is not None:
        shutil.copy2(export_metadata_path, params_dir / "export_metadata.json")


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    output_dir = Path(args.output_dir)
    materialize_bundle(bundle_dir, output_dir, robot=args.robot, force=args.force)

    print(f"Materialized unitree_rl_lab-compatible deploy layout at: {output_dir}")
    print(f"- {output_dir / 'exported' / 'policy.onnx'}")
    print(f"- {output_dir / 'params' / 'deploy.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
