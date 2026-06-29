#!/usr/bin/env python3
"""MuJoCo sim2sim preflight and optional runtime execution for exported bundles."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from mujoco_runtime import BridgeConfig, Go2MujocoDeployBridge


DEFAULT_GO2_MODEL_ENV = "GO2_MUJOCO_MODEL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--model-path", default="", help=f"MuJoCo scene XML. Defaults to ${DEFAULT_GO2_MODEL_ENV} if set.")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if blockers are found.")
    parser.add_argument("--execute-runtime", action="store_true", help="Run the MuJoCo bridge instead of preflight only.")
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--trace-steps", type=int, default=25)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--real-time-factor", type=float, default=1.0)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _artifact(bundle_dir: Path, manifest: dict, suffix: str) -> Path | None:
    for name in manifest.get("exported_artifacts", []):
        if str(name).endswith(suffix):
            candidate = bundle_dir / str(name)
            if candidate.exists():
                return candidate
    return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    report = {
        "status": "blocked",
        "bundle_dir": str(bundle_dir),
        "execute_runtime": bool(args.execute_runtime),
        "blockers": [],
    }
    if not manifest_path.exists():
        report["blockers"].append(f"missing_manifest:{manifest_path}")
        print(json.dumps(report, indent=2))
        return 2 if args.strict else 0

    manifest = _load_json(manifest_path)
    metadata_path = _artifact(bundle_dir, manifest, ".export_metadata.json")
    deploy_cfg_path = _artifact(bundle_dir, manifest, ".deploy_config.json")
    policy_path = _artifact(bundle_dir, manifest, ".torchscript.pt")
    if metadata_path is None:
        report["blockers"].append("missing_export_metadata")
    if deploy_cfg_path is None:
        report["blockers"].append("missing_deploy_config")
    if policy_path is None:
        report["blockers"].append("missing_torchscript")

    model_path_raw = args.model_path or ""
    if not model_path_raw:
        import os

        model_path_raw = os.environ.get(DEFAULT_GO2_MODEL_ENV, "")
    model_path = Path(model_path_raw).expanduser().resolve() if model_path_raw else None
    if model_path is None:
        report["blockers"].append(
            f"missing_model_path:set --model-path or environment variable {DEFAULT_GO2_MODEL_ENV}"
        )
    elif not model_path.exists():
        report["blockers"].append(f"missing_model_xml:{model_path}")

    if not _module_available("mujoco"):
        report["blockers"].append("missing_python_module:mujoco")
    if not _module_available("torch"):
        report["blockers"].append("missing_python_module:torch")

    if metadata_path is not None and deploy_cfg_path is not None:
        metadata = _load_json(metadata_path)
        deploy_cfg = _load_json(deploy_cfg_path)
        report["tensor_contract"] = metadata.get("tensor_contract", {})
        report["deploy_observation_order"] = [term["name"] for term in deploy_cfg["observations"]["policy_order"]]
        report["policy_kind"] = manifest.get("policy_kind")
        report["expected_step_dt"] = deploy_cfg.get("control", {}).get("step_dt")

    report["policy_path"] = str(policy_path) if policy_path is not None else None
    report["deploy_config_path"] = str(deploy_cfg_path) if deploy_cfg_path is not None else None
    report["model_path"] = str(model_path) if model_path is not None else None

    if report["blockers"]:
        report["status"] = "blocked"
    elif not args.execute_runtime:
        report["status"] = "ready"
    else:
        cfg = BridgeConfig(
            model_path=model_path,
            policy_artifact_path=policy_path,
            deploy_config_path=deploy_cfg_path,
            control_dt=float(deploy_cfg["control"]["step_dt"]),
            physics_dt=float(deploy_cfg["control"].get("physics_dt", 0.005)),
            command_x=args.command_x,
            command_y=args.command_y,
            command_yaw=args.command_yaw,
            max_steps=args.max_steps,
            trace_steps=args.trace_steps,
            viewer=args.viewer,
            viewer_dt=args.viewer_dt,
            real_time_factor=args.real_time_factor,
        )
        runtime = Go2MujocoDeployBridge(cfg)
        runtime_report = runtime.run()
        report["runtime"] = runtime_report
        report["status"] = "pass" if runtime_report.get("status") == "complete" else "fail"

    if args.json_out:
        Path(args.json_out).expanduser().resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] in {"blocked", "fail"} and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
