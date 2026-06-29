#!/usr/bin/env python3
"""Validate the local exported blind-history deployment bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2_deploy_contract import (
    ACTION_DIM,
    BLIND_HISTORY_OBSERVATION_GROUPS,
    BLIND_HISTORY_POLICY_KIND,
    HISTORY_LAYOUT,
    policy_obs_dim,
)
from history_layout import resolve_history_layout


REQUIRED_MANIFEST_FIELDS = [
    "policy_name",
    "source_checkpoint",
    "task",
    "phase",
    "policy_kind",
    "deployable_observation_groups",
    "control_rate_hz",
    "history_layout",
    "exported_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, help="Bundle directory produced by export_policy.py.")
    parser.add_argument(
        "--allow-missing-source-checkpoint",
        action="store_true",
        help="Skip the source-checkpoint existence check for moved/shared bundles.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        raise SystemExit(f"Failed to parse JSON file: {path}") from exc


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    export_request_path = bundle_dir / "export_request.json"

    _expect(manifest_path.exists(), f"Missing bundle manifest: {manifest_path}")
    _expect(export_request_path.exists(), f"Missing export request: {export_request_path}")

    manifest = _read_json(manifest_path)
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    _expect(not missing, f"Manifest missing required fields: {', '.join(missing)}")

    _expect(
        manifest["policy_kind"] == BLIND_HISTORY_POLICY_KIND,
        f"Unsupported policy kind in manifest: {manifest['policy_kind']!r}",
    )
    _expect(
        list(manifest["deployable_observation_groups"]) == BLIND_HISTORY_OBSERVATION_GROUPS,
        "Manifest deployable observation groups do not match the blind-history contract.",
    )
    _expect(
        str(manifest["history_layout"]) == HISTORY_LAYOUT,
        f"Manifest history layout must be {HISTORY_LAYOUT!r}.",
    )

    source_checkpoint = Path(manifest["source_checkpoint"]).expanduser()
    if not args.allow_missing_source_checkpoint:
        _expect(source_checkpoint.exists(), f"Missing source checkpoint: {source_checkpoint}")

    artifacts = list(manifest.get("exported_artifacts", []))
    _expect(artifacts, "Manifest exported_artifacts is empty.")
    for artifact in artifacts:
        artifact_path = bundle_dir / artifact
        _expect(artifact_path.exists(), f"Missing exported artifact: {artifact_path}")

    model_artifacts = [name for name in artifacts if name.endswith(".torchscript.pt") or name.endswith(".onnx")]
    _expect(model_artifacts, "Bundle must include at least one exported model artifact.")

    metadata_candidates = [bundle_dir / name for name in artifacts if name.endswith(".export_metadata.json")]
    config_candidates = [bundle_dir / name for name in artifacts if name.endswith(".deploy_config.json")]
    yaml_candidates = [bundle_dir / name for name in artifacts if name.endswith(".deploy.yaml")]
    _expect(metadata_candidates, "Bundle is missing *.export_metadata.json.")
    _expect(config_candidates, "Bundle is missing *.deploy_config.json.")
    _expect(yaml_candidates, "Bundle is missing *.deploy.yaml.")

    metadata = _read_json(metadata_candidates[0])
    deploy_config = _read_json(config_candidates[0])

    tensor_contract = metadata.get("tensor_contract", {})
    observations = deploy_config.get("observations", {})
    control = deploy_config.get("control", {})
    actions = deploy_config.get("actions", {})

    expected_policy_dim = policy_obs_dim()
    expected_history_length = int(observations.get("policy_history_length", 0))
    expected_history_dim = expected_policy_dim * expected_history_length

    _expect(
        int(tensor_contract.get("policy_obs_dim", -1)) == expected_policy_dim,
        f"Tensor contract policy_obs_dim must be {expected_policy_dim}.",
    )
    _expect(
        int(tensor_contract.get("policy_history_dim", -1)) == expected_history_dim,
        f"Tensor contract policy_history_dim must be {expected_history_dim}.",
    )
    _expect(
        int(tensor_contract.get("history_length", -1)) == expected_history_length,
        "Tensor contract history_length does not match the deploy config.",
    )
    _expect(
        int(tensor_contract.get("action_dim", -1)) == ACTION_DIM,
        f"Tensor contract action_dim must be {ACTION_DIM}.",
    )
    _expect(
        int(observations.get("policy_dim", -1)) == expected_policy_dim,
        f"Deploy config policy_dim must be {expected_policy_dim}.",
    )
    _expect(
        int(observations.get("policy_history_dim", -1)) == expected_history_dim,
        "Deploy config policy_history_dim is inconsistent with policy_dim * history_length.",
    )
    _expect(
        resolve_history_layout(observations) == HISTORY_LAYOUT,
        "Deploy config history layout does not match the expected IsaacLab term-major layout.",
    )
    _expect(
        int(len(actions.get("joint_names", []))) == ACTION_DIM,
        f"Deploy config must expose {ACTION_DIM} joint action targets.",
    )
    _expect(
        abs(float(control.get("step_dt", -1.0)) - 0.02) < 1e-9,
        "Deploy config step_dt must remain 0.02 seconds.",
    )

    print(f"Bundle is structurally valid: {bundle_dir}")
    print(f"- policy_obs_dim={expected_policy_dim}")
    print(f"- history_length={expected_history_length}")
    print(f"- policy_history_dim={expected_history_dim}")
    print(f"- model_artifacts={len(model_artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
