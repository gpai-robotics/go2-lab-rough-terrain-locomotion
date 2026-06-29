#!/usr/bin/env python3
"""Stage a frozen Go2 bundle into the unitree_rl_mjlab C++ runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from materialize_unitree_rl_lab_layout import materialize_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_FSM_RUNTIME = REPO_ROOT / "scripts" / "deploy" / "validate_unitree_mjlab_go2_fsm_runtime.py"
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "artifacts"
    / "exported"
    / "go2_blind_rough_asymppo_mjlab_v1_candidate"
)
DEFAULT_GO2_DEPLOY_DIR = (
    REPO_ROOT / "reference_repos" / "unitree_rl_mjlab" / "deploy" / "robots" / "go2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--go2-deploy-dir", default=str(DEFAULT_GO2_DEPLOY_DIR))
    parser.add_argument(
        "--runtime-name",
        default=None,
        help="Policy directory name. Defaults to the bundle policy_name.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Point Velocity.policy_dir directly at the staged runtime.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the post-stage FSM/runtime config audit.",
    )
    parser.add_argument(
        "--strict-fixstand-gains",
        action="store_true",
        help="Require FixStand gains to match the known unitree_rl_mjlab Go2 reference.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _policy_name(bundle_dir: Path) -> str:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    return str(manifest["policy_name"])


def _activate_runtime(config_path: Path, relative_policy_dir: str) -> None:
    lines = config_path.read_text().splitlines()
    output: list[str] = []
    in_velocity = False
    replaced = False
    for line in lines:
        stripped = line.strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            in_velocity = stripped == "Velocity:"
        if in_velocity and stripped.startswith("policy_dir:"):
            indent = line[: len(line) - len(line.lstrip())]
            output.append(f"{indent}policy_dir: {relative_policy_dir}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        raise RuntimeError(f"Could not find Velocity.policy_dir in {config_path}")
    config_path.write_text("\n".join(output) + "\n")


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    go2_deploy_dir = Path(args.go2_deploy_dir).resolve()
    bundle_policy_name = _policy_name(bundle_dir)
    runtime_name = args.runtime_name or bundle_policy_name
    output_dir = go2_deploy_dir / "config" / "policy" / "velocity" / runtime_name

    materialize_bundle(bundle_dir, output_dir, robot="go2", force=args.force)

    relative_policy_dir = f"config/policy/velocity/{runtime_name}"
    if args.activate:
        _activate_runtime(go2_deploy_dir / "config" / "config.yaml", relative_policy_dir)

    if not args.skip_validate:
        validate_cmd = [
            sys.executable,
            str(VALIDATE_FSM_RUNTIME),
            "--go2-deploy-dir",
            str(go2_deploy_dir),
            "--expected-policy-name",
            bundle_policy_name,
            "--json-out",
            str(REPO_ROOT / "artifacts" / "deployment_validation" / runtime_name / "unitree_mjlab_fsm_runtime_audit.json"),
        ]
        if args.strict_fixstand_gains:
            validate_cmd.append("--strict-fixstand-gains")
        subprocess.run(validate_cmd, check=True)

    print(f"Staged runtime: {output_dir}")
    print(f"Policy directory: {relative_policy_dir}")
    print(f"Activated: {'yes' if args.activate else 'no'}")
    print(f"Validated: {'no' if args.skip_validate else 'yes'}")
    print("Simulation controller:")
    print(f"  {go2_deploy_dir / 'build' / 'go2_ctrl'} --network=lo")
    print("Hardware controller:")
    print(f"  {go2_deploy_dir / 'build' / 'go2_ctrl'} --network=<ethernet-interface>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
