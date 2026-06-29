#!/usr/bin/env python3
"""Run the deployment validation gate for an exported Go2 bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_BUNDLE = REPO_ROOT / "scripts" / "deploy" / "validate_bundle.py"
VALIDATE_INFERENCE_PARITY = REPO_ROOT / "scripts" / "deploy" / "validate_policy_inference_parity.py"
RUN_SIM2SIM = REPO_ROOT / "scripts" / "deploy" / "run_sim2sim.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "deployment_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--policy-name", default="", help="Optional output-name override.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--expected-policy-obs-dim", type=int, default=0)
    parser.add_argument("--expected-history-length", type=int, default=0)
    parser.add_argument("--expected-action-dim", type=int, default=12)
    parser.add_argument("--model-path", default="", help="Optional MuJoCo scene XML path.")
    parser.add_argument("--run-sim2sim", action="store_true", help="Execute the MuJoCo runtime, not just preflight.")
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--trace-steps", type=int, default=25)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _artifact(bundle_dir: Path, manifest: dict, suffix: str) -> Path:
    for name in manifest.get("exported_artifacts", []):
        if str(name).endswith(suffix):
            path = bundle_dir / str(name)
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing artifact ending with {suffix!r} in {bundle_dir}")


def _run_step(name: str, cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "cmd": cmd,
        "stdout_tail": completed.stdout[-4000:] if completed.stdout else "",
        "stderr_tail": completed.stderr[-4000:] if completed.stderr else "",
    }


def _contract_checks(args: argparse.Namespace, metadata: dict, deploy_cfg: dict, manifest: dict) -> list[dict]:
    tensor_contract = metadata["tensor_contract"]
    obs_cfg = deploy_cfg["observations"]
    policy_order = obs_cfg["policy_order"]
    checks = []

    def add(name: str, ok: bool, detail: str, blocking: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "blocking": bool(blocking)})

    policy_dim = int(tensor_contract["policy_obs_dim"])
    history_dim = int(tensor_contract["policy_history_dim"])
    history_len = int(obs_cfg["policy_history_length"])
    action_dim = int(tensor_contract["action_dim"])
    order_dim = sum(int(term["dim"]) for term in policy_order)

    add("policy_kind", manifest.get("policy_kind") == "blind_history_policy", f"found={manifest.get('policy_kind')}")
    add("policy_order_dim_sum", order_dim == policy_dim, f"order_sum={order_dim} policy_dim={policy_dim}")
    add("history_dim_consistent", history_dim == policy_dim * history_len, f"history_dim={history_dim} expected={policy_dim * history_len}")
    add("action_dim", action_dim == args.expected_action_dim, f"found={action_dim} expected={args.expected_action_dim}")

    if args.expected_policy_obs_dim > 0:
        add("expected_policy_obs_dim", policy_dim == args.expected_policy_obs_dim, f"found={policy_dim} expected={args.expected_policy_obs_dim}")
    if args.expected_history_length > 0:
        add("expected_history_length", history_len == args.expected_history_length, f"found={history_len} expected={args.expected_history_length}")

    names = [str(term["name"]) for term in policy_order]
    add("no_base_lin_vel_on_actor", "base_lin_vel" not in names, f"policy_order={names}")
    return checks


def _torchscript_smoke(policy_path: Path, metadata: dict) -> dict:
    tensor_contract = metadata["tensor_contract"]
    policy_dim = int(tensor_contract["policy_obs_dim"])
    history_dim = int(tensor_contract["policy_history_dim"])
    action_dim = int(tensor_contract["action_dim"])
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    with torch.inference_mode():
        output = policy(torch.zeros(1, policy_dim), torch.zeros(1, history_dim))
    return {
        "name": "torchscript_forward_smoke",
        "ok": tuple(output.shape) == (1, action_dim) and bool(torch.isfinite(output).all()),
        "output_shape": list(output.shape),
        "output_abs_max": float(output.abs().max()),
    }


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    metadata = _load_json(_artifact(bundle_dir, manifest, ".export_metadata.json"))
    deploy_cfg = _load_json(_artifact(bundle_dir, manifest, ".deploy_config.json"))
    policy_path = _artifact(bundle_dir, manifest, ".torchscript.pt")

    bundle_name = args.policy_name or bundle_dir.name
    output_dir = (args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()) / bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []
    steps.append(_run_step("bundle_structural_validation", [args.python_exe, str(VALIDATE_BUNDLE), "--bundle-dir", str(bundle_dir)]))
    steps.append(_torchscript_smoke(policy_path, metadata))
    steps.append(
        _run_step(
            "golden_inference_parity",
            [
                args.python_exe,
                str(VALIDATE_INFERENCE_PARITY),
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir / "golden_inference"),
            ],
        )
    )
    sim2sim_cmd = [
        args.python_exe,
        str(RUN_SIM2SIM),
        "--bundle-dir",
        str(bundle_dir),
        "--strict",
        "--command-x",
        str(args.command_x),
        "--command-y",
        str(args.command_y),
        "--command-yaw",
        str(args.command_yaw),
        "--max-steps",
        str(args.max_steps),
        "--trace-steps",
        str(args.trace_steps),
        "--json-out",
        str(output_dir / "sim2sim_report.json"),
    ]
    if args.model_path:
        sim2sim_cmd.extend(["--model-path", args.model_path])
    if args.run_sim2sim:
        sim2sim_cmd.append("--execute-runtime")
    if args.viewer:
        sim2sim_cmd.append("--viewer")
    steps.append(_run_step("sim2sim_gate", sim2sim_cmd))

    contract_checks = _contract_checks(args, metadata, deploy_cfg, manifest)
    all_contract_ok = all(check["ok"] or not check["blocking"] for check in contract_checks)
    all_steps_ok = all(step["ok"] for step in steps)
    status = "pass" if all_contract_ok and all_steps_ok else "blocked"
    blockers = [f"contract:{check['name']}:{check['detail']}" for check in contract_checks if not check["ok"]]
    blockers += [f"step:{step['name']}:returncode={step['returncode']}" for step in steps if not step["ok"]]

    report = {
        "status": status,
        "bundle_dir": str(bundle_dir),
        "bundle_name": bundle_name,
        "tensor_contract": metadata["tensor_contract"],
        "deploy_observation_order": [term["name"] for term in deploy_cfg["observations"]["policy_order"]],
        "contract_checks": contract_checks,
        "steps": steps,
        "blockers": blockers,
        "policy_path": str(policy_path),
        "report_dir": str(output_dir),
    }
    report_path = output_dir / "validation_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
