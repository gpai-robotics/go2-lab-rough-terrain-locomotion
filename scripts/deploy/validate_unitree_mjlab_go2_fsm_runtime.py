#!/usr/bin/env python3
"""Validate the unitree_rl_mjlab Go2 FSM runtime wiring before deployment.

This is a configuration audit only. It does not initialize DDS, publish LowCmd,
or touch the robot. The C++ FSM remains the source of truth for FixStand and
Velocity deployment.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # keep this audit runnable in lean deployment envs
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GO2_DEPLOY_DIR = REPO_ROOT / "reference_repos" / "unitree_rl_mjlab" / "deploy" / "robots" / "go2"
EXPECTED_GO2_POLICY_TO_SDK_MAP = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
EXPECTED_FIXSTAND_KP = [60, 80, 80, 60, 80, 80, 60, 80, 80, 60, 80, 80]
EXPECTED_FIXSTAND_KD = [5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go2-deploy-dir", type=Path, default=DEFAULT_GO2_DEPLOY_DIR)
    parser.add_argument("--config", type=Path, default=None, help="Override config.yaml path.")
    parser.add_argument("--expected-policy-name", default="", help="Optional expected policy_name from bundle manifest.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--strict-fixstand-gains",
        action="store_true",
        help="Require FixStand gains to exactly match the known unitree_rl_mjlab Go2 reference.",
    )
    return parser.parse_args()


def _check(name: str, ok: bool, detail: str, *, blocking: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, "blocking": bool(blocking)}


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return _load_yaml_subset(path)
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise SystemExit(f"YAML root is not a mapping: {path}")
    return data


def _literal_list(text: str) -> list[Any]:
    text = text.strip().rstrip(",")
    return ast.literal_eval(text)


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _extract_inline_or_bracket_list(lines: list[str], key: str) -> list[Any]:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*)$")
    for index, line in enumerate(lines):
        match = pattern.match(_strip_comment(line))
        if not match:
            continue
        value = match.group(1).strip()
        if value.startswith("[") and value.count("[") == value.count("]"):
            return _literal_list(value)
        collected = [value] if value else []
        balance = value.count("[") - value.count("]")
        for extra in lines[index + 1 :]:
            stripped = _strip_comment(extra).strip()
            if not stripped:
                continue
            collected.append(stripped)
            balance += stripped.count("[") - stripped.count("]")
            if balance <= 0 and "]" in stripped:
                break
        return _literal_list(" ".join(collected))
    return []


def _extract_block_scalar_list(lines: list[str], key: str) -> list[Any]:
    pattern = re.compile(rf"^({re.escape(key)}):\s*(.*)$")
    for index, line in enumerate(lines):
        stripped_line = _strip_comment(line)
        match = pattern.match(stripped_line)
        if not match:
            continue
        inline = match.group(2).strip()
        if inline.startswith("["):
            return _literal_list(inline)
        values: list[Any] = []
        for extra in lines[index + 1 :]:
            if extra and not extra.startswith(" "):
                break
            stripped = _strip_comment(extra).strip()
            if not stripped:
                continue
            if not stripped.startswith("-"):
                break
            values.append(ast.literal_eval(stripped[1:].strip()))
        return values
    return []


def _extract_velocity_policy_dir(lines: list[str]) -> str:
    in_velocity = False
    for line in lines:
        stripped = _strip_comment(line).strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            in_velocity = stripped == "Velocity:"
        if in_velocity and stripped.startswith("policy_dir:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _extract_top_level_child_block(lines: list[str], child_name: str) -> list[str]:
    block: list[str] = []
    in_block = False
    child_header = f"  {child_name}:"
    for line in lines:
        stripped = _strip_comment(line).strip()
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            if in_block:
                break
            in_block = line.strip() == child_header.strip()
        if in_block:
            block.append(line)
    return block


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    text = "\n".join(lines)
    if "FSM:" in text:
        fixstand_lines = _extract_top_level_child_block(lines, "FixStand")
        return {
            "FSM": {
                "FixStand": {
                    "kp": _extract_inline_or_bracket_list(fixstand_lines, "kp"),
                    "kd": _extract_inline_or_bracket_list(fixstand_lines, "kd"),
                    "ts": _extract_inline_or_bracket_list(fixstand_lines, "ts"),
                    "qs": _extract_inline_or_bracket_list(fixstand_lines, "qs"),
                },
                "Velocity": {
                    "policy_dir": _extract_velocity_policy_dir(lines),
                },
            }
        }
    return {
        "joint_ids_map": _extract_block_scalar_list(lines, "joint_ids_map"),
        "stiffness": _extract_block_scalar_list(lines, "stiffness"),
        "damping": _extract_block_scalar_list(lines, "damping"),
        "default_joint_pos": _extract_block_scalar_list(lines, "default_joint_pos"),
        "step_dt": (_extract_block_scalar_list(lines, "step_dt") or [float(_extract_top_level_scalar(lines, "step_dt", "-1.0"))])[0],
    }


def _extract_top_level_scalar(lines: list[str], key: str, default: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$")
    for line in lines:
        match = pattern.match(_strip_comment(line))
        if match:
            return match.group(1).strip() or default
    return default


def main() -> int:
    args = parse_args()
    go2_deploy_dir = args.go2_deploy_dir.resolve()
    config_path = (args.config or (go2_deploy_dir / "config" / "config.yaml")).resolve()
    checks: list[dict[str, Any]] = []

    config_exists = config_path.exists()
    checks.append(_check("config_yaml_exists", config_exists, str(config_path)))
    if not config_exists:
        report = {"status": "fail", "checks": checks}
        print(json.dumps(report, indent=2))
        return 1

    config = _load_yaml(config_path)
    fsm = config.get("FSM", {})
    fixstand = fsm.get("FixStand", {}) if isinstance(fsm, dict) else {}
    velocity = fsm.get("Velocity", {}) if isinstance(fsm, dict) else {}

    checks.append(_check("fixstand_state_present", isinstance(fixstand, dict) and bool(fixstand), "FSM.FixStand"))
    checks.append(_check("velocity_state_present", isinstance(velocity, dict) and bool(velocity), "FSM.Velocity"))

    kp = _as_float_list(fixstand.get("kp"))
    kd = _as_float_list(fixstand.get("kd"))
    ts = _as_float_list(fixstand.get("ts"))
    qs = fixstand.get("qs", [])
    checks.append(_check("fixstand_kp_dim", len(kp) == 12, f"len={len(kp)} values={kp}"))
    checks.append(_check("fixstand_kd_dim", len(kd) == 12, f"len={len(kd)} values={kd}"))
    checks.append(_check("fixstand_ts_qs_dim", isinstance(qs, list) and len(ts) == len(qs) and len(ts) >= 2, f"len(ts)={len(ts)} len(qs)={len(qs) if isinstance(qs, list) else 'not-list'}"))
    final_q = _as_float_list(qs[-1]) if isinstance(qs, list) and qs else []
    checks.append(_check("fixstand_final_pose_dim", len(final_q) == 12, f"len={len(final_q)} values={final_q}"))

    if args.strict_fixstand_gains:
        checks.append(_check("fixstand_kp_reference", [round(v, 6) for v in kp] == EXPECTED_FIXSTAND_KP, f"found={kp} expected={EXPECTED_FIXSTAND_KP}"))
        checks.append(_check("fixstand_kd_reference", [round(v, 6) for v in kd] == EXPECTED_FIXSTAND_KD, f"found={kd} expected={EXPECTED_FIXSTAND_KD}"))
    else:
        checks.append(_check("fixstand_kp_reasonable", bool(kp) and min(kp) >= 20.0 and max(kp) <= 120.0, f"min={min(kp) if kp else None} max={max(kp) if kp else None}", blocking=False))
        checks.append(_check("fixstand_kd_reasonable", bool(kd) and min(kd) >= 1.0 and max(kd) <= 10.0, f"min={min(kd) if kd else None} max={max(kd) if kd else None}", blocking=False))

    policy_dir_value = velocity.get("policy_dir") if isinstance(velocity, dict) else None
    policy_dir_ok = isinstance(policy_dir_value, str) and bool(policy_dir_value.strip())
    checks.append(_check("velocity_policy_dir_set", policy_dir_ok, str(policy_dir_value)))
    policy_dir = (go2_deploy_dir / policy_dir_value).resolve() if policy_dir_ok else go2_deploy_dir / "__missing__"
    params_dir = policy_dir / "params"
    exported_dir = policy_dir / "exported"
    deploy_yaml = params_dir / "deploy.yaml"
    manifest_path = params_dir / "bundle_manifest.json"
    onnx_path = exported_dir / "policy.onnx"
    checks.append(_check("policy_dir_exists", policy_dir.exists(), str(policy_dir)))
    checks.append(_check("runtime_deploy_yaml_exists", deploy_yaml.exists(), str(deploy_yaml)))
    checks.append(_check("runtime_policy_onnx_exists", onnx_path.exists(), str(onnx_path)))
    checks.append(_check("runtime_manifest_exists", manifest_path.exists(), str(manifest_path)))

    deploy_cfg: dict[str, Any] = {}
    if deploy_yaml.exists():
        deploy_cfg = _load_yaml(deploy_yaml)
        joint_ids_map = _as_int_list(deploy_cfg.get("joint_ids_map"))
        checks.append(_check("runtime_joint_ids_map_dim", len(joint_ids_map) == 12, f"len={len(joint_ids_map)} values={joint_ids_map}"))
        checks.append(_check("runtime_joint_ids_map_expected", joint_ids_map == EXPECTED_GO2_POLICY_TO_SDK_MAP, f"found={joint_ids_map} expected={EXPECTED_GO2_POLICY_TO_SDK_MAP}"))
        stiffness = _as_float_list(deploy_cfg.get("stiffness"))
        damping = _as_float_list(deploy_cfg.get("damping"))
        default_joint_pos = _as_float_list(deploy_cfg.get("default_joint_pos"))
        checks.append(_check("runtime_stiffness_dim", len(stiffness) == 12, f"len={len(stiffness)} values={stiffness}"))
        checks.append(_check("runtime_damping_dim", len(damping) == 12, f"len={len(damping)} values={damping}"))
        checks.append(_check("runtime_default_joint_pos_dim", len(default_joint_pos) == 12, f"len={len(default_joint_pos)} values={default_joint_pos}"))
        step_dt = float(deploy_cfg.get("step_dt", -1.0))
        checks.append(_check("runtime_step_dt_50hz", abs(step_dt - 0.02) < 1e-6, f"step_dt={step_dt}"))

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        policy_name = str(manifest.get("policy_name", ""))
        expected_policy_name = args.expected_policy_name.strip()
        checks.append(_check("manifest_policy_name_present", bool(policy_name), policy_name))
        if expected_policy_name:
            checks.append(_check("manifest_policy_name_expected", policy_name == expected_policy_name, f"found={policy_name} expected={expected_policy_name}"))

    blocking_failures = [check for check in checks if check["blocking"] and not check["ok"]]
    review_failures = [check for check in checks if not check["blocking"] and not check["ok"]]
    status = "fail" if blocking_failures else ("review" if review_failures else "pass")
    report = {
        "status": status,
        "go2_deploy_dir": str(go2_deploy_dir),
        "config": str(config_path),
        "policy_dir": str(policy_dir),
        "fixstand": {
            "kp": kp,
            "kd": kd,
            "ts": ts,
            "final_q": final_q,
        },
        "checks": checks,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"status: {status}")
    print(f"config: {config_path}")
    print(f"policy_dir: {policy_dir}")
    print(f"fixstand_final_q: {final_q}")
    for check in checks:
        label = "PASS" if check["ok"] else ("FAIL" if check["blocking"] else "REVIEW")
        print(f"  [{label}] {check['name']}: {check['detail']}")
    if args.json_out is not None:
        print(f"report: {args.json_out}")
    return 0 if status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
