#!/usr/bin/env python3
"""Generate golden policy vectors and validate checkpoint/export parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from deploy_blind_history_policy import build_deployable_module, import_torch, load_checkpoint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/deployment_validation/golden_inference")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--skip-cpp", action="store_true", help="Skip optional C++ ONNX parity.")
    parser.add_argument(
        "--onnx-validator-source",
        default="",
        help="Optional path to a C++ ONNX validator source file compatible with this bundle contract.",
    )
    parser.add_argument(
        "--onnxruntime-root",
        default="",
        help="Optional ONNX Runtime SDK root for compiling the C++ validator.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.9g")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def find_artifact(bundle_dir: Path, manifest: dict, suffix: str) -> Path:
    for name in manifest.get("exported_artifacts", []):
        if str(name).endswith(suffix):
            candidate = bundle_dir / str(name)
            if candidate.exists():
                return candidate
    raise SystemExit(f"Could not find artifact ending with {suffix!r} in {bundle_dir}")


def generate_inputs(
    rng: np.random.Generator,
    num_cases: int,
    policy_dim: int,
    history_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    policy_obs = rng.normal(0.0, 0.35, size=(num_cases, policy_dim)).astype(np.float32)
    policy_history = rng.normal(0.0, 0.30, size=(num_cases, history_dim)).astype(np.float32)

    policy_obs[0] = 0.0
    policy_history[0] = 0.0
    if num_cases > 1:
        policy_obs[1] = np.linspace(-1.0, 1.0, policy_dim, dtype=np.float32)
        policy_history[1] = np.tile(policy_obs[1], history_dim // policy_dim)
    if num_cases > 2:
        policy_obs[2, 3:6] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        policy_obs[2, 6:9] = np.array([0.5, -0.3, 0.6], dtype=np.float32)
    return policy_obs, policy_history


def run_cpp_parity(
    *,
    output_dir: Path,
    onnx_path: Path,
    obs_path: Path,
    history_path: Path,
    expected_path: Path,
    tolerance: float,
    validator_source: Path | None,
    onnxruntime_root: Path | None,
) -> dict[str, object]:
    compiler = shutil.which("g++")
    if compiler is None:
        return {"status": "blocked", "reason": "g++ was not found in PATH."}
    if validator_source is None or onnxruntime_root is None:
        return {
            "status": "blocked",
            "reason": "C++ parity is optional. Pass --onnx-validator-source and --onnxruntime-root to enable it.",
        }

    ort_include = onnxruntime_root / "include"
    ort_lib_dir = onnxruntime_root / "lib"
    so_candidates = sorted(ort_lib_dir.glob("libonnxruntime.so*"))
    if not validator_source.exists():
        return {"status": "blocked", "reason": f"Missing validator source: {validator_source}"}
    if not ort_include.exists() or not ort_lib_dir.exists() or not so_candidates:
        return {"status": "blocked", "reason": "ONNX Runtime headers or libraries were not found."}

    chosen_lib = so_candidates[0]
    executable_path = output_dir / "validate_onnx_bundle"
    compile_cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        str(validator_source),
        f"-I{ort_include}",
        f"-L{ort_lib_dir}",
        f"-Wl,-rpath,{ort_lib_dir}",
        f"-l:{chosen_lib.name}",
        "-o",
        str(executable_path),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        return {
            "status": "fail",
            "stage": "compile",
            "returncode": compile_result.returncode,
            "stdout": compile_result.stdout,
            "stderr": compile_result.stderr,
        }

    run_cmd = [
        str(executable_path),
        str(onnx_path),
        str(obs_path),
        str(history_path),
        str(expected_path),
        str(tolerance),
    ]
    run_result = subprocess.run(run_cmd, capture_output=True, text=True)
    return {
        "status": "pass" if run_result.returncode == 0 else "fail",
        "stage": "run",
        "returncode": run_result.returncode,
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(bundle_dir / "bundle_manifest.json")
    metadata = load_json(find_artifact(bundle_dir, manifest, ".export_metadata.json"))
    deploy_cfg = load_json(find_artifact(bundle_dir, manifest, ".deploy_config.json"))
    policy_path = find_artifact(bundle_dir, manifest, ".torchscript.pt")
    onnx_path = None
    try:
        onnx_path = find_artifact(bundle_dir, manifest, ".onnx")
    except SystemExit:
        onnx_path = None

    tensor_contract = metadata["tensor_contract"]
    policy_dim = int(tensor_contract["policy_obs_dim"])
    history_dim = int(tensor_contract["policy_history_dim"])
    history_length = int(tensor_contract["history_length"])
    action_dim = int(tensor_contract["action_dim"])

    policy_order_dim = sum(int(term["dim"]) for term in deploy_cfg["observations"]["policy_order"])
    if policy_order_dim != policy_dim:
        raise SystemExit(f"Deploy config policy_order sums to {policy_order_dim}, expected {policy_dim}.")

    torch, _ = import_torch()
    source_policy = build_deployable_module(
        state_dict=load_checkpoint_state(Path(manifest["source_checkpoint"]).expanduser()),
        policy_obs_dim=policy_dim,
        policy_history_length=history_length,
    )
    source_policy.eval()
    export_policy = torch.jit.load(str(policy_path), map_location="cpu")
    export_policy.eval()

    rng = np.random.default_rng(args.seed)
    policy_obs_np, policy_history_np = generate_inputs(rng, args.num_cases, policy_dim, history_dim)
    policy_obs = torch.from_numpy(policy_obs_np)
    policy_history = torch.from_numpy(policy_history_np)
    with torch.inference_mode():
        expected_actions = source_policy(policy_obs, policy_history)
        exported_actions = export_policy(policy_obs, policy_history)
    abs_diff = (expected_actions - exported_actions).abs()
    max_abs_error = float(abs_diff.max().item())
    mean_abs_error = float(abs_diff.mean().item())
    passed = bool(max_abs_error <= args.tolerance)

    obs_path = output_dir / "policy_obs.txt"
    history_path = output_dir / "policy_history.txt"
    expected_path = output_dir / "expected_actions.txt"
    exported_path = output_dir / "exported_actions.txt"
    write_rows(obs_path, policy_obs_np)
    write_rows(history_path, policy_history_np)
    write_rows(expected_path, expected_actions.cpu().numpy())
    write_rows(exported_path, exported_actions.cpu().numpy())

    cpp_result = None
    if onnx_path is not None and not args.skip_cpp:
        validator_source = Path(args.onnx_validator_source).expanduser().resolve() if args.onnx_validator_source else None
        onnxruntime_root = Path(args.onnxruntime_root).expanduser().resolve() if args.onnxruntime_root else None
        cpp_result = run_cpp_parity(
            output_dir=output_dir,
            onnx_path=onnx_path,
            obs_path=obs_path,
            history_path=history_path,
            expected_path=expected_path,
            tolerance=args.tolerance,
            validator_source=validator_source,
            onnxruntime_root=onnxruntime_root,
        )

    report = {
        "status": "pass" if passed else "fail",
        "bundle_dir": str(bundle_dir),
        "policy_path": str(policy_path),
        "onnx_path": str(onnx_path) if onnx_path is not None else None,
        "tensor_contract": tensor_contract,
        "policy_history_length": history_length,
        "action_dim": action_dim,
        "num_cases": args.num_cases,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "expected_sha256": sha256(expected_path),
        "exported_sha256": sha256(exported_path),
        "cpp_parity": cpp_result,
        "artifacts": {
            "policy_obs": str(obs_path),
            "policy_history": str(history_path),
            "expected_actions": str(expected_path),
            "exported_actions": str(exported_path),
        },
    }
    report_path = output_dir / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
