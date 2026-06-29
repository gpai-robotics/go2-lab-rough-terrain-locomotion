#!/usr/bin/env python3
"""Preflight checks for the Go2 rough-terrain IsaacLab setup."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def main() -> None:
    print("[INFO] Python:", sys.executable)
    print("[INFO] CWD:", Path.cwd())

    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/opt/IsaacLab")).expanduser()
    if not (isaaclab_root / "isaaclab.sh").is_file():
        _fail(f"ISAACLAB_ROOT does not point to IsaacLab: {isaaclab_root}")
    _ok(f"ISAACLAB_ROOT={isaaclab_root}")

    try:
        import torch
    except ImportError as exc:
        detail = str(exc)
        if "libcusparseLt" in detail:
            _fail(
                "torch cannot find libcusparseLt.so.0. Launch through "
                "`bash scripts/isaaclab_user.sh ...`, not IsaacLab/isaaclab.sh directly."
            )
        _fail(f"torch import failed: {exc}")

    _ok(f"torch={torch.__version__} from {Path(torch.__file__).resolve()}")

    user_site = Path(site.getusersitepackages()).resolve()
    user_torch = user_site / "torch"
    if user_torch.exists():
        _warn(
            f"user-site torch exists at {user_torch}. If imports fail with NCCL/CUDA symbol errors, "
            "remove user-site torch/nvidia packages and reinstall this repo with `--no-deps`."
        )

    try:
        import gymnasium as gym
        import go2_rough  # noqa: F401
        from go2_rough.envs.asset_contract import bundled_go2_usd_path, go2_usd_path
    except Exception as exc:
        _fail(f"go2_rough import failed: {exc}")

    for task_id in ("Go2-Flat-MJLAB-Prior-V1", "Go2-Blind-Rough-MJLAB-AsymPPO-V1"):
        try:
            gym.spec(task_id)
        except Exception as exc:
            _fail(f"Gym task is not registered: {task_id}: {exc}")
        _ok(f"task registered: {task_id}")

    active_go2_usd_path = Path(go2_usd_path()).expanduser()
    bundled_path = bundled_go2_usd_path()
    base_body_name = os.environ.get("GO2_BASE_BODY_NAME", "base")
    foot_body_regex = os.environ.get("GO2_FOOT_BODY_REGEX", ".*_foot")
    height_scanner_prim = os.environ.get("GO2_HEIGHT_SCANNER_PRIM", f"{{ENV_REGEX_NS}}/Robot/{base_body_name}")

    print("[INFO] Asset contract:")
    print(f"[INFO]   bundled_go2_usd={bundled_path}")
    print(f"[INFO]   active_go2_usd={active_go2_usd_path}")
    print(f"[INFO]   GO2_BASE_BODY_NAME={base_body_name}")
    print(f"[INFO]   GO2_FOOT_BODY_REGEX={foot_body_regex}")
    print(f"[INFO]   GO2_HEIGHT_SCANNER_PRIM={height_scanner_prim}")

    if active_go2_usd_path.is_file():
        _ok(f"Go2 USD exists: {active_go2_usd_path}")
        if os.environ.get("GO2_USD_PATH") and base_body_name == "base" and foot_body_regex == ".*_foot":
            _warn(
                "Using the default IsaacLab Go2 naming contract with a custom USD. If this USD has base_link and "
                "no *_foot links, set GO2_BASE_BODY_NAME=base_link, GO2_FOOT_BODY_REGEX='.*_calf', and "
                "GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'."
            )
    else:
        _fail(f"Go2 USD does not exist: {active_go2_usd_path}")

    _ok("preflight complete")


if __name__ == "__main__":
    main()
