#!/usr/bin/env python3
"""Preflight checks for the public Go2 rough-terrain IsaacLab setup."""

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
    except Exception as exc:
        _fail(f"go2_rough import failed: {exc}")

    for task_id in ("Go2-Flat-MJLAB-Prior-V1", "Go2-Blind-Rough-MJLAB-AsymPPO-V1"):
        try:
            gym.spec(task_id)
        except Exception as exc:
            _fail(f"Gym task is not registered: {task_id}: {exc}")
        _ok(f"task registered: {task_id}")

    go2_usd_path = os.environ.get("GO2_USD_PATH")
    if go2_usd_path:
        path = Path(go2_usd_path).expanduser()
        if path.is_file():
            _ok(f"GO2_USD_PATH={path}")
        else:
            _warn(f"GO2_USD_PATH is set but does not exist: {path}")
    else:
        _warn("GO2_USD_PATH is not set. Set it before training if the default public placeholder asset is unavailable.")

    _ok("preflight complete")


if __name__ == "__main__":
    main()
