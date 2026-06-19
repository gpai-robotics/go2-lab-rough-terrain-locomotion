#!/usr/bin/env python3
"""Train the blind rough AsymPPO policy."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Go2-Blind-Rough-MJLAB-AsymPPO-V1")
parser.add_argument("--flat-prior-checkpoint", default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-dir", default="logs/go2_blind_rough_asymppo_mjlab_v1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.flat_prior_checkpoint:
    os.environ["GO2_FLAT_PRIOR_CKPT"] = args_cli.flat_prior_checkpoint

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import go2_rough  # noqa: F401


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = int(args_cli.seed)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.max_iterations is not None:
        runner_cfg.max_iterations = int(args_cli.max_iterations)

    env_cfg.sim.log_dir = str(Path(args_cli.log_dir).expanduser() / "isaaclab")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    except Exception:
        print("[ERROR] Environment creation failed.")
        print(
            "[HINT] If your Go2 USD uses base_link and has no *_foot bodies, set "
            "GO2_BASE_BODY_NAME=base_link, GO2_FOOT_BODY_REGEX='.*_calf', and "
            "GO2_HEIGHT_SCANNER_PRIM='{ENV_REGEX_NS}/Robot/base_link'."
        )
        traceback.print_exc()
        raise
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = Path(args_cli.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device)
    runner.learn(num_learning_iterations=runner_cfg.max_iterations)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
