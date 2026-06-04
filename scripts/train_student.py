#!/usr/bin/env python3
"""Train the blind-history rough student in IsaacLab."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="RMA-Go2-BlindHistory-Rough-StageA")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log-dir", type=str, default="logs/student")
parser.add_argument("--teacher-checkpoint", type=str, default=None)
parser.add_argument("--blind-warmstart", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
    if args_cli.teacher_checkpoint is not None:
        runner_cfg.algorithm.teacher_expert_path = str(args_cli.teacher_checkpoint)
    if args_cli.blind_warmstart is not None:
        runner_cfg.policy.actor_init_path = str(args_cli.blind_warmstart)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    log_dir = Path(args_cli.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=runner_cfg.device)
    runner.learn(num_learning_iterations=runner_cfg.max_iterations)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
