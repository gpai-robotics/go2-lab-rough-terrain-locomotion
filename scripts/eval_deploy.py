#!/usr/bin/env python3
"""Run a simple IsaacLab deploy rehearsal using an exported bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle-dir", required=True)
parser.add_argument("--task", default="RMA-Go2-BlindHistory-Rough-StageA")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--json-out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import go2_rough  # noqa: F401


def _unwrap_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _step_env(env, actions):
    step_out = env.step(actions)
    if len(step_out) == 5:
        obs, rewards, terminated, truncated, infos = step_out
        dones = terminated | truncated
        return obs, rewards, dones, infos
    obs, rewards, dones, infos = step_out
    return obs, rewards, dones, infos


def main() -> None:
    bundle_dir = Path(args_cli.bundle_dir)
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    deploy_cfg = json.loads((bundle_dir / f"{manifest['policy_name']}.deploy_config.json").read_text())

    ts_path = None
    for artifact in manifest["exported_artifacts"]:
        if artifact.endswith(".torchscript.pt"):
            ts_path = bundle_dir / artifact
            break
    if ts_path is None:
        raise SystemExit("Could not find a TorchScript artifact in the bundle.")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    obs = _unwrap_obs(env.get_observations())

    policy = torch.jit.load(str(ts_path), map_location=env.unwrapped.device)
    policy.eval()

    robot = env.unwrapped.scene["robot"]
    vel_err_sum = 0.0
    yaw_err_sum = 0.0
    tilt_sum = 0.0
    height_sum = 0.0
    done_count = 0

    with torch.inference_mode():
        for _ in range(args_cli.max_steps):
            actions = policy(obs["policy"], obs["policy_history"])
            obs, _, dones, _ = _step_env(env, actions)
            obs = _unwrap_obs(obs)

            command = env.unwrapped.command_manager.get_command("base_velocity")
            root_lin_vel = robot.data.root_lin_vel_b[:, :2]
            root_ang_vel = robot.data.root_ang_vel_b[:, 2]
            projected_gravity = robot.data.projected_gravity_b[:, :2]

            vel_err_sum += float((root_lin_vel - command[:, :2]).norm(dim=-1).mean().item())
            yaw_err_sum += float((root_ang_vel - command[:, 2]).abs().mean().item())
            tilt_sum += float(projected_gravity.norm(dim=-1).mean().item())
            height_sum += float(robot.data.root_pos_w[:, 2].mean().item())
            done_count += int(dones.sum().item())

    results = {
        "bundle_dir": str(bundle_dir),
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "max_steps": args_cli.max_steps,
        "policy_dim": deploy_cfg["observations"]["policy_dim"],
        "policy_history_length": deploy_cfg["observations"]["policy_history_length"],
        "vel_err_step_mean": vel_err_sum / args_cli.max_steps,
        "yaw_err_step_mean": yaw_err_sum / args_cli.max_steps,
        "base_height_mean": height_sum / args_cli.max_steps,
        "base_tilt_projected_gravity_xy_mean": tilt_sum / args_cli.max_steps,
        "terminations_total": done_count,
    }

    print(json.dumps(results, indent=2))
    if args_cli.json_out:
        Path(args_cli.json_out).write_text(json.dumps(results, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
