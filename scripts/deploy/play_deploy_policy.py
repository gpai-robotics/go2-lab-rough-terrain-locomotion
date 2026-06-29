#!/usr/bin/env python3
"""Deployment-side Isaac rehearsal for exported blind-history bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle-dir", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--json-out", type=str, default=None)
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)
parser.add_argument("--compare-source", action="store_true")

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        parser.print_help()
        print("\nIsaacLab launcher arguments are available when this script runs under IsaacLab.")
        raise SystemExit(0) from exc
    raise SystemExit(
        "IsaacLab is required for deploy-side rehearsal. Run with:\n"
        "  bash scripts/isaaclab_user.sh -p scripts/deploy/play_deploy_policy.py ..."
    ) from exc

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

repo_root = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from deploy_blind_history_policy import build_deployable_module, load_checkpoint_state
from go2_deploy_contract import BLIND_HISTORY_OBSERVATION_GROUPS
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
    if len(step_out) == 4:
        obs, rewards, dones, infos = step_out
        return obs, rewards, dones, infos
    raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")


def _find_artifact(bundle_dir: Path, manifest: dict, suffix: str) -> Path:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(suffix):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    raise SystemExit(f"Could not find an artifact ending with {suffix!r} in {bundle_dir}")


def main() -> int:
    bundle_dir = Path(args_cli.bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    groups = manifest.get("deployable_observation_groups")
    if groups != BLIND_HISTORY_OBSERVATION_GROUPS:
        raise SystemExit(
            f"Unsupported deployable observation contract: {groups}. "
            f"Expected {BLIND_HISTORY_OBSERVATION_GROUPS}."
        )
    policy_path = _find_artifact(bundle_dir, manifest, ".torchscript.pt")
    deploy_cfg = json.loads(_find_artifact(bundle_dir, manifest, ".deploy_config.json").read_text())
    policy_history_length = int(deploy_cfg["observations"]["policy_history_length"])

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if (
        args_cli.command_x is not None
        or args_cli.command_y is not None
        or args_cli.command_yaw is not None
    ):
        cmd = env_cfg.commands.base_velocity
        if args_cli.command_x is not None:
            cmd.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
        if args_cli.command_y is not None:
            cmd.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
        if args_cli.command_yaw is not None:
            cmd.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)
        cmd.resampling_time_range = (1.0e9, 1.0e9)
        cmd.rel_standing_envs = 0.0
        cmd.rel_heading_envs = 0.0
        cmd.heading_command = False

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        device = env.unwrapped.device
        policy = torch.jit.load(str(policy_path), map_location=device)
        policy.eval()

        source_policy = None
        if args_cli.compare_source:
            source_policy = build_deployable_module(
                state_dict=load_checkpoint_state(Path(manifest["source_checkpoint"]).expanduser()),
                policy_obs_dim=int(deploy_cfg["observations"]["policy_dim"]),
                policy_history_length=policy_history_length,
            ).to(device)
            source_policy.eval()

        obs = _unwrap_obs(env.reset())
        reward_sum = torch.zeros(env.unwrapped.num_envs, device=device)
        done_count = 0
        action_abs_max = 0.0
        source_export_abs_diff_max = 0.0
        source_export_abs_diff_mean_sum = 0.0
        diff_samples = 0

        for _step_idx in range(args_cli.max_steps):
            policy_obs = obs["policy"]
            history_obs = obs["policy_history"]
            with torch.inference_mode():
                actions = policy(policy_obs, history_obs)
                if source_policy is not None:
                    source_actions = source_policy(policy_obs, history_obs)
                    diff = (source_actions - actions).abs()
                    source_export_abs_diff_max = max(source_export_abs_diff_max, float(diff.max().item()))
                    source_export_abs_diff_mean_sum += float(diff.mean().item())
                    diff_samples += 1
            action_abs_max = max(action_abs_max, float(actions.abs().max().item()))
            obs, rewards, dones, _infos = _step_env(env, actions)
            obs = _unwrap_obs(obs)
            reward_sum += rewards
            done_count += int(dones.sum().item())

        report = {
            "status": "ok",
            "bundle_dir": str(bundle_dir),
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "mean_reward_sum": float(reward_sum.mean().item()),
            "done_count": done_count,
            "action_abs_max": action_abs_max,
            "compare_source": bool(source_policy is not None),
            "source_export_abs_diff_max": source_export_abs_diff_max if source_policy is not None else None,
            "source_export_abs_diff_mean": (
                source_export_abs_diff_mean_sum / diff_samples if source_policy is not None and diff_samples else None
            ),
        }
        if args_cli.json_out:
            Path(args_cli.json_out).expanduser().resolve().write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
