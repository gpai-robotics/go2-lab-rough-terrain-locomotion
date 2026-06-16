"""Shared observation-contract helpers for the mjlab-style branch."""

from __future__ import annotations

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass


def gait_phase(
    env,
    period: float = 0.6,
    command_name: str = "base_velocity",
    stand_command_threshold: float = 0.05,
) -> torch.Tensor:
    """Global gait clock used by the mjlab-style actor contract.

    This mirrors the `unitree_rl_mjlab` idea:
    - provide a compact sinusoidal phase signal to the actor
    - zero it out when the command is effectively standstill
    """

    global_phase = (env.episode_length_buf.float() * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    command = env.command_manager.get_command(command_name)
    stand_mask = torch.linalg.norm(command[:, :3], dim=1) < stand_command_threshold
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


@configclass
class MjlabCriticPrivilegedObsCfg(ObsGroup):
    """Critic-only privileged group for signals kept off the actor."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def apply_mjlab_policy_contract(
    policy_obs,
    *,
    include_gait_phase: bool = False,
    period: float = 0.6,
) -> None:
    """Mutate an IsaacLab locomotion policy observation group in place.

    Current IsaacLab velocity tasks place `base_lin_vel` inside the shared
    `policy` group. For the mjlab branch we remove it from the actor-facing
    group and optionally add an explicit gait phase term.
    """

    policy_obs.base_lin_vel = None
    if include_gait_phase:
        policy_obs.gait_phase = ObsTerm(
            func=gait_phase,
            params={"period": period, "command_name": "base_velocity"},
        )
    else:
        policy_obs.gait_phase = None
