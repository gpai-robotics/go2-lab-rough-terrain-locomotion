"""Standalone MuJoCo runtime bridge for exported Go2 blind-history bundles."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from history_layout import flatten_policy_history, resolve_history_layout


@dataclass
class BridgeConfig:
    model_path: Path
    policy_artifact_path: Path
    deploy_config_path: Path
    control_dt: float = 0.02
    physics_dt: float = 0.005
    command_x: float = 0.5
    command_y: float = 0.0
    command_yaw: float = 0.0
    max_steps: int = 900
    trace_steps: int = 25
    viewer: bool = False
    viewer_dt: float = 0.02
    real_time_factor: float = 1.0


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


class Go2MujocoDeployBridge:
    """Minimal MuJoCo-side runtime for exported blind-history policies."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MuJoCo Python package is not available.") from exc
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is not available for loading the exported policy.") from exc

        self.mujoco = mujoco
        self.torch = torch
        self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(cfg.physics_dt)
        self.policy = torch.jit.load(str(cfg.policy_artifact_path), map_location="cpu")
        self.policy.eval()
        self.deploy_cfg = json.loads(Path(cfg.deploy_config_path).read_text())

        self.joint_names = list(self.deploy_cfg["robot"]["joint_names"])
        self.actuator_names = list(self.deploy_cfg["robot"]["actuator_names"])
        self.default_joint_pos = np.asarray(self.deploy_cfg["robot"]["default_joint_pos"], dtype=np.float32)
        self.joint_stiffness = np.asarray(self.deploy_cfg["robot"]["joint_stiffness"], dtype=np.float32)
        self.joint_damping = np.asarray(self.deploy_cfg["robot"]["joint_damping"], dtype=np.float32)
        self.action_scale = np.asarray(self.deploy_cfg["actions"]["scale"], dtype=np.float32)
        self.action_offset = np.asarray(self.deploy_cfg["actions"]["offset"], dtype=np.float32)
        self.policy_order = list(self.deploy_cfg["observations"]["policy_order"])
        self.policy_dim = int(self.deploy_cfg["observations"]["policy_dim"])
        self.policy_history_length = int(self.deploy_cfg["observations"]["policy_history_length"])
        self.history_layout = resolve_history_layout(self.deploy_cfg["observations"])
        self.command = np.asarray([cfg.command_x, cfg.command_y, cfg.command_yaw], dtype=np.float32)

        self.base_body_id = self._body_id(["base", "base_link", "trunk"])
        self.joint_qpos_indices = np.asarray([self._joint_qpos_index(name) for name in self.joint_names], dtype=np.int32)
        self.joint_dof_indices = np.asarray([self._joint_dof_index(name) for name in self.joint_names], dtype=np.int32)
        self.actuator_indices = np.asarray([self._actuator_index(name) for name in self.actuator_names], dtype=np.int32)
        self.ctrl_limit = np.asarray(self.model.actuator_ctrlrange[self.actuator_indices], dtype=np.float32)

        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        self.history = np.zeros((self.policy_history_length, self.policy_dim), dtype=np.float32)
        self._reset_state()

    def _body_id(self, candidates: list[str]) -> int:
        for name in candidates:
            try:
                return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            except Exception:
                continue
        raise RuntimeError(f"Could not locate any base body from candidates: {candidates}")

    def _joint_qpos_index(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dof_index(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_dofadr[joint_id])

    def _actuator_index(self, actuator_name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))

    def _reset_state(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray(self.deploy_cfg["robot"].get("base_init_pos", [0.0, 0.0, 0.4]), dtype=np.float32)
        self.data.qpos[3:7] = np.asarray(
            self.deploy_cfg["robot"].get("base_init_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.data.qpos[self.joint_qpos_indices] = self.default_joint_pos
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        obs = self.current_policy_obs()
        self.history[:] = obs

    def current_policy_obs(self) -> np.ndarray:
        base_ang_vel = np.asarray(self.data.qvel[3:6], dtype=np.float32)
        quat_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float32)
        projected_gravity = projected_gravity_from_quat(quat_wxyz)
        q = np.asarray(self.data.qpos[self.joint_qpos_indices], dtype=np.float32)
        dq = np.asarray(self.data.qvel[self.joint_dof_indices], dtype=np.float32)
        joint_pos_rel = q - self.default_joint_pos
        obs = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                self.command,
                joint_pos_rel,
                dq,
                self.last_action,
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.policy_dim:
            raise RuntimeError(f"Policy observation dimension mismatch: got {obs.shape[0]}, expected {self.policy_dim}")
        return obs

    def current_policy_history(self) -> np.ndarray:
        return flatten_policy_history(self.history, self.policy_order, layout=self.history_layout)

    def step_policy(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.current_policy_obs()
        hist = self.current_policy_history()
        with self.torch.inference_mode():
            action = self.policy(
                self.torch.from_numpy(obs).unsqueeze(0),
                self.torch.from_numpy(hist).unsqueeze(0),
            )[0].cpu().numpy().astype(np.float32)
        self.last_action[:] = action
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs
        q_target = self.action_offset + self.action_scale * action
        return action, q_target

    def apply_pd(self, q_target: np.ndarray) -> np.ndarray:
        q = np.asarray(self.data.qpos[self.joint_qpos_indices], dtype=np.float32)
        dq = np.asarray(self.data.qvel[self.joint_dof_indices], dtype=np.float32)
        tau = self.joint_stiffness * (q_target - q) - self.joint_damping * dq
        lower = self.ctrl_limit[:, 0]
        upper = self.ctrl_limit[:, 1]
        tau = np.clip(tau, lower, upper)
        self.data.ctrl[self.actuator_indices] = tau
        return tau

    def run(self) -> dict[str, Any]:
        decimation = max(1, int(round(self.cfg.control_dt / self.cfg.physics_dt)))
        trace = []
        action_abs_max = 0.0
        torque_abs_max = 0.0
        base_height_min = float("inf")
        start_wall = time.time()

        viewer_ctx = None
        if self.cfg.viewer:
            viewer_ctx = self.mujoco.viewer.launch_passive(self.model, self.data)

        try:
            q_target = self.default_joint_pos.copy()
            for step_idx in range(self.cfg.max_steps):
                if step_idx % decimation == 0:
                    action, q_target = self.step_policy()
                    action_abs_max = max(action_abs_max, float(np.max(np.abs(action))))
                tau = self.apply_pd(q_target)
                torque_abs_max = max(torque_abs_max, float(np.max(np.abs(tau))))
                self.mujoco.mj_step(self.model, self.data)
                base_height_min = min(base_height_min, float(self.data.qpos[2]))
                if len(trace) < self.cfg.trace_steps:
                    trace.append(
                        {
                            "step": step_idx,
                            "time": float(self.data.time),
                            "command": self.command.tolist(),
                            "base_height": float(self.data.qpos[2]),
                            "action_abs_max": float(np.max(np.abs(self.last_action))),
                            "torque_abs_max": float(np.max(np.abs(tau))),
                        }
                    )
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    sleep_s = self.cfg.viewer_dt / max(self.cfg.real_time_factor, 1.0e-6)
                    time.sleep(max(0.0, sleep_s))
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()

        end_wall = time.time()
        return {
            "status": "complete",
            "sim_time_s": float(self.data.time),
            "wall_time_s": float(end_wall - start_wall),
            "max_steps": int(self.cfg.max_steps),
            "base_height_final": float(self.data.qpos[2]),
            "base_height_min": float(base_height_min),
            "action_abs_max": float(action_abs_max),
            "torque_abs_max": float(torque_abs_max),
            "trace": trace,
        }
