#!/usr/bin/env python3
"""Bundle-driven low-level deployment runner for Unitree Go2 hardware."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from go2_monitor_schema import POLICY_TO_SDK, SDK_TO_POLICY
from history_layout import flatten_policy_history, resolve_history_layout

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--net-if", required=True)
    parser.add_argument("--unitree-sdk-root", default="", help="Path to a unitree_sdk2py checkout.")
    parser.add_argument("--mode-switch-script", default="", help="Optional SDK mode-switch helper path.")
    parser.add_argument("--skip-mode-switch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stance-only", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--command-x", type=float, default=0.3)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--stance-ramp-s", type=float, default=2.0)
    return parser.parse_args()


class RemoteController:
    def __init__(self) -> None:
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16

    def set(self, data: bytes) -> None:
        if len(data) < 24:
            return
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


def projected_gravity_from_quat(quaternion_wxyz: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    qw, qx, qy, qz = quaternion_wxyz
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


def _find_exported_artifact(bundle_dir: Path, suffix: str) -> Path:
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(suffix):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    raise SystemExit(f"Could not find artifact ending with {suffix!r} in {bundle_dir}")


def _ensure_sdk_import_path(root: str) -> None:
    sdk_root = root or str((Path(__file__).resolve().parents[2] / "third_party" / "unitree_sdk2py").resolve())
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)


def _prepare_low_level_mode(net_if: str, mode_switch_script: str, skip_mode_switch: bool) -> None:
    if skip_mode_switch:
        return
    if not mode_switch_script:
        raise SystemExit("Refusing to switch modes without --mode-switch-script or --skip-mode-switch.")
    subprocess.run([sys.executable, mode_switch_script, net_if], check=True)


def create_zero_cmd(cmd: Any) -> None:
    for motor in cmd.motor_cmd:
        motor.q = 0.0
        motor.qd = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


def init_cmd_go(cmd: Any) -> None:
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for motor in cmd.motor_cmd:
        motor.mode = 0x0A
        motor.q = POS_STOP_F
        motor.qd = VEL_STOP_F
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


@dataclass
class HardwareContract:
    policy_obs_dim: int
    policy_history_length: int
    action_dim: int
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray


class Go2HardwareRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        self.remote = RemoteController()
        self.low_state = None

        manifest = json.loads((self.bundle_dir / "bundle_manifest.json").read_text())
        if manifest.get("policy_kind") != "blind_history_policy":
            raise SystemExit("This hardware runner currently supports only blind_history_policy bundles.")

        deploy_cfg = json.loads(_find_exported_artifact(self.bundle_dir, ".deploy_config.json").read_text())
        metadata = json.loads(_find_exported_artifact(self.bundle_dir, ".export_metadata.json").read_text())
        policy_path = _find_exported_artifact(self.bundle_dir, ".torchscript.pt")
        tensor_contract = metadata["tensor_contract"]
        self.contract = HardwareContract(
            policy_obs_dim=int(tensor_contract["policy_obs_dim"]),
            policy_history_length=int(deploy_cfg["observations"]["policy_history_length"]),
            action_dim=int(tensor_contract["action_dim"]),
            default_joint_pos=np.asarray(deploy_cfg["robot"]["default_joint_pos"], dtype=np.float32),
            joint_stiffness=np.asarray(deploy_cfg["robot"]["joint_stiffness"], dtype=np.float32),
            joint_damping=np.asarray(deploy_cfg["robot"]["joint_damping"], dtype=np.float32),
            action_scale=np.asarray(deploy_cfg["actions"]["scale"], dtype=np.float32),
            action_offset=np.asarray(deploy_cfg["actions"]["offset"], dtype=np.float32),
        )
        self.control_dt = float(deploy_cfg["control"]["step_dt"])
        self.policy_order = list(deploy_cfg["observations"]["policy_order"])
        self.history_layout = resolve_history_layout(deploy_cfg["observations"])
        self.command_default = np.asarray([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
        if args.forward_only:
            self.command_default[1:] = 0.0
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()
        self.history = np.zeros((self.contract.policy_history_length, self.contract.policy_obs_dim), dtype=np.float32)
        self.last_action = np.zeros(self.contract.action_dim, dtype=np.float32)

        if args.dry_run:
            self._dds_ready = False
            return

        _ensure_sdk_import_path(args.unitree_sdk_root)
        _prepare_low_level_mode(args.net_if, args.mode_switch_script, args.skip_mode_switch)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_ as LowCmdGo,
            unitree_go_msg_dds__LowState_ as LowStateGo,
        )

        ChannelFactoryInitialize(0, args.net_if)
        self.low_cmd = LowCmdGo()
        self.low_state = LowStateGo()
        self.pub = ChannelPublisher("rt/lowcmd", type(self.low_cmd))
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.sub.Init(self._on_lowstate, 10)
        self.crc = CRC()
        init_cmd_go(self.low_cmd)
        self._dds_ready = True

    def _on_lowstate(self, msg: Any) -> None:
        self.low_state = msg
        try:
            self.remote.set(bytes(self.low_state.wireless_remote))
        except Exception:
            pass

    def _send_cmd(self) -> None:
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _wait_for_state(self) -> None:
        while self.low_state is None or getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)

    def _policy_order_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        q_hw = np.array([self.low_state.motor_state[i].q for i in range(self.contract.action_dim)], dtype=np.float32)
        dq_hw = np.array([self.low_state.motor_state[i].dq for i in range(self.contract.action_dim)], dtype=np.float32)
        return q_hw[SDK_TO_POLICY], dq_hw[SDK_TO_POLICY]

    def _policy_obs(self, command: np.ndarray) -> np.ndarray:
        q, dq = self._policy_order_joint_state()
        imu = self.low_state.imu_state
        base_ang_vel = np.asarray([float(x) for x in imu.gyroscope], dtype=np.float32)
        quat_wxyz = [float(imu.quaternion[0]), float(imu.quaternion[1]), float(imu.quaternion[2]), float(imu.quaternion[3])]
        projected_gravity = projected_gravity_from_quat(quat_wxyz)
        obs = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                command,
                q - self.contract.default_joint_pos,
                dq,
                self.last_action,
            ],
            dtype=np.float32,
        )
        return obs

    def _write_targets(self, q_target_policy: np.ndarray) -> None:
        q_target_sdk = q_target_policy[POLICY_TO_SDK]
        kp_sdk = self.contract.joint_stiffness[POLICY_TO_SDK]
        kd_sdk = self.contract.joint_damping[POLICY_TO_SDK]
        for i in range(self.contract.action_dim):
            motor = self.low_cmd.motor_cmd[i]
            motor.q = float(q_target_sdk[i])
            motor.qd = 0.0
            motor.kp = float(kp_sdk[i])
            motor.kd = float(kd_sdk[i])
            motor.tau = 0.0

    def dry_run_report(self) -> dict[str, Any]:
        return {
            "status": "dry_run",
            "bundle_dir": str(self.bundle_dir),
            "net_if": self.args.net_if,
            "control_dt": self.control_dt,
            "policy_obs_dim": self.contract.policy_obs_dim,
            "policy_history_length": self.contract.policy_history_length,
            "action_dim": self.contract.action_dim,
        }

    def run(self) -> dict[str, Any]:
        if self.args.dry_run:
            return self.dry_run_report()

        self._wait_for_state()
        stance_start = time.time()
        while time.time() - stance_start < self.args.stance_ramp_s:
            alpha = min(1.0, (time.time() - stance_start) / max(self.args.stance_ramp_s, 1e-6))
            q, _ = self._policy_order_joint_state()
            q_target = (1.0 - alpha) * q + alpha * self.contract.default_joint_pos
            self._write_targets(q_target)
            self._send_cmd()
            time.sleep(self.control_dt)

        command = self.command_default.copy()
        obs = self._policy_obs(command)
        self.history[:] = obs
        start = time.time()
        steps = 0
        action_abs_max = 0.0

        while time.time() - start < self.args.duration_s:
            if self.args.stance_only:
                q_target = self.contract.default_joint_pos.copy()
            else:
                obs = self._policy_obs(command)
                self.history = np.roll(self.history, -1, axis=0)
                self.history[-1] = obs
                history_flat = flatten_policy_history(self.history, self.policy_order, layout=self.history_layout)
                with torch.inference_mode():
                    action = self.policy(
                        torch.from_numpy(obs).unsqueeze(0),
                        torch.from_numpy(history_flat).unsqueeze(0),
                    )[0].cpu().numpy().astype(np.float32)
                self.last_action[:] = action
                action_abs_max = max(action_abs_max, float(np.max(np.abs(action))))
                q_target = self.contract.action_offset + self.contract.action_scale * action
            self._write_targets(q_target)
            self._send_cmd()
            time.sleep(self.control_dt)
            steps += 1

        create_zero_cmd(self.low_cmd)
        for _ in range(20):
            self._send_cmd()
            time.sleep(self.control_dt)
        return {
            "status": "complete",
            "steps": steps,
            "duration_s": self.args.duration_s,
            "stance_only": self.args.stance_only,
            "forward_only": self.args.forward_only,
            "action_abs_max": action_abs_max,
        }


def main() -> int:
    args = parse_args()
    runner = Go2HardwareRunner(args)
    report = runner.run()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
