#!/usr/bin/env python3
"""Live read-only Go2 monitor for deployment tests."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from go2_monitor_schema import POLICY_JOINT_NAMES, POLICY_TO_SDK, SCHEMA_NAME, SCHEMA_VERSION, SDK_TO_POLICY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", required=True)
    parser.add_argument("--history-sec", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--subscribe-lowcmd", action="store_true")
    parser.add_argument("--jsonl-out", default="")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--unitree-sdk-root", default="", help="Path to a unitree_sdk2py checkout.")
    return parser.parse_args()


class RemoteController:
    def __init__(self) -> None:
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0

    def set(self, data: bytes) -> None:
        if len(data) < 24:
            return
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


class RollingSeries:
    def __init__(self, maxlen: int) -> None:
        self.t = deque(maxlen=maxlen)
        self.values = deque(maxlen=maxlen)

    def append(self, t: float, values: np.ndarray) -> None:
        self.t.append(float(t))
        self.values.append(np.asarray(values, dtype=np.float32))


def _ensure_sdk_import_path(root: str) -> None:
    sdk_root = root or str((Path(__file__).resolve().parents[2] / "third_party" / "unitree_sdk2py").resolve())
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)


class Go2RealtimeMonitor:
    def __init__(self, net_if: str, history_sec: float, sample_hz: float, subscribe_lowcmd: bool, jsonl_out: Path | None = None) -> None:
        self.net_if = net_if
        self.history_sec = history_sec
        self.sample_hz = sample_hz
        self.subscribe_lowcmd = subscribe_lowcmd
        self.jsonl_out = jsonl_out
        self.remote = RemoteController()
        self.lock = threading.Lock()

        maxlen = max(10, int(history_sec * sample_hz) + 5)
        self.joint_pos = RollingSeries(maxlen)
        self.joint_vel = RollingSeries(maxlen)
        self.tau_est = RollingSeries(maxlen)
        self.q_err = RollingSeries(maxlen)
        self.latest_status: dict[str, Any] = {"low_hz": 0.0, "sport_hz": 0.0, "lowcmd_hz": 0.0}
        self._latest_q = np.zeros(12, dtype=np.float32)
        self._latest_dq = np.zeros(12, dtype=np.float32)
        self._latest_tau_est = np.zeros(12, dtype=np.float32)
        self._latest_q_des = np.zeros(12, dtype=np.float32)
        self._has_lowcmd = False
        self._last_low_sample_t = 0.0
        self._last_sport_sample_t = 0.0

        self._jsonl_handle = None
        if self.jsonl_out is not None:
            self.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_handle = self.jsonl_out.open("w", encoding="utf-8", buffering=1)

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_ as LowCmdGo,
            unitree_go_msg_dds__LowState_ as LowStateGo,
            unitree_go_msg_dds__SportModeState_ as SportModeStateGo,
        )

        self._topic_counts = {"low": 0, "sport": 0, "lowcmd": 0}
        self._topic_first_t = {"low": None, "sport": None, "lowcmd": None}
        ChannelFactoryInitialize(0, net_if)

        self.low_state = LowStateGo()
        self.low_sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.low_sub.Init(self._on_lowstate, 10)

        self.sport_state = SportModeStateGo()
        self.sport_sub = ChannelSubscriber("rt/sportmodestate", type(self.sport_state))
        self.sport_sub.Init(self._on_sportstate, 10)

        self.lowcmd_sub = None
        if subscribe_lowcmd:
            self.low_cmd = LowCmdGo()
            self.lowcmd_sub = ChannelSubscriber("rt/lowcmd", type(self.low_cmd))
            self.lowcmd_sub.Init(self._on_lowcmd, 10)

    def _mark_topic(self, name: str) -> float:
        now = time.time()
        if self._topic_first_t[name] is None:
            self._topic_first_t[name] = now
        self._topic_counts[name] += 1
        first = self._topic_first_t[name]
        count = self._topic_counts[name]
        hz = 0.0 if count < 2 or first is None or now <= first else float(count - 1) / (now - first)
        self.latest_status[{"low": "low_hz", "sport": "sport_hz", "lowcmd": "lowcmd_hz"}[name]] = hz
        return now

    def _on_lowstate(self, msg: Any) -> None:
        self.low_state = msg
        now = self._mark_topic("low")
        try:
            self.remote.set(bytes(self.low_state.wireless_remote))
        except Exception:
            pass
        if now - self._last_low_sample_t < (1.0 / self.sample_hz):
            return
        self._last_low_sample_t = now
        q = np.asarray([float(msg.motor_state[i].q) for i in SDK_TO_POLICY], dtype=np.float32)
        dq = np.asarray([float(msg.motor_state[i].dq) for i in SDK_TO_POLICY], dtype=np.float32)
        tau_est = np.asarray([float(msg.motor_state[i].tau_est) for i in SDK_TO_POLICY], dtype=np.float32)
        with self.lock:
            self._latest_q = q
            self._latest_dq = dq
            self._latest_tau_est = tau_est
            self.joint_pos.append(now, q)
            self.joint_vel.append(now, dq)
            self.tau_est.append(now, tau_est)
            if self._has_lowcmd:
                self.q_err.append(now, self._latest_q_des - q)
            self._write_jsonl_locked(now)

    def _on_sportstate(self, msg: Any) -> None:
        now = self._mark_topic("sport")
        if now - self._last_sport_sample_t < (1.0 / self.sample_hz):
            return
        self._last_sport_sample_t = now

    def _on_lowcmd(self, msg: Any) -> None:
        self.low_cmd = msg
        self._mark_topic("lowcmd")
        q_des = np.asarray([float(msg.motor_cmd[i].q) for i in SDK_TO_POLICY], dtype=np.float32)
        with self.lock:
            self._latest_q_des = q_des
            self._has_lowcmd = True

    def _write_jsonl_locked(self, now: float) -> None:
        if self._jsonl_handle is None:
            return
        payload = {
            "schema": {
                "name": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
                "joint_order": "policy",
                "joint_names": POLICY_JOINT_NAMES,
                "policy_to_sdk": POLICY_TO_SDK.tolist(),
            },
            "wall_time": now,
            "dds_hz": self.latest_status.copy(),
            "remote_cmd": {"vx": self.remote.ly, "vy": -self.remote.lx, "wz": -self.remote.rx},
            "latest": {
                "q": self._latest_q.tolist(),
                "q_des": self._latest_q_des.tolist(),
                "joint_vel": self._latest_dq.tolist(),
                "tau_est": self._latest_tau_est.tolist(),
            },
        }
        self._jsonl_handle.write(json.dumps(payload) + "\n")

    def print_status(self) -> None:
        with self.lock:
            q_err = self._latest_q_des - self._latest_q if self._has_lowcmd else np.zeros(12, dtype=np.float32)
            print(
                json.dumps(
                    {
                        "dds_hz": self.latest_status,
                        "remote_cmd": {"vx": self.remote.ly, "vy": -self.remote.lx, "wz": -self.remote.rx},
                        "joint_pos_mean": float(self._latest_q.mean()),
                        "joint_vel_abs_max": float(np.max(np.abs(self._latest_dq))),
                        "tau_est_abs_max": float(np.max(np.abs(self._latest_tau_est))),
                        "q_err_abs_max": float(np.max(np.abs(q_err))),
                    }
                ),
                flush=True,
            )

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()


def main() -> int:
    args = parse_args()
    _ensure_sdk_import_path(args.unitree_sdk_root)
    jsonl_out = Path(args.jsonl_out).expanduser().resolve() if args.jsonl_out else None
    monitor = Go2RealtimeMonitor(args.net_if, args.history_sec, args.sample_hz, args.subscribe_lowcmd, jsonl_out)
    try:
        start = time.time()
        while True:
            time.sleep(1.0)
            monitor.print_status()
            if args.duration_s > 0 and time.time() - start >= args.duration_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
