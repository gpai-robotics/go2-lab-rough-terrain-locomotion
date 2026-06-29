#!/usr/bin/env python3
"""Read-only Go2 DDS probe."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go2_monitor_schema import SDK_TO_POLICY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--subscribe-sport", action="store_true")
    parser.add_argument("--subscribe-lowcmd", action="store_true")
    parser.add_argument("--json-out", default="")
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


@dataclass
class TopicStats:
    name: str
    count: int = 0
    first_wall_time: float | None = None
    last_wall_time: float | None = None

    def mark(self) -> None:
        now = time.time()
        if self.first_wall_time is None:
            self.first_wall_time = now
        self.last_wall_time = now
        self.count += 1

    def hz(self) -> float | None:
        if self.first_wall_time is None or self.last_wall_time is None or self.count < 2:
            return None
        duration = self.last_wall_time - self.first_wall_time
        if duration <= 0:
            return None
        return float(self.count - 1) / duration


def _ensure_sdk_import_path(root: str) -> None:
    sdk_root = root or str((Path(__file__).resolve().parents[2] / "third_party" / "unitree_sdk2py").resolve())
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)


class Go2ReadOnlyProbe:
    def __init__(self, net_if: str, subscribe_sport: bool, subscribe_lowcmd: bool) -> None:
        self.net_if = net_if
        self.remote = RemoteController()
        self.low_state = None
        self.sport_state = None
        self.low_cmd = None
        self.low_stats = TopicStats("rt/lowstate")
        self.sport_stats = TopicStats("rt/sportmodestate")
        self.lowcmd_stats = TopicStats("rt/lowcmd")

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_ as LowCmdGo,
            unitree_go_msg_dds__LowState_ as LowStateGo,
            unitree_go_msg_dds__SportModeState_ as SportModeStateGo,
        )

        ChannelFactoryInitialize(0, net_if)
        self.low_state = LowStateGo()
        self.low_sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.low_sub.Init(self._on_lowstate, 10)

        self.sport_sub = None
        if subscribe_sport:
            self.sport_state = SportModeStateGo()
            self.sport_sub = ChannelSubscriber("rt/sportmodestate", type(self.sport_state))
            self.sport_sub.Init(self._on_sportstate, 10)

        self.lowcmd_sub = None
        if subscribe_lowcmd:
            self.low_cmd = LowCmdGo()
            self.lowcmd_sub = ChannelSubscriber("rt/lowcmd", type(self.low_cmd))
            self.lowcmd_sub.Init(self._on_lowcmd, 10)

    def _on_lowstate(self, msg: Any) -> None:
        self.low_state = msg
        self.low_stats.mark()
        try:
            self.remote.set(bytes(self.low_state.wireless_remote))
        except Exception:
            pass

    def _on_sportstate(self, msg: Any) -> None:
        self.sport_state = msg
        self.sport_stats.mark()

    def _on_lowcmd(self, msg: Any) -> None:
        self.low_cmd = msg
        self.lowcmd_stats.mark()

    def summary(self) -> dict[str, Any]:
        latest_q = None
        if self.low_state is not None:
            latest_q = np.asarray([float(self.low_state.motor_state[i].q) for i in SDK_TO_POLICY], dtype=np.float32).tolist()
        return {
            "net_if": self.net_if,
            "topics": {
                "rt/lowstate": {"count": self.low_stats.count, "hz": self.low_stats.hz()},
                "rt/sportmodestate": {"count": self.sport_stats.count, "hz": self.sport_stats.hz()},
                "rt/lowcmd": {"count": self.lowcmd_stats.count, "hz": self.lowcmd_stats.hz()},
            },
            "remote": {
                "vx": self.remote.ly,
                "vy": -self.remote.lx,
                "wz": -self.remote.rx,
            },
            "latest_joint_pos_policy_order": latest_q,
        }


def main() -> int:
    args = parse_args()
    _ensure_sdk_import_path(args.unitree_sdk_root)
    probe = Go2ReadOnlyProbe(args.net_if, args.subscribe_sport, args.subscribe_lowcmd)
    time.sleep(args.duration_s)
    report = probe.summary()
    if args.json_out:
        Path(args.json_out).expanduser().resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["topics"]["rt/lowstate"]["count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
