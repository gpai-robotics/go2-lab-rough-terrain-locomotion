import json
import os
from dataclasses import dataclass, field
from datetime import datetime
import torch


@dataclass
class OODMetrics:
    """Tracks and reports out-of-distribution evaluation metrics across episodes."""

    total_terminals: int = 0
    total_timeout: int = 0
    total_base_contact: int = 0
    total_bad_orientation: int = 0

    vel_tracking_errors: list[tuple[float, float]] = field(default_factory=list)

    def compute_velocity_error(self, env) -> tuple[float | None, float | None]:
        """Compute the velocity tracking errors"""
        try:
            robot = env.scene["robot"]
            command = env.command_manager.get_command("base_velocity")

            lin_vel = robot.data.root_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_ang_vel_b[:, 2]

            vel_err = torch.linalg.norm(lin_vel - command[:, :2], dim=-1).mean().item()
            yaw_err = torch.abs(yaw_vel - command[:, 2]).mean().item()
            return float(vel_err), float(yaw_err)
        except Exception:
            return None, None
        
    def update(self, dones, info, env) -> None:
        """Call once per sim step to accumulate terminal statistics."""
        base_contact = env.termination_manager.get_term("base_contact")
        bad_orientation = env.termination_manager.get_term("bad_orientation")
        time_out = env.termination_manager.get_term("time_out")

        self.total_terminals += int(dones.sum().item())
        self.total_base_contact += int((dones & base_contact).sum().item())
        self.total_bad_orientation += int((dones & bad_orientation).sum().item())
        self.total_timeout += int((dones & time_out).sum().item())

        if dones.any():
            vel_err, yaw_err = self.compute_velocity_error(env)
            if vel_err is not None and yaw_err is not None:
                self.vel_tracking_errors.append((vel_err, yaw_err))

    @property
    def computed_fractions(self) -> dict | None:
        """Returns per-cause terminal fractions, or None if no episodes completed."""
        if self.total_terminals == 0:
            return None
        n = self.total_terminals
        return {
            "timeout_fraction": self.total_timeout / n,
            "base_contact_fraction": self.total_base_contact / n,
            "bad_orientation_fraction": self.total_bad_orientation / n,
            "avg_vel_error": (
                sum(err[0] for err in self.vel_tracking_errors) / len(self.vel_tracking_errors)
                if self.vel_tracking_errors else None
            ),
            "avg_yaw_error": (
                sum(err[1] for err in self.vel_tracking_errors) / len(self.vel_tracking_errors)
                if self.vel_tracking_errors else None
            )
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout. No-ops if no episodes completed."""
        fracs = self.computed_fractions
        if fracs is None:
            print("[OOD] No episodes completed — nothing to report.")
            return

        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print("\n===== OOD Metrics =====")
        print(f"Episodes completed:       {self.total_terminals}")
        print(f"Average velocity error:   {fmt(fracs['avg_vel_error'])}")
        print(f"Average yaw error:        {fmt(fracs['avg_yaw_error'])}")
        print(f"Timeout fraction:         {fmt(fracs['timeout_fraction'])}")
        print(f"Base-contact fraction:    {fmt(fracs['base_contact_fraction'])}")
        print(f"Bad-orientation fraction: {fmt(fracs['bad_orientation_fraction'])}")

    def save(self, task_name: str, description: str, output_dir: str = "Metrics") -> str:
        """
        Persist metrics to a timestamped JSON file.

        Returns the path the file was written to.
        """
        fracs = self.computed_fractions or {}
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results = {
            "description": description,
            "timestamp": timestamp,
            "total_terminals": self.total_terminals,
            "total_timeout": self.total_timeout,
            "total_base_contact": self.total_base_contact,
            "total_bad_orientation": self.total_bad_orientation,
            **fracs,
        }

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"ood_metrics_{task_name}_{timestamp}.json")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        return save_path