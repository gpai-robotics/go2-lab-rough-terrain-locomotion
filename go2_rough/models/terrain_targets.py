from __future__ import annotations

import torch


def terrain_lite_from_scan(scan: torch.Tensor) -> torch.Tensor:
    """Compress a 187-point height scan into gait-relevant terrain cues."""

    if scan.shape[-1] != 187:
        raise RuntimeError(
            "terrain_lite_from_scan expects the standard 1.6m x 1.0m, 0.1m-resolution "
            f"height scan with 187 samples, got shape {tuple(scan.shape)}."
        )

    grid = scan.view(scan.shape[0], 17, 11)
    rear = grid[:, :5, :]
    center = grid[:, 5:12, :]
    front = grid[:, 12:, :]
    left = grid[:, :, :5]
    right = grid[:, :, 6:]

    front_mean = front.mean(dim=(1, 2))
    center_mean = center.mean(dim=(1, 2))
    rear_mean = rear.mean(dim=(1, 2))
    left_mean = left.mean(dim=(1, 2))
    right_mean = right.mean(dim=(1, 2))
    height_std = grid.std(dim=(1, 2))
    height_range = grid.amax(dim=(1, 2)) - grid.amin(dim=(1, 2))
    forward_slope = (front_mean - rear_mean) / 1.6
    lateral_slope = (left_mean - right_mean) / 1.0

    foot_patches = torch.stack(
        [
            grid[:, :5, :5].std(dim=(1, 2)),
            grid[:, :5, 6:].std(dim=(1, 2)),
            grid[:, 12:, :5].std(dim=(1, 2)),
            grid[:, 12:, 6:].std(dim=(1, 2)),
        ],
        dim=-1,
    )
    contact_roughness = foot_patches.mean(dim=-1)

    return torch.stack(
        [
            front_mean,
            center_mean,
            rear_mean,
            left_mean,
            right_mean,
            front_mean - center_mean,
            center_mean - rear_mean,
            left_mean - right_mean,
            height_std,
            height_range,
            forward_slope,
            lateral_slope,
            contact_roughness,
        ],
        dim=-1,
    )
