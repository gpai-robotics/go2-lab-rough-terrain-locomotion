"""Custom height-field terrains for TRAKR."""

from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Callable

import numpy as np
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def stairs_stepping_stones_terrain(
    difficulty: float, cfg: HfStairsSteppingStonesTerrainCfg
) -> np.ndarray:
    """Generate rugged random blocks with crevices and a gradual uphill trend.

    The terrain is split into irregular block patches. Heights follow a noisy
    directional climb, so the result resembles raised paving slabs that get
    progressively taller along the x-axis without forming a center platform or
    pyramid.
    """

    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    if cfg.inverted:
        step_height *= -1.0

    block_width = cfg.stone_width_range[1] - difficulty * (
        cfg.stone_width_range[1] - cfg.stone_width_range[0]
    )
    crevice_width = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    crevice_depth = cfg.crevice_depth_range[0] + difficulty * (
        cfg.crevice_depth_range[1] - cfg.crevice_depth_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)

    block_width = max(1, int(block_width / cfg.horizontal_scale))
    crevice_width = max(1, int(crevice_width / cfg.horizontal_scale))
    block_height_noise = max(0, int(cfg.stone_height_max / cfg.vertical_scale))
    crevice_depth = max(1, int(crevice_depth / cfg.vertical_scale))

    min_block_width = max(1, int(0.6 * block_width))
    max_block_width = max(min_block_width + 1, int(1.4 * block_width))
    max_height = max(
        abs(step_height),
        int((min(cfg.size) / max(cfg.step_width, cfg.horizontal_scale)) * abs(step_height) * 0.6),
    )

    x_edges = _make_irregular_edges(width_pixels, min_block_width, max_block_width, crevice_width)
    y_edges = _make_irregular_edges(length_pixels, min_block_width, max_block_width, crevice_width)
    block_heights = _make_random_block_heights(
        len(x_edges),
        len(y_edges),
        max_height,
        block_height_noise,
        cfg.inverted,
    )

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    for x_index, (x_start, x_stop, x_crevice_stop) in enumerate(x_edges):
        for y_index, (y_start, y_stop, y_crevice_stop) in enumerate(y_edges):
            block_height = block_heights[x_index, y_index]
            hf_raw[x_start:x_stop, y_start:y_stop] = block_height
            hf_raw[x_start:x_stop, y_stop:y_crevice_stop] = block_height - crevice_depth

        crevice_base = block_heights[x_index, :].min() if cfg.inverted else block_heights[x_index, :].max()
        hf_raw[x_stop:x_crevice_stop, :] = crevice_base - crevice_depth

    return np.rint(hf_raw).astype(np.int16)


def _make_irregular_edges(
    num_pixels: int,
    min_block_width: int,
    max_block_width: int,
    crevice_width: int,
) -> list[tuple[int, int, int]]:
    """Create irregular block and crevice bounds along one axis."""

    edges = []
    start = 0
    while start < num_pixels:
        block_width = np.random.randint(min_block_width, max_block_width)
        stop = min(num_pixels, start + block_width)
        crevice_stop = min(num_pixels, stop + crevice_width)
        edges.append((start, stop, crevice_stop))
        start = crevice_stop
    return edges


def _make_random_block_heights(
    num_x: int,
    num_y: int,
    max_height: int,
    block_height_noise: int,
    inverted: bool,
) -> np.ndarray:
    """Create a noisy height map that climbs along x without a pyramid center."""

    x_trend = np.linspace(0, max_height, num_x, dtype=np.float64).reshape(num_x, 1)
    y_offsets = np.random.uniform(-0.12 * max_height, 0.12 * max_height, size=(1, num_y))
    random_walk = np.cumsum(
        np.random.randint(0, block_height_noise + 1, size=(num_x, num_y)),
        axis=0,
    )
    local_noise = np.random.randint(-block_height_noise, block_height_noise + 1, size=(num_x, num_y))
    heights = x_trend + y_offsets + random_walk + local_noise
    heights -= heights.min()
    if heights.max() > 0:
        heights *= max_height / heights.max()
    if inverted:
        heights *= -1
    return np.rint(heights).astype(np.int16)


@configclass
class HfStairsSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for rugged random block terrain."""

    function: Callable = stairs_stepping_stones_terrain

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = MISSING
    """The width of the steps (in m)."""

    stone_height_max: float = MISSING
    """The maximum random height offset of each block (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the terrain blocks (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum width of crevices between blocks (in m)."""

    crevice_depth_range: tuple[float, float] = (0.01, 0.04)
    """The minimum and maximum depth of crevices between blocks (in m)."""

    holes_depth: float = -10.0
    """Unused compatibility field. Crevices use :attr:`crevice_depth_range`."""

    platform_width: float = 1.0
    """Unused compatibility field. No center platform is generated."""

    inverted: bool = False
    """Whether the stair profile is inverted."""


@configclass
class HfInvertedStairsSteppingStonesTerrainCfg(HfStairsSteppingStonesTerrainCfg):
    """Configuration for inverted rugged block stairs."""

    inverted: bool = True
