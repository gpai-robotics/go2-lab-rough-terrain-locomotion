"""Custom trimesh terrains for TRAKR."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


def rising_random_grid_terrain(
    difficulty: float, cfg: MeshRisingRandomGridTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a random grid terrain whose cell elevation rises along x."""

    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")

    grid_height = cfg.grid_height_range[0] + difficulty * (
        cfg.grid_height_range[1] - cfg.grid_height_range[0]
    )
    slope_height = cfg.slope_height_range[0] + difficulty * (
        cfg.slope_height_range[1] - cfg.slope_height_range[0]
    )

    meshes_list = []
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    terrain_height = 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        meshes_list += make_border(cfg.size, border_inner_size, terrain_height, border_center)
    else:
        raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")

    grid_dim = [cfg.grid_width, cfg.grid_width, terrain_height]
    grid_position = [0.5 * cfg.grid_width, 0.5 * cfg.grid_width, -terrain_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    template_vertices = template_box.vertices
    template_faces = template_box.faces

    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    x = torch.arange(0, num_boxes_x, device=device)
    y = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)

    offsets = cfg.grid_width * xx_yy + border_width / 2
    vertices[:, :, :2] += offsets.unsqueeze(1)

    num_boxes = len(vertices)
    x_indices = xx.flatten().to(torch.float32)

    if num_boxes_x > 1:
        center = (num_boxes_x - 1) / 2.0

        # distance from center
        distance = torch.abs(x_indices - center)

        # normalize:
        # edges -> 0
        # center -> 1
        x_progress = 1.0 - (distance / center)
        x_progress = torch.clamp(x_progress, 0.0, 1.0)
    else:
        x_progress = torch.ones_like(x_indices)

    climb = x_progress * slope_height

    random_height = torch.empty(num_boxes, device=device).uniform_(-grid_height, grid_height)
    lateral_variation = torch.empty(num_boxes_y, device=device).uniform_(-0.25 * grid_height, 0.25 * grid_height)
    lateral_height = lateral_variation[yy.flatten()]
    top_height = climb + random_height + lateral_height
    if cfg.inverted:
        top_height *= -1.0

    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise[:, :, 2] = top_height.unsqueeze(1)
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces = faces.view(-1, 3).cpu().numpy()

    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    origin_height = slope_height if not cfg.inverted else -slope_height
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_height])
    return meshes_list, origin


@configclass
class MeshRisingRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain with increasing elevation."""

    function: Callable = rising_random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""

    grid_height_range: tuple[float, float] = MISSING
    """The random height range around each cell's rising baseline (in m)."""

    slope_height_range: tuple[float, float] = MISSING
    """The total elevation gain across the terrain along x (in m)."""

    inverted: bool = False
    """Whether the terrain descends along x instead of ascending."""


@configclass
class MeshInvertedRisingRandomGridTerrainCfg(MeshRisingRandomGridTerrainCfg):
    """Configuration for descending random grid mesh terrain."""

    inverted: bool = True
