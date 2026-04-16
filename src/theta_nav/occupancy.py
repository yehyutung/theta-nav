from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import MujocoNavWorld


@dataclass(frozen=True)
class SensorConfig:
    fov_rad: float = np.pi * 0.9
    max_range: float = 3.0
    num_rays: int = 72


class OccupancyGrid:
    """Occupancy grid values: 0 unknown, 1 free, -1 occupied."""

    def __init__(self, world_half_size: float, resolution: float = 0.1):
        self.resolution = float(resolution)
        self.half_size = float(world_half_size)
        self.grid_dim = int(np.ceil((2.0 * self.half_size) / self.resolution))
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int8)

    def reset(self) -> None:
        self.grid.fill(0)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.floor((x + self.half_size) / self.resolution))
        gy = int(np.floor((y + self.half_size) / self.resolution))
        gx = int(np.clip(gx, 0, self.grid_dim - 1))
        gy = int(np.clip(gy, 0, self.grid_dim - 1))
        return gx, gy

    def mark_disk(self, x: float, y: float, radius: float, value: int = 1) -> None:
        gx, gy = self.world_to_grid(x, y)
        r_cells = max(1, int(np.ceil(radius / self.resolution)))
        xmin = max(0, gx - r_cells)
        xmax = min(self.grid_dim - 1, gx + r_cells)
        ymin = max(0, gy - r_cells)
        ymax = min(self.grid_dim - 1, gy + r_cells)

        for ix in range(xmin, xmax + 1):
            for iy in range(ymin, ymax + 1):
                wx = (ix + 0.5) * self.resolution - self.half_size
                wy = (iy + 0.5) * self.resolution - self.half_size
                if (wx - x) ** 2 + (wy - y) ** 2 <= radius**2:
                    self.grid[ix, iy] = value

    def update_from_world(
        self,
        world: MujocoNavWorld,
        sensor_cfg: SensorConfig,
        mark_agent_free_radius: float = 0.15,
    ) -> None:
        x, y, yaw = world.pose
        self.mark_disk(x, y, radius=mark_agent_free_radius, value=1)

        ray_angles = np.linspace(-sensor_cfg.fov_rad / 2.0, sensor_cfg.fov_rad / 2.0, sensor_cfg.num_rays)
        for offset in ray_angles:
            hit_range = world.raycast_distance(x, y, yaw, offset, max_range=sensor_cfg.max_range)
            angle = yaw + offset

            for d in np.arange(0.0, hit_range, self.resolution):
                px = x + d * np.cos(angle)
                py = y + d * np.sin(angle)
                gx, gy = self.world_to_grid(px, py)
                self.grid[gx, gy] = 1

            if hit_range < sensor_cfg.max_range:
                ox = x + hit_range * np.cos(angle)
                oy = y + hit_range * np.sin(angle)
                gx, gy = self.world_to_grid(ox, oy)
                self.grid[gx, gy] = -1

    @property
    def explored_fraction(self) -> float:
        known = np.count_nonzero(self.grid != 0)
        return float(known) / float(self.grid.size)
