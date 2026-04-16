from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import Obstacle


@dataclass(frozen=True)
class MapConfig:
    room_half_size: float = 5.0
    obstacle_min_half_size: float = 0.25
    obstacle_max_half_size: float = 0.8
    obstacle_margin_to_wall: float = 0.8
    obstacle_margin_to_origin: float = 1.0


def make_open_map() -> list[Obstacle]:
    return []


def make_semi_cluttered_map(
    num_obstacles: int = 10,
    seed: int = 0,
    cfg: MapConfig | None = None,
) -> list[Obstacle]:
    cfg = cfg or MapConfig()
    rng = np.random.default_rng(seed)
    obstacles: list[Obstacle] = []

    min_c = -cfg.room_half_size + cfg.obstacle_margin_to_wall
    max_c = cfg.room_half_size - cfg.obstacle_margin_to_wall

    trials = 0
    max_trials = num_obstacles * 50
    while len(obstacles) < num_obstacles and trials < max_trials:
        trials += 1
        sx = float(rng.uniform(cfg.obstacle_min_half_size, cfg.obstacle_max_half_size))
        sy = float(rng.uniform(cfg.obstacle_min_half_size, cfg.obstacle_max_half_size))
        x = float(rng.uniform(min_c, max_c))
        y = float(rng.uniform(min_c, max_c))

        if np.hypot(x, y) < cfg.obstacle_margin_to_origin:
            continue

        if _overlaps_any(x, y, sx, sy, obstacles, padding=0.2):
            continue

        obstacles.append((x, y, sx, sy))

    return obstacles


def _overlaps_any(
    x: float,
    y: float,
    sx: float,
    sy: float,
    others: list[Obstacle],
    padding: float,
) -> bool:
    for ox, oy, osx, osy in others:
        x_overlap = abs(x - ox) < (sx + osx + padding)
        y_overlap = abs(y - oy) < (sy + osy + padding)
        if x_overlap and y_overlap:
            return True
    return False
