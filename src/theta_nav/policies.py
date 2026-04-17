from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .occupancy import OccupancyGrid
    from .world import MujocoNavWorld
else:
    OccupancyGrid = Any
    MujocoNavWorld = Any


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class LevyWalkPolicy:
    forward_speed: float = 1.0
    turn_speed_limit: float = 1.2
    alpha: float = 1.5
    min_segment_steps: int = 15
    max_segment_steps: int = 120
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._remaining = 0
        self._target_yaw = 0.0

    def _sample_segment_length(self) -> int:
        # Pareto-like heavy-tailed segment lengths.
        raw = self.rng.pareto(self.alpha) + 1.0
        span = self.max_segment_steps - self.min_segment_steps
        steps = self.min_segment_steps + int(np.clip(raw / 6.0, 0.0, 1.0) * span)
        return int(np.clip(steps, self.min_segment_steps, self.max_segment_steps))

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del t
        _, _, yaw = pose
        if self._remaining <= 0:
            self._target_yaw = float(self.rng.uniform(-np.pi, np.pi))
            self._remaining = self._sample_segment_length()

        yaw_err = _wrap_to_pi(self._target_yaw - yaw)
        turn = float(np.clip(2.0 * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        self._remaining -= 1
        return self.forward_speed, turn


@dataclass
class FrontierGreedyPolicy:
    forward_speed: float = 1.0
    turn_speed_limit: float = 1.2
    steering_kp: float = 2.0
    fallback_turn_amp: float = 0.5
    fallback_turn_freq: float = 0.05

    def __post_init__(self) -> None:
        self._target_xy: tuple[float, float] | None = None

    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        del t
        x, y, _ = world.pose
        self._target_xy = _nearest_frontier_target(grid, x, y)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        x, y, yaw = pose
        if self._target_xy is None:
            return self.forward_speed, self.fallback_turn_amp * np.sin(t * self.fallback_turn_freq)

        tx, ty = self._target_xy
        desired = float(np.arctan2(ty - y, tx - x))
        yaw_err = _wrap_to_pi(desired - yaw)
        turn = float(np.clip(self.steering_kp * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        align = max(0.1, 1.0 - abs(yaw_err) / np.pi)
        return self.forward_speed * align, turn


@dataclass
class ThetaSweepPolicy:
    forward_speed: float = 1.0
    turn_speed_limit: float = 1.2
    steering_kp: float = 2.0
    sweep_angle_deg: float = 30.0
    decay: float = 0.97
    num_angle_bins: int = 36
    overlap_weight: float = 1.0
    obstacle_weight: float = 0.8
    probe_range: float = 2.5
    sweep_side_hold_steps: int = 24
    center_precession_deg: float = 14.0

    def __post_init__(self) -> None:
        self._memory = np.zeros(self.num_angle_bins, dtype=np.float32)
        self._target_yaw = 0.0
        self._prefer_left = True
        self._side_steps_remaining = max(1, int(self.sweep_side_hold_steps))
        self._center_yaw: float | None = None

    def _bin_id(self, angle: float) -> int:
        a = _wrap_to_pi(angle)
        f = (a + np.pi) / (2.0 * np.pi)
        return int(np.clip(f * self.num_angle_bins, 0, self.num_angle_bins - 1))

    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        del t, grid
        x, y, yaw = world.pose
        self._memory *= self.decay
        self._memory[self._bin_id(yaw)] += 1.0

        if self._center_yaw is None:
            self._center_yaw = yaw

        if self._side_steps_remaining <= 0:
            self._prefer_left = not self._prefer_left
            self._side_steps_remaining = max(1, int(self.sweep_side_hold_steps))
            if self._prefer_left:
                # Advance center once per full left-right cycle to avoid narrow oscillations.
                self._center_yaw = _wrap_to_pi(self._center_yaw + np.deg2rad(self.center_precession_deg))
        self._side_steps_remaining -= 1

        base = self._center_yaw
        d = np.deg2rad(self.sweep_angle_deg)
        candidate_offsets = np.array([-2 * d, -d, -0.5 * d, 0.5 * d, d, 2 * d], dtype=np.float64)

        if self._prefer_left:
            side_mask = candidate_offsets > 0
        else:
            side_mask = candidate_offsets < 0
        if np.any(side_mask):
            candidate_offsets = candidate_offsets[side_mask]

        best_score = np.inf
        best_target = base
        for offset in candidate_offsets:
            cand = base + float(offset)
            memory_cost = self._memory[self._bin_id(cand)]
            ray_dist = world.raycast_distance(x, y, yaw, float(offset), max_range=self.probe_range)
            obstacle_cost = 1.0 - (ray_dist / self.probe_range)
            score = self.overlap_weight * memory_cost + self.obstacle_weight * obstacle_cost
            if score < best_score:
                best_score = score
                best_target = cand

        self._target_yaw = _wrap_to_pi(best_target)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del t
        _, _, yaw = pose
        yaw_err = _wrap_to_pi(self._target_yaw - yaw)
        turn = float(np.clip(self.steering_kp * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        # Keep translation active during sweeps so the path remains space-filling.
        align = max(0.45, 1.0 - abs(yaw_err) / (1.4 * np.pi))
        return self.forward_speed * align, turn


def _nearest_frontier_target(grid: OccupancyGrid, ref_x: float, ref_y: float) -> tuple[float, float] | None:
    g = grid.grid
    free = g == 1
    unknown = g == 0

    frontier = np.zeros_like(free, dtype=bool)
    # 4-neighborhood frontier: free cell adjacent to unknown.
    frontier[1:, :] |= free[1:, :] & unknown[:-1, :]
    frontier[:-1, :] |= free[:-1, :] & unknown[1:, :]
    frontier[:, 1:] |= free[:, 1:] & unknown[:, :-1]
    frontier[:, :-1] |= free[:, :-1] & unknown[:, 1:]

    coords = np.argwhere(frontier)
    if coords.size == 0:
        return None

    center = np.array(grid.world_to_grid(ref_x, ref_y))
    d2 = np.sum((coords - center[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    gx, gy = coords[idx]
    wx = (gx + 0.5) * grid.resolution - grid.half_size
    wy = (gy + 0.5) * grid.resolution - grid.half_size
    return float(wx), float(wy)
