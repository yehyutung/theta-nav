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

    def __post_init__(self) -> None:
        self._memory = np.zeros(self.num_angle_bins, dtype=np.float32)
        self._target_yaw = 0.0
        self._prefer_left = True

    def _bin_id(self, angle: float) -> int:
        a = _wrap_to_pi(angle)
        f = (a + np.pi) / (2.0 * np.pi)
        return int(np.clip(f * self.num_angle_bins, 0, self.num_angle_bins - 1))

    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        del t, grid
        x, y, yaw = world.pose
        self._memory *= self.decay
        self._memory[self._bin_id(yaw)] += 1.0

        base = yaw
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
        self._prefer_left = not self._prefer_left

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del t
        _, _, yaw = pose
        yaw_err = _wrap_to_pi(self._target_yaw - yaw)
        turn = float(np.clip(self.steering_kp * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        align = max(0.2, 1.0 - abs(yaw_err) / np.pi)
        return self.forward_speed * align, turn


@dataclass
class ThetaCycleSweepPolicy:
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
        align = max(0.45, 1.0 - abs(yaw_err) / (1.4 * np.pi))
        return self.forward_speed * align, turn


@dataclass
class VollanGreedySweepPolicy:
    forward_speed: float = 1.0
    turn_speed_limit: float = 1.2
    steering_kp: float = 2.0
    decay: float = 0.97
    num_angle_bins: int = 72
    num_candidates: int = 72
    overlap_weight: float = 1.0
    obstacle_weight: float = 0.6
    probe_range: float = 2.5
    movement_eps: float = 1e-6

    def __post_init__(self) -> None:
        self._memory = np.zeros(self.num_angle_bins, dtype=np.float32)
        self._target_yaw = 0.0
        self._last_pose: tuple[float, float, float] | None = None

    def _bin_id(self, angle: float) -> int:
        a = _wrap_to_pi(angle)
        f = (a + np.pi) / (2.0 * np.pi)
        return int(np.clip(f * self.num_angle_bins, 0, self.num_angle_bins - 1))

    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        del t, grid
        x, y, yaw = world.pose
        self._memory *= self.decay

        if self._last_pose is None:
            moved_heading = yaw
        else:
            x_prev, y_prev, yaw_prev = self._last_pose
            dx = x - x_prev
            dy = y - y_prev
            if dx * dx + dy * dy > self.movement_eps * self.movement_eps:
                moved_heading = float(np.arctan2(dy, dx))
            else:
                moved_heading = yaw_prev
        self._memory[self._bin_id(moved_heading)] += 1.0
        self._last_pose = (x, y, yaw)

        candidate_offsets = np.linspace(-np.pi, np.pi, self.num_candidates, endpoint=False)
        best_score = np.inf
        best_target = yaw
        for offset in candidate_offsets:
            cand = yaw + float(offset)
            overlap_cost = float(self._memory[self._bin_id(cand)])
            ray_dist = world.raycast_distance(x, y, yaw, float(offset), max_range=self.probe_range)
            obstacle_cost = 1.0 - (ray_dist / self.probe_range)
            score = self.overlap_weight * overlap_cost + self.obstacle_weight * obstacle_cost
            if score < best_score:
                best_score = score
                best_target = cand

        self._target_yaw = _wrap_to_pi(best_target)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del t
        _, _, yaw = pose
        yaw_err = _wrap_to_pi(self._target_yaw - yaw)
        turn = float(np.clip(self.steering_kp * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        align = max(0.25, 1.0 - abs(yaw_err) / np.pi)
        return self.forward_speed * align, turn


@dataclass
class VollanGreedySweepPolicyV2:
    """Movement-frame relative coverage trace with Gaussian spatial footprint.

    Core fix over VollanGreedySweepPolicy:
    1. Candidates are relative offsets from a smoothed travel direction (not absolute headings).
    2. Each chosen sweep stamps a Gaussian footprint across nearby bins in relative-offset
       space, approximating the 2D spatial footprint in Vollan 2025 (D001/D002). After
       sweeping left, the left angular region is elevated → greedy picks right → L-R
       alternation emerges without any explicit alternation rule.
    3. Candidates restricted to the forward half-plane (±sweep_range_deg of travel direction)
       because theta sweeps look ahead; backward sweeps are biologically excluded.
    4. Forward motion stamps a narrow forward-direction footprint each step, so the directly
       ahead region accumulates coverage, pushing the equilibrium sweep angle off-axis toward
       the biological ±30–40° sweet spot.
    """

    forward_speed: float = 1.0
    turn_speed_limit: float = 1.2
    steering_kp: float = 2.0
    travel_dir_smoothing: float = 0.10
    decay: float = 0.94
    num_candidates: int = 60
    num_memory_bins: int = 120
    overlap_weight: float = 1.8
    obstacle_weight: float = 0.2
    probe_range: float = 2.5
    sweep_range_deg: float = 90.0
    footprint_width_deg: float = 35.0
    forward_weight: float = 0.4
    forward_width_deg: float = 20.0
    dither_deg: float = 3.0
    sweep_hold_steps: int = 20
    seed: int = 0
    movement_eps: float = 1e-6

    def __post_init__(self) -> None:
        self._memory = np.zeros(self.num_memory_bins, dtype=np.float32)
        self._target_yaw = 0.0
        self._travel_dir: float | None = None
        self._last_pos: tuple[float, float] | None = None
        self._rng = np.random.default_rng(self.seed)
        # Precompute bin centers in relative-offset space [-π, π).
        self._bin_centers = np.linspace(-np.pi, np.pi, self.num_memory_bins, endpoint=False)
        # Candidates: uniformly spaced in the forward half-plane ±sweep_range_deg.
        r = np.deg2rad(self.sweep_range_deg)
        self._candidate_offsets = np.linspace(-r, r, self.num_candidates)
        self.last_chosen_offset: float = 0.0
        self.last_decision_step: bool = False
        # Initialise at sweep_hold_steps so the very first observe() triggers a decision.
        self._hold_counter: int = self.sweep_hold_steps

    def _gaussian_footprint(self, center: float, sigma_deg: float) -> np.ndarray:
        """Circular Gaussian footprint centered on `center` (radians) over bin_centers."""
        sigma = np.deg2rad(sigma_deg)
        diffs = np.arctan2(
            np.sin(self._bin_centers - center),
            np.cos(self._bin_centers - center),
        )
        return np.exp(-0.5 * (diffs / sigma) ** 2).astype(np.float32)

    def _coverage_at(self, offsets: np.ndarray) -> np.ndarray:
        """Read coverage memory at each offset (radians) via nearest-bin lookup."""
        bins = np.clip(
            ((offsets + np.pi) / (2.0 * np.pi) * self.num_memory_bins).astype(int),
            0, self.num_memory_bins - 1,
        )
        return self._memory[bins]

    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        del t, grid
        x, y, yaw = world.pose

        # Update travel direction via circular EMA of movement heading.
        if self._last_pos is not None:
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            if dx * dx + dy * dy > self.movement_eps ** 2:
                move_heading = float(np.arctan2(dy, dx))
                if self._travel_dir is None:
                    self._travel_dir = move_heading
                else:
                    err = _wrap_to_pi(move_heading - self._travel_dir)
                    self._travel_dir = _wrap_to_pi(self._travel_dir + self.travel_dir_smoothing * err)

        if self._travel_dir is None:
            self._travel_dir = yaw
        self._last_pos = (x, y)

        self._memory *= self.decay

        # Forward motion coverage: stamp ahead region every physics step so that the
        # directly-ahead sector accumulates coverage continuously as the agent moves.
        # This biases the equilibrium sweep angle off-axis toward ±30–40°.
        self._memory += self.forward_weight * self._gaussian_footprint(0.0, self.forward_width_deg)

        # Hold between decisions: only re-evaluate the greedy sweep every sweep_hold_steps
        # physics steps (biological analogue: one new sweep target per theta cycle).
        self._hold_counter += 1
        if self._hold_counter < self.sweep_hold_steps:
            self.last_decision_step = False
            return
        self._hold_counter = 0
        self.last_decision_step = True

        # Candidate offsets: restricted to forward half-plane (matching biological model
        # where theta sweeps project ahead, not backward).
        offsets = self._candidate_offsets
        coverage_costs = self._coverage_at(offsets)

        obstacle_costs = np.empty(len(offsets), dtype=np.float64)
        for i, offset in enumerate(offsets):
            cand_abs = self._travel_dir + offset
            ray_offset = _wrap_to_pi(cand_abs - yaw)
            ray_dist = world.raycast_distance(x, y, yaw, ray_offset, max_range=self.probe_range)
            obstacle_costs[i] = 1.0 - (ray_dist / self.probe_range)

        scores = self.overlap_weight * coverage_costs + self.obstacle_weight * obstacle_costs
        best_idx = int(np.argmin(scores))
        best_offset = float(offsets[best_idx])

        # Dither: ±half a direction bin, matching biological model (D003 ±3°).
        dither_rad = np.deg2rad(self.dither_deg)
        best_offset += float(self._rng.uniform(-dither_rad, dither_rad))
        self.last_chosen_offset = best_offset

        # Sweep footprint update: stamp wide Gaussian around the chosen offset.
        # After sweeping left (+30°), the left region [0°, +60°] is elevated →
        # greedy picks right (~-30°) → emergent L-R alternation.
        self._memory += self._gaussian_footprint(best_offset, self.footprint_width_deg)

        self._target_yaw = _wrap_to_pi(self._travel_dir + best_offset)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del t
        _, _, yaw = pose
        yaw_err = _wrap_to_pi(self._target_yaw - yaw)
        turn = float(np.clip(self.steering_kp * yaw_err, -self.turn_speed_limit, self.turn_speed_limit))
        align = max(0.25, 1.0 - abs(yaw_err) / np.pi)
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
