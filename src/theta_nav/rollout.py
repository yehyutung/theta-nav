from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .occupancy import OccupancyGrid, SensorConfig
from .world import MujocoNavWorld


class Policy(Protocol):
    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        ...


class ContextPolicy(Protocol):
    def observe(self, t: int, world: MujocoNavWorld, grid: OccupancyGrid) -> None:
        ...

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        ...


class SinTurnPolicy:
    """Simple deterministic baseline policy for smoke tests."""

    def __init__(self, forward_speed: float = 1.0, turn_amp: float = 0.6, turn_freq: float = 0.03):
        self.forward_speed = float(forward_speed)
        self.turn_amp = float(turn_amp)
        self.turn_freq = float(turn_freq)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del pose
        return self.forward_speed, self.turn_amp * np.sin(t * self.turn_freq)


@dataclass(frozen=True)
class EpisodeConfig:
    steps: int = 500
    collision_penalty: float = 0.0
    step_penalty: float = 0.0
    mark_agent_free_radius: float = 0.2
    intrinsic_overlap_weight: float = 0.0
    intrinsic_overlap_decay: float = 0.97
    intrinsic_num_angle_bins: int = 36
    intrinsic_movement_eps: float = 1e-6


@dataclass
class EpisodeResult:
    trail: np.ndarray
    collisions_per_step: np.ndarray
    contact_count_per_step: np.ndarray
    rewards_per_step: np.ndarray
    intrinsic_rewards_per_step: np.ndarray
    explored_fraction_per_step: np.ndarray
    total_collisions: int
    total_reward: float
    final_explored_fraction: float


def run_episode(
    world: MujocoNavWorld,
    grid: OccupancyGrid,
    policy: Policy,
    sensor_cfg: SensorConfig,
    episode_cfg: EpisodeConfig,
) -> EpisodeResult:
    world.reset()
    grid.reset()
    grid.update_from_world(
        world,
        sensor_cfg=sensor_cfg,
        mark_agent_free_radius=episode_cfg.mark_agent_free_radius,
    )

    steps = episode_cfg.steps
    trail = np.empty((steps, 2), dtype=np.float32)
    collisions = np.zeros(steps, dtype=np.int32)
    contact_counts = np.zeros(steps, dtype=np.int32)
    rewards = np.zeros(steps, dtype=np.float32)
    intrinsic_rewards = np.zeros(steps, dtype=np.float32)
    explored = np.zeros(steps, dtype=np.float32)
    heading_memory = np.zeros(max(1, int(episode_cfg.intrinsic_num_angle_bins)), dtype=np.float32)

    def _bin_id(angle: float) -> int:
        wrapped = float(np.arctan2(np.sin(angle), np.cos(angle)))
        f = (wrapped + np.pi) / (2.0 * np.pi)
        n = heading_memory.size
        return int(np.clip(f * n, 0, n - 1))

    x_prev, y_prev, yaw_prev = world.pose

    prev_known = 0
    for t in range(steps):
        if hasattr(policy, "observe"):
            policy.observe(t, world, grid)
        pose = world.pose
        forward, turn = policy.action(t, pose)
        x, y, _ = world.step(forward, turn)
        trail[t] = (x, y)

        in_collision = 1 if world.is_in_collision else 0
        collisions[t] = in_collision
        contact_counts[t] = world.num_contacts

        grid.update_from_world(
            world,
            sensor_cfg=sensor_cfg,
            mark_agent_free_radius=episode_cfg.mark_agent_free_radius,
        )
        explored[t] = grid.explored_fraction

        known_now = int(np.count_nonzero(grid.grid != 0))
        info_gain = float(known_now - prev_known)
        prev_known = known_now
        extrinsic_reward = info_gain - episode_cfg.collision_penalty * in_collision - episode_cfg.step_penalty

        # Intrinsic objective: penalize overlap with recency-weighted directional trace.
        # The overlap is measured on the heading bin of the direction actually moved.
        dx = x - x_prev
        dy = y - y_prev
        disp2 = float(dx * dx + dy * dy)
        if disp2 > episode_cfg.intrinsic_movement_eps**2:
            move_heading = float(np.arctan2(dy, dx))
        else:
            move_heading = yaw_prev
        move_bin = _bin_id(move_heading)
        overlap_cost = float(heading_memory[move_bin])
        intrinsic_reward = -episode_cfg.intrinsic_overlap_weight * overlap_cost
        intrinsic_rewards[t] = intrinsic_reward

        rewards[t] = extrinsic_reward + intrinsic_reward

        heading_memory *= episode_cfg.intrinsic_overlap_decay
        heading_memory[move_bin] += 1.0
        x_prev, y_prev, yaw_prev = x, y, world.pose[2]

    return EpisodeResult(
        trail=trail,
        collisions_per_step=collisions,
        contact_count_per_step=contact_counts,
        rewards_per_step=rewards,
        intrinsic_rewards_per_step=intrinsic_rewards,
        explored_fraction_per_step=explored,
        total_collisions=int(np.sum(collisions)),
        total_reward=float(np.sum(rewards)),
        final_explored_fraction=float(explored[-1]),
    )
