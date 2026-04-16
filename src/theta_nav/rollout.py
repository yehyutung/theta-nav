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


@dataclass
class EpisodeResult:
    trail: np.ndarray
    collisions_per_step: np.ndarray
    contact_count_per_step: np.ndarray
    rewards_per_step: np.ndarray
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
    explored = np.zeros(steps, dtype=np.float32)

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
        rewards[t] = info_gain - episode_cfg.collision_penalty * in_collision - episode_cfg.step_penalty

    return EpisodeResult(
        trail=trail,
        collisions_per_step=collisions,
        contact_count_per_step=contact_counts,
        rewards_per_step=rewards,
        explored_fraction_per_step=explored,
        total_collisions=int(np.sum(collisions)),
        total_reward=float(np.sum(rewards)),
        final_explored_fraction=float(explored[-1]),
    )
