from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav.policies import LevyWalkPolicy, ThetaSweepPolicy
from theta_nav.world import WorldConfig, build_world_xml


class DummyGrid:
    """ThetaSweepPolicy requires a grid in observe(); this test does not."""


class HeadlessMujocoWorld:
    """MuJoCo world without renderer dependency for batch diagnostics."""

    def __init__(self, config: WorldConfig, obstacles: list[tuple[float, float, float, float]] | None = None):
        self.config = config
        self.model = mujoco.MjModel.from_xml_string(build_world_xml(config, obstacles or []))
        self.data = mujoco.MjData(self.model)

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> tuple[float, float, float]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = yaw
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.pose

    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])

    def step(self, forward_speed: float, turn_speed: float) -> tuple[float, float, float]:
        x, y, yaw = self.pose
        fwd = float(np.clip(forward_speed, -self.config.max_linear_speed, self.config.max_linear_speed))
        turn = float(np.clip(turn_speed, -self.config.max_turn_speed, self.config.max_turn_speed))
        self.data.ctrl[0] = fwd * np.cos(yaw)
        self.data.ctrl[1] = fwd * np.sin(yaw)
        self.data.ctrl[2] = turn
        mujoco.mj_step(self.model, self.data)
        return self.pose

    def raycast_distance(
        self,
        x: float,
        y: float,
        yaw: float,
        angle_offset: float,
        max_range: float = 3.0,
    ) -> float:
        angle = yaw + angle_offset
        origin = np.array([x, y, 0.2], dtype=np.float64)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, -1, geom_id)
        if distance <= 0.0:
            return max_range
        return float(min(distance, max_range))


class StraightThenTurnPolicy:
    def __init__(self, forward_speed: float = 1.0, turn_speed: float = 0.7, turn_period: int = 120):
        self.forward_speed = float(forward_speed)
        self.turn_speed = float(turn_speed)
        self.turn_period = int(turn_period)

    def action(self, t: int, pose: tuple[float, float, float]) -> tuple[float, float]:
        del pose
        if (t // self.turn_period) % 2 == 0:
            return self.forward_speed, 0.0
        return self.forward_speed, self.turn_speed


def _simulate_policy(policy, steps: int, room_half_size: float = 5.0) -> dict[str, np.ndarray]:
    world = HeadlessMujocoWorld(config=WorldConfig(room_half_size=room_half_size, timestep=0.02), obstacles=[])
    world.reset(0.0, 0.0, 0.0)
    grid = DummyGrid()

    trail = np.zeros((steps, 2), dtype=np.float64)
    heading = np.zeros(steps, dtype=np.float64)
    turn_cmd = np.zeros(steps, dtype=np.float64)
    target_heading = np.zeros(steps, dtype=np.float64)

    for t in range(steps):
        if hasattr(policy, "observe"):
            policy.observe(t, world, grid)
        pose = world.pose
        forward, turn = policy.action(t, pose)
        x, y, yaw = world.step(forward, turn)
        trail[t] = (x, y)
        heading[t] = yaw
        turn_cmd[t] = turn
        target_heading[t] = getattr(policy, "_target_yaw", yaw)

    return {
        "trail": trail,
        "heading": np.arctan2(np.sin(heading), np.cos(heading)),
        "turn_cmd": turn_cmd,
        "target_heading": np.arctan2(np.sin(target_heading), np.cos(target_heading)),
    }


def _heading_hist_uniformity(heading: np.ndarray, bins: int = 36) -> tuple[np.ndarray, np.ndarray, float]:
    counts, edges = np.histogram(heading, bins=bins, range=(-np.pi, np.pi), density=False)
    probs = counts / max(1, counts.sum())
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    entropy_ratio = float(entropy / np.log(bins))
    return counts, edges, entropy_ratio


def _turn_sign_alternation_ratio(turn_cmd: np.ndarray, threshold: float = 0.05) -> float:
    significant = turn_cmd[np.abs(turn_cmd) > threshold]
    if significant.size < 2:
        return 0.0
    signs = np.sign(significant)
    flips = np.sum(signs[1:] * signs[:-1] < 0)
    return float(flips / (significant.size - 1))


def _coverage_ratio(trail: np.ndarray, room_half_size: float = 5.0, cells: int = 60) -> float:
    x = np.clip(trail[:, 0], -room_half_size, room_half_size)
    y = np.clip(trail[:, 1], -room_half_size, room_half_size)
    ix = np.clip(((x + room_half_size) / (2 * room_half_size) * cells).astype(np.int32), 0, cells - 1)
    iy = np.clip(((y + room_half_size) / (2 * room_half_size) * cells).astype(np.int32), 0, cells - 1)
    visited = np.zeros((cells, cells), dtype=bool)
    visited[ix, iy] = True
    return float(np.mean(visited))


def _plot_behavior(
    theta_data: dict[str, np.ndarray],
    levy_data: dict[str, np.ndarray],
    straight_data: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    ax0 = axes[0]
    ax0.plot(np.rad2deg(theta_data["heading"]), color="tab:blue", linewidth=1.1, label="theta heading")
    ax0.plot(
        np.rad2deg(theta_data["target_heading"]),
        color="tab:cyan",
        linewidth=0.9,
        alpha=0.75,
        label="theta target heading",
    )
    ax0.set_title("ThetaSweep heading over time (expect alternating zigzag sweeps)")
    ax0.set_xlabel("step")
    ax0.set_ylabel("heading (deg)")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8, loc="upper right")

    ax1 = axes[1]
    theta_counts, edges, theta_uniformity = _heading_hist_uniformity(theta_data["heading"])
    levy_counts, _, levy_uniformity = _heading_hist_uniformity(levy_data["heading"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.9
    ax1.bar(centers, theta_counts, width=width, color="tab:blue", alpha=0.7, label="theta")
    ax1.plot(centers, levy_counts, color="tab:orange", linewidth=1.5, label="random-walk-like baseline")
    ax1.set_title(
        f"Heading histogram (entropy ratio: theta={theta_uniformity:.3f}, random={levy_uniformity:.3f})"
    )
    ax1.set_xlabel("heading (rad)")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8, loc="upper right")

    ax2 = axes[2]
    ax2.plot(theta_data["trail"][:, 0], theta_data["trail"][:, 1], color="tab:blue", linewidth=1.2, label="theta")
    ax2.plot(
        levy_data["trail"][:, 0],
        levy_data["trail"][:, 1],
        color="tab:orange",
        linewidth=0.9,
        alpha=0.6,
        label="random-walk-like baseline",
    )
    ax2.plot(
        straight_data["trail"][:, 0],
        straight_data["trail"][:, 1],
        color="tab:green",
        linewidth=0.9,
        alpha=0.6,
        label="straight-then-turn baseline",
    )
    ax2.scatter(theta_data["trail"][0, 0], theta_data["trail"][0, 1], c="green", s=20, label="start")
    ax2.scatter(theta_data["trail"][-1, 0], theta_data["trail"][-1, 1], c="red", s=20, label="end")
    ax2.set_title("Open-arena trajectory (theta should look space-filling and distinct)")
    ax2.set_xlim(-5.0, 5.0)
    ax2.set_ylim(-5.0, 5.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MuJoCo-backed ThetaSweepPolicy behavior diagnostics.")
    parser.add_argument("--steps", type=int, default=1400, help="Number of sim steps.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random-walk baseline.")
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "theta_behavior_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "theta_behavior_diagnostics.png"

    theta_policy = ThetaSweepPolicy()
    levy_policy = LevyWalkPolicy(seed=args.seed)
    straight_policy = StraightThenTurnPolicy()

    theta_data = _simulate_policy(theta_policy, args.steps)
    levy_data = _simulate_policy(levy_policy, args.steps)
    straight_data = _simulate_policy(straight_policy, args.steps)

    _plot_behavior(theta_data, levy_data, straight_data, out_path)

    theta_uniformity = _heading_hist_uniformity(theta_data["heading"])[2]
    levy_uniformity = _heading_hist_uniformity(levy_data["heading"])[2]
    theta_alternation = _turn_sign_alternation_ratio(theta_data["turn_cmd"])
    levy_alternation = _turn_sign_alternation_ratio(levy_data["turn_cmd"])
    theta_coverage = _coverage_ratio(theta_data["trail"])
    levy_coverage = _coverage_ratio(levy_data["trail"])
    straight_coverage = _coverage_ratio(straight_data["trail"])

    print(f"theta_turn_alternation={theta_alternation:.3f}")
    print(f"random_turn_alternation={levy_alternation:.3f}")
    print(f"theta_heading_uniformity={theta_uniformity:.3f}")
    print(f"random_heading_uniformity={levy_uniformity:.3f}")
    print(f"theta_coverage={theta_coverage:.3f}")
    print(f"random_coverage={levy_coverage:.3f}")
    print(f"straight_then_turn_coverage={straight_coverage:.3f}")
    print(f"saved_artifact={out_path}")


if __name__ == "__main__":
    main()
