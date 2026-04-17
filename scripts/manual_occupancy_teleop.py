from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    MapConfig,
    MujocoNavWorld,
    OccupancyGrid,
    SensorConfig,
    make_open_map,
    make_semi_cluttered_map,
)


@dataclass
class TeleopState:
    pressed: set[str] = field(default_factory=set)
    paused: bool = False


def build_obstacles(map_name: str, seed: int, num_obstacles: int, room_half_size: float):
    if map_name == "open":
        return make_open_map()
    if map_name == "semi":
        cfg = MapConfig(room_half_size=room_half_size)
        return make_semi_cluttered_map(num_obstacles=num_obstacles, seed=seed, cfg=cfg)
    raise ValueError(f"Unsupported map '{map_name}'. Use 'open' or 'semi'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual teleop + live occupancy mapping.")
    parser.add_argument("--map", type=str, default="semi", choices=["open", "semi"])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--room-half-size", type=float, default=5.0)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--fov-rad", type=float, default=float(np.pi * 0.9))
    parser.add_argument("--max-range", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=72)
    parser.add_argument("--forward-speed", type=float, default=1.0)
    parser.add_argument("--turn-speed", type=float, default=1.2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    obstacles = build_obstacles(args.map, args.seed, args.num_obstacles, args.room_half_size)
    world = MujocoNavWorld(obstacles=obstacles)
    grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=args.resolution)
    sensor = SensorConfig(fov_rad=args.fov_rad, max_range=args.max_range, num_rays=args.num_rays)
    state = TeleopState()

    world.reset()
    grid.reset()
    grid.update_from_world(world, sensor_cfg=sensor, mark_agent_free_radius=0.2)
    trail: list[tuple[float, float]] = [world.pose[:2]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top_ax, occ_ax = axes
    fig.suptitle(
        "Manual mapping teleop (arrows or I/J/K/L move, space pause, R reset, Esc/Q quit)"
    )

    top_img = top_ax.imshow(world.render_topdown_view())
    top_ax.set_title("MuJoCo top-down")
    top_ax.axis("off")

    extent = (-grid.half_size, grid.half_size, -grid.half_size, grid.half_size)
    occ_img = occ_ax.imshow(
        grid.grid.T,
        cmap="RdYlGn",
        origin="lower",
        vmin=-1,
        vmax=1,
        extent=extent,
        interpolation="nearest",
    )
    (trail_line,) = occ_ax.plot([trail[0][0]], [trail[0][1]], color="tab:blue", linewidth=0.8, label="trajectory")
    start_scatter = occ_ax.scatter(trail[0][0], trail[0][1], c="limegreen", s=24, zorder=3, label="start")
    end_scatter = occ_ax.scatter(trail[0][0], trail[0][1], c="red", s=24, zorder=3, label="current")
    occ_ax.set_title("Occupancy grid (unknown/free/occupied)")
    occ_ax.set_xlabel("x (m)")
    occ_ax.set_ylabel("y (m)")
    occ_ax.set_xlim(-grid.half_size, grid.half_size)
    occ_ax.set_ylim(-grid.half_size, grid.half_size)
    occ_ax.set_aspect("equal", adjustable="box")
    occ_ax.legend(loc="upper left", fontsize=8)

    status_text = occ_ax.text(
        0.02,
        0.02,
        "",
        transform=occ_ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    def reset_world() -> None:
        world.reset()
        grid.reset()
        grid.update_from_world(world, sensor_cfg=sensor, mark_agent_free_radius=0.2)
        trail.clear()
        trail.append(world.pose[:2])

    def on_key_press(event):
        key = (event.key or "").lower()
        if key in {"up", "down", "left", "right", "i", "j", "k", "l"}:
            state.pressed.add(key)
        elif key == " ":
            state.paused = not state.paused
        elif key == "r":
            reset_world()
        elif key in {"escape", "q"}:
            plt.close(fig)

    def on_key_release(event):
        key = (event.key or "").lower()
        if key in state.pressed:
            state.pressed.remove(key)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    def current_command() -> tuple[float, float]:
        forward = 0.0
        turn = 0.0
        if "up" in state.pressed or "i" in state.pressed:
            forward += args.forward_speed
        if "down" in state.pressed or "k" in state.pressed:
            forward -= args.forward_speed
        if "left" in state.pressed or "j" in state.pressed:
            turn += args.turn_speed
        if "right" in state.pressed or "l" in state.pressed:
            turn -= args.turn_speed
        return forward, turn

    def update(_frame_idx: int):
        if not state.paused:
            forward, turn = current_command()
            world.step(forward_speed=forward, turn_speed=turn)
            grid.update_from_world(world, sensor_cfg=sensor, mark_agent_free_radius=0.2)
            trail.append(world.pose[:2])
        else:
            forward, turn = 0.0, 0.0

        top_img.set_data(world.render_topdown_view())
        occ_img.set_data(grid.grid.T)

        trail_arr = np.array(trail, dtype=np.float32)
        trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        end_scatter.set_offsets([[trail_arr[-1, 0], trail_arr[-1, 1]]])
        start_scatter.set_offsets([[trail_arr[0, 0], trail_arr[0, 1]]])

        x, y, yaw = world.pose
        status_text.set_text(
            f"pose=({x:.2f}, {y:.2f}, {yaw:.2f})\n"
            f"cmd=(fwd {forward:.2f}, turn {turn:.2f})\n"
            f"contacts={world.num_contacts}, explored={grid.explored_fraction:.3f}"
        )
        return top_img, occ_img, trail_line, end_scatter, start_scatter, status_text

    interval_ms = int(1000 / max(1, args.fps))
    _anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
