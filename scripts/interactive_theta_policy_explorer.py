from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    EpisodeConfig,
    MapConfig,
    MujocoNavWorld,
    OccupancyGrid,
    SensorConfig,
    ThetaSweepPolicy,
    WorldConfig,
    make_open_map,
    make_semi_cluttered_map,
    run_episode,
)


def _build_obstacles(map_name: str, map_seed: int) -> list[tuple[float, float, float, float]]:
    if map_name == "open":
        return make_open_map()
    if map_name == "semi":
        cfg = MapConfig(room_half_size=5.0)
        return make_semi_cluttered_map(num_obstacles=10, seed=map_seed, cfg=cfg)
    raise ValueError(f"Unknown map name: {map_name}")


def _run_theta_case(params: dict, map_name: str, map_seed: int, steps: int) -> tuple:
    policy = ThetaSweepPolicy(**params)
    obstacles = _build_obstacles(map_name, map_seed)
    world = MujocoNavWorld(config=WorldConfig(room_half_size=5.0), obstacles=obstacles)
    grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=0.1)
    sensor = SensorConfig()
    episode_cfg = EpisodeConfig(steps=steps, collision_penalty=0.1)
    result = run_episode(world, grid, policy, sensor, episode_cfg)
    return result, grid, policy


def main() -> None:
    initial_params = {
        "forward_speed": 1.0,
        "turn_speed_limit": 1.2,
        "steering_kp": 2.0,
        "sweep_angle_deg": 30.0,
        "decay": 0.97,
        "num_angle_bins": 36,
        "overlap_weight": 1.0,
        "obstacle_weight": 0.8,
        "probe_range": 2.5,
    }
    initial_map = "semi"
    initial_seed = 7
    initial_steps = 600

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Theta Policy Parameter Explorer", fontsize=14)

    ax_occ = fig.add_axes([0.05, 0.38, 0.40, 0.56])
    ax_curve = fig.add_axes([0.48, 0.58, 0.47, 0.36])
    ax_mem = fig.add_axes([0.48, 0.38, 0.47, 0.16])
    ax_text = fig.add_axes([0.05, 0.25, 0.90, 0.10])
    ax_text.axis("off")

    slider_axes = {
        "forward_speed": fig.add_axes([0.08, 0.19, 0.27, 0.025]),
        "turn_speed_limit": fig.add_axes([0.08, 0.16, 0.27, 0.025]),
        "steering_kp": fig.add_axes([0.08, 0.13, 0.27, 0.025]),
        "sweep_angle_deg": fig.add_axes([0.08, 0.10, 0.27, 0.025]),
        "decay": fig.add_axes([0.08, 0.07, 0.27, 0.025]),
        "num_angle_bins": fig.add_axes([0.08, 0.04, 0.27, 0.025]),
        "overlap_weight": fig.add_axes([0.42, 0.19, 0.27, 0.025]),
        "obstacle_weight": fig.add_axes([0.42, 0.16, 0.27, 0.025]),
        "probe_range": fig.add_axes([0.42, 0.13, 0.27, 0.025]),
        "map_seed": fig.add_axes([0.42, 0.10, 0.27, 0.025]),
        "steps": fig.add_axes([0.42, 0.07, 0.27, 0.025]),
    }

    sliders = {
        "forward_speed": Slider(slider_axes["forward_speed"], "forward_speed", 0.2, 1.5, valinit=1.0),
        "turn_speed_limit": Slider(slider_axes["turn_speed_limit"], "turn_speed_limit", 0.3, 1.5, valinit=1.2),
        "steering_kp": Slider(slider_axes["steering_kp"], "steering_kp", 0.5, 5.0, valinit=2.0),
        "sweep_angle_deg": Slider(slider_axes["sweep_angle_deg"], "sweep_angle_deg", 5.0, 75.0, valinit=30.0),
        "decay": Slider(slider_axes["decay"], "decay", 0.80, 0.999, valinit=0.97),
        "num_angle_bins": Slider(
            slider_axes["num_angle_bins"], "num_angle_bins", 8, 72, valinit=36, valstep=1
        ),
        "overlap_weight": Slider(slider_axes["overlap_weight"], "overlap_weight", 0.0, 3.0, valinit=1.0),
        "obstacle_weight": Slider(slider_axes["obstacle_weight"], "obstacle_weight", 0.0, 3.0, valinit=0.8),
        "probe_range": Slider(slider_axes["probe_range"], "probe_range", 0.5, 4.5, valinit=2.5),
        "map_seed": Slider(slider_axes["map_seed"], "map_seed", 0, 50, valinit=7, valstep=1),
        "steps": Slider(slider_axes["steps"], "steps", 150, 1200, valinit=600, valstep=10),
    }

    ax_radio = fig.add_axes([0.74, 0.05, 0.12, 0.12])
    map_radio = RadioButtons(ax_radio, ("semi", "open"), active=0)
    ax_radio.set_title("map", fontsize=9)

    ax_run = fig.add_axes([0.88, 0.07, 0.08, 0.06])
    run_button = Button(ax_run, "Run")
    ax_reset = fig.add_axes([0.88, 0.145, 0.08, 0.04])
    reset_button = Button(ax_reset, "Reset")

    state = {"map_name": initial_map}

    def _collect_params() -> tuple[dict, int, int]:
        params = {
            "forward_speed": float(sliders["forward_speed"].val),
            "turn_speed_limit": float(sliders["turn_speed_limit"].val),
            "steering_kp": float(sliders["steering_kp"].val),
            "sweep_angle_deg": float(sliders["sweep_angle_deg"].val),
            "decay": float(sliders["decay"].val),
            "num_angle_bins": int(sliders["num_angle_bins"].val),
            "overlap_weight": float(sliders["overlap_weight"].val),
            "obstacle_weight": float(sliders["obstacle_weight"].val),
            "probe_range": float(sliders["probe_range"].val),
        }
        map_seed = int(sliders["map_seed"].val)
        steps = int(sliders["steps"].val)
        return params, map_seed, steps

    def _draw() -> None:
        params, map_seed, steps = _collect_params()
        result, grid, policy = _run_theta_case(params, state["map_name"], map_seed, steps)

        occ = grid.grid.T
        extent = (-grid.half_size, grid.half_size, -grid.half_size, grid.half_size)

        ax_occ.clear()
        ax_occ.imshow(
            occ,
            cmap="RdYlGn",
            origin="lower",
            vmin=-1,
            vmax=1,
            extent=extent,
            interpolation="nearest",
        )
        ax_occ.plot(result.trail[:, 0], result.trail[:, 1], linewidth=0.8, color="tab:blue")
        ax_occ.scatter(result.trail[0, 0], result.trail[0, 1], c="green", s=24, label="start")
        ax_occ.scatter(result.trail[-1, 0], result.trail[-1, 1], c="red", s=24, label="end")
        ax_occ.set_title(f"Occupancy + trajectory ({state['map_name']} map, seed={map_seed})")
        ax_occ.set_aspect("equal", adjustable="box")
        ax_occ.set_xlim(-grid.half_size, grid.half_size)
        ax_occ.set_ylim(-grid.half_size, grid.half_size)
        ax_occ.legend(loc="upper right", fontsize=8)

        ax_curve.clear()
        collision_rate = np.cumsum(result.collisions_per_step) / np.arange(1, len(result.collisions_per_step) + 1)
        ax_curve.plot(result.explored_fraction_per_step, color="tab:green", label="explored_fraction")
        ax_curve.plot(collision_rate, color="tab:red", label="collision_rate")
        ax_curve.set_ylim(0.0, 1.0)
        ax_curve.set_title("Exploration vs collision over time")
        ax_curve.set_xlabel("step")
        ax_curve.grid(alpha=0.3)
        ax_curve.legend(fontsize=8)

        ax_mem.clear()
        bins = np.arange(policy._memory.size)
        ax_mem.bar(bins, policy._memory, color="tab:purple", width=0.85)
        ax_mem.set_title("Final yaw-memory bins (recency-weighted heading usage)")
        ax_mem.set_xlabel("angle bin")
        ax_mem.grid(alpha=0.2, axis="y")

        ax_text.clear()
        ax_text.axis("off")
        efficiency = result.final_explored_fraction / max(1, result.total_collisions)
        text = (
            f"final_explored={result.final_explored_fraction:.3f} | "
            f"total_collisions={result.total_collisions:d} | "
            f"mean_collision_rate={float(np.mean(result.collisions_per_step)):.3f} | "
            f"total_reward={result.total_reward:.1f} | "
            f"efficiency(explored/collision)={efficiency:.3f}"
        )
        ax_text.text(0.01, 0.5, text, fontsize=11, va="center", ha="left")

        fig.canvas.draw_idle()

    def _on_run(_event) -> None:
        _draw()

    def _on_map_change(label: str) -> None:
        state["map_name"] = label

    def _on_reset(_event) -> None:
        defaults = {
            "forward_speed": 1.0,
            "turn_speed_limit": 1.2,
            "steering_kp": 2.0,
            "sweep_angle_deg": 30.0,
            "decay": 0.97,
            "num_angle_bins": 36,
            "overlap_weight": 1.0,
            "obstacle_weight": 0.8,
            "probe_range": 2.5,
            "map_seed": 7,
            "steps": 600,
        }
        for k, v in defaults.items():
            sliders[k].set_val(v)
        map_radio.set_active(0)
        _draw()

    run_button.on_clicked(_on_run)
    reset_button.on_clicked(_on_reset)
    map_radio.on_clicked(_on_map_change)

    _draw()
    plt.show()


if __name__ == "__main__":
    main()
