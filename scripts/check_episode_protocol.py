from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    EpisodeConfig,
    MujocoNavWorld,
    OccupancyGrid,
    SensorConfig,
    SinTurnPolicy,
    make_semi_cluttered_map,
    run_episode,
)


def main() -> None:
    out = ROOT / "artifacts" / "episode_protocol_check"
    out.mkdir(parents=True, exist_ok=True)

    world = MujocoNavWorld(obstacles=make_semi_cluttered_map(num_obstacles=10, seed=3))
    grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=0.1)
    sensor = SensorConfig()
    policy = SinTurnPolicy(forward_speed=1.0, turn_amp=0.6, turn_freq=0.03)
    ep_cfg = EpisodeConfig(steps=500, collision_penalty=0.1, step_penalty=0.0)

    result = run_episode(world, grid, policy, sensor, ep_cfg)

    occ = grid.grid.T
    extent = (-grid.half_size, grid.half_size, -grid.half_size, grid.half_size)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].imshow(
        occ,
        cmap="RdYlGn",
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        extent=extent,
    )
    points = np.column_stack([result.trail[:, 0], result.trail[:, 1]])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    color_progress = np.linspace(0.0, 1.0, len(segments))
    lc = LineCollection(segments, cmap="viridis", linewidths=0.6)
    lc.set_array(color_progress)
    axes[0, 0].add_collection(lc)
    axes[0, 0].scatter(points[0, 0], points[0, 1], color="limegreen", s=20, zorder=3)
    axes[0, 0].scatter(points[-1, 0], points[-1, 1], color="red", s=20, zorder=3)
    legend_handles = [
        Line2D([], [], color="black", lw=0, marker="s", markersize=7, markerfacecolor="darkred", label="occupied"),
        Line2D([], [], color="black", lw=0, marker="s", markersize=7, markerfacecolor="yellow", label="unknown"),
        Line2D([], [], color="black", lw=0, marker="s", markersize=7, markerfacecolor="darkgreen", label="free"),
        Line2D([], [], color=plt.get_cmap("viridis")(0.6), lw=1.0, label="trajectory (time gradient)"),
        Line2D([], [], color="limegreen", marker="o", lw=0, markersize=5, label="start"),
        Line2D([], [], color="red", marker="o", lw=0, markersize=5, label="end"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
    axes[0, 0].set_title("Occupancy + trajectory")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_xlim(-grid.half_size, grid.half_size)
    axes[0, 0].set_ylim(-grid.half_size, grid.half_size)
    axes[0, 0].set_aspect("equal", adjustable="box")
    # Tiny inset for quick alignment sanity-check against raw MuJoCo render.
    top_raw = world.render_topdown_view()
    ax_inset = inset_axes(axes[0, 0], width="28%", height="28%", loc="lower right", borderpad=0.8)
    ax_inset.imshow(top_raw)
    ax_inset.set_title("MuJoCo top", fontsize=7)
    ax_inset.axis("off")

    axes[0, 1].plot(result.explored_fraction_per_step, color="tab:green")
    axes[0, 1].set_title("Explored fraction over steps")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("explored fraction")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(result.collisions_per_step, color="tab:red")
    axes[1, 0].set_title("Collision events per step")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("collision (0/1)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(np.cumsum(result.rewards_per_step), color="tab:purple")
    axes[1, 1].set_title("Cumulative reward")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("sum reward")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "episode_protocol_summary.png", dpi=130)
    plt.close(fig)

    agent = world.render_agent_view()
    fig_top, ax_top = plt.subplots(figsize=(6, 6))
    ax_top.imshow(
        occ,
        cmap="RdYlGn",
        origin="lower",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        extent=extent,
    )
    lc_top = LineCollection(segments, cmap="viridis", linewidths=0.6)
    lc_top.set_array(color_progress)
    ax_top.add_collection(lc_top)
    ax_top.scatter(points[0, 0], points[0, 1], color="limegreen", s=26, zorder=3, label="start")
    ax_top.scatter(points[-1, 0], points[-1, 1], color="red", s=26, zorder=3, label="end")
    ax_top.set_title("Final occupancy and trajectory")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_xlim(-grid.half_size, grid.half_size)
    ax_top.set_ylim(-grid.half_size, grid.half_size)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig_top.tight_layout()
    fig_top.savefig(out / "episode_topdown_final.png", dpi=140)
    plt.close(fig_top)
    plt.imsave(out / "episode_agent_final.png", agent)

    print(f"final_explored_fraction={result.final_explored_fraction:.3f}")
    print(f"total_collisions={result.total_collisions}")
    print(f"total_reward={result.total_reward:.2f}")
    print(f"saved_artifacts={out}")


if __name__ == "__main__":
    main()
