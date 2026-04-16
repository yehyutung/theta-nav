from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    MujocoNavWorld,
    OccupancyGrid,
    SensorConfig,
    make_open_map,
    make_semi_cluttered_map,
)


def run_episode(world: MujocoNavWorld, grid: OccupancyGrid, steps: int) -> np.ndarray:
    trail = np.empty((steps, 2), dtype=np.float32)
    sensor = SensorConfig()
    world.reset()
    grid.reset()

    for t in range(steps):
        forward = 1.0
        turn = 0.6 * np.sin(t * 0.03)
        x, y, _ = world.step(forward, turn)
        trail[t] = (x, y)
        grid.update_from_world(world, sensor_cfg=sensor, mark_agent_free_radius=0.2)
    return trail


def save_case(case_name: str, obstacles, output_dir: Path) -> None:
    world = MujocoNavWorld(obstacles=obstacles)
    grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=0.1)
    trail = run_episode(world, grid, steps=500)

    top = world.render_topdown_view()
    agent = world.render_agent_view()
    plt.imsave(output_dir / f"{case_name}_topdown.png", top)
    plt.imsave(output_dir / f"{case_name}_agent.png", agent)

    occ = grid.grid.T
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(occ, cmap="RdYlGn", origin="lower", vmin=-1, vmax=1)
    axes[0].set_title(f"{case_name}: occupancy")
    axes[0].axis("off")

    axes[1].imshow(occ, cmap="RdYlGn", origin="lower", vmin=-1, vmax=1)
    gx = (trail[:, 0] + grid.half_size) / grid.resolution
    gy = (trail[:, 1] + grid.half_size) / grid.resolution
    axes[1].plot(gx, gy, "b-", linewidth=0.8)
    axes[1].set_title(f"{case_name}: occupancy + trajectory")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{case_name}_occupancy.png", dpi=120)
    plt.close(fig)

    print(f"{case_name}: explored_fraction={grid.explored_fraction:.3f}")


def main() -> None:
    out = ROOT / "artifacts" / "mapping_check"
    out.mkdir(parents=True, exist_ok=True)

    save_case("open", make_open_map(), out)
    save_case("semi_cluttered", make_semi_cluttered_map(num_obstacles=10, seed=3), out)

    print(f"Saved artifacts under: {out}")
    print("- open_topdown.png, open_agent.png, open_occupancy.png")
    print("- semi_cluttered_topdown.png, semi_cluttered_agent.png, semi_cluttered_occupancy.png")


if __name__ == "__main__":
    main()
