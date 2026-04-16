from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    EpisodeConfig,
    FrontierGreedyPolicy,
    LevyWalkPolicy,
    MujocoNavWorld,
    OccupancyGrid,
    SensorConfig,
    ThetaSweepPolicy,
    make_semi_cluttered_map,
    run_episode,
)


def _run_case(policy_obj, obstacles):
    world = MujocoNavWorld(obstacles=obstacles)
    grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=0.1)
    sensor = SensorConfig()
    ep_cfg = EpisodeConfig(steps=600, collision_penalty=0.1)
    result = run_episode(world, grid, policy_obj, sensor, ep_cfg)
    return result, grid


def main() -> None:
    out = ROOT / "artifacts" / "policies_check"
    out.mkdir(parents=True, exist_ok=True)

    obstacles = make_semi_cluttered_map(num_obstacles=10, seed=7)
    policies = [
        ("theta", ThetaSweepPolicy()),
        ("frontier", FrontierGreedyPolicy()),
        ("levy", LevyWalkPolicy(seed=7)),
    ]

    results = []
    for name, policy in policies:
        result, grid = _run_case(policy, obstacles)
        results.append((name, result, grid))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (name, result, grid) in enumerate(results):
        occ = grid.grid.T
        extent = (-grid.half_size, grid.half_size, -grid.half_size, grid.half_size)
        ax_occ = axes[0, i]
        ax_occ.imshow(occ, cmap="RdYlGn", origin="lower", vmin=-1, vmax=1, extent=extent, interpolation="nearest")
        ax_occ.plot(result.trail[:, 0], result.trail[:, 1], linewidth=0.7, color="tab:blue")
        ax_occ.scatter(result.trail[0, 0], result.trail[0, 1], c="green", s=20)
        ax_occ.scatter(result.trail[-1, 0], result.trail[-1, 1], c="red", s=20)
        ax_occ.set_title(f"{name}: occ+traj")
        ax_occ.set_aspect("equal", adjustable="box")
        ax_occ.set_xlim(-grid.half_size, grid.half_size)
        ax_occ.set_ylim(-grid.half_size, grid.half_size)

        ax_curve = axes[1, i]
        ax_curve.plot(result.explored_fraction_per_step, color="tab:green", label="explored frac")
        ax_curve.plot(np.cumsum(result.collisions_per_step) / np.arange(1, len(result.collisions_per_step) + 1),
                      color="tab:red", label="collision rate")
        ax_curve.set_title(
            f"{name}: final={result.final_explored_fraction:.3f}, collisions={result.total_collisions}"
        )
        ax_curve.set_xlabel("step")
        ax_curve.grid(alpha=0.3)
        if i == 0:
            ax_curve.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "policies_comparison.png", dpi=140)
    plt.close(fig)

    for name, result, _ in results:
        print(
            f"{name}: explored={result.final_explored_fraction:.3f}, "
            f"collisions={result.total_collisions}, reward={result.total_reward:.1f}"
        )
    print(f"saved_artifact={out / 'policies_comparison.png'}")


if __name__ == "__main__":
    main()
