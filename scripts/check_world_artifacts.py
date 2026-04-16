from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import MujocoNavWorld


def run_check(output_dir: Path, steps: int = 500) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    obstacles = [
        (1.0, 1.5, 0.6, 0.4),
        (-1.5, 0.5, 0.4, 0.7),
        (0.0, -1.8, 0.8, 0.2),
    ]
    world = MujocoNavWorld(obstacles=obstacles)
    world.reset()

    trail = np.empty((steps, 2), dtype=np.float32)
    sample_steps = [0, steps // 2, steps - 1]

    for t in range(steps):
        # Simple smooth turn profile to verify controls and collision behavior.
        fwd = 1.0
        turn = 0.5 * np.sin(t * 0.02)
        x, y, _ = world.step(forward_speed=fwd, turn_speed=turn)
        trail[t] = (x, y)

        if t in sample_steps:
            top = world.render_topdown_view()
            agent = world.render_agent_view()
            plt.imsave(output_dir / f"topdown_{t:03d}.png", top)
            plt.imsave(output_dir / f"agent_{t:03d}.png", agent)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(trail[:, 0], trail[:, 1], color="tab:blue", linewidth=1.5, label="trail")
    ax.scatter(trail[0, 0], trail[0, 1], c="green", s=35, label="start", zorder=3)
    ax.scatter(trail[-1, 0], trail[-1, 1], c="red", s=35, label="end", zorder=3)
    ax.set_title("World Smoke Check: Agent Trajectory")
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_check.png", dpi=120)
    plt.close(fig)

    print(f"Saved artifacts to: {output_dir}")
    print("- trajectory_check.png")
    for t in sample_steps:
        print(f"- topdown_{t:03d}.png")
        print(f"- agent_{t:03d}.png")


if __name__ == "__main__":
    run_check(output_dir=ROOT / "artifacts" / "world_check")
