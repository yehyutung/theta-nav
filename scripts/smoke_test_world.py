from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import MujocoNavWorld


def main() -> None:
    obstacles = [
        (1.0, 1.5, 0.6, 0.4),
        (-1.5, 0.5, 0.4, 0.7),
        (0.0, -1.8, 0.8, 0.2),
    ]
    world = MujocoNavWorld(obstacles=obstacles)
    world.reset()

    steps = 400
    trail = np.empty((steps, 2), dtype=np.float32)

    for t in range(steps):
        if t < 100:
            fwd, turn = 1.2, 0.0
        elif t < 200:
            fwd, turn = 1.0, 0.8
        elif t < 300:
            fwd, turn = 1.1, -0.6
        else:
            fwd, turn = 0.8, 0.4
        x, y, _ = world.step(fwd, turn)
        trail[t] = (x, y)

    top = world.render_topdown_view()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(top)
    axes[0].set_title("MuJoCo top-down")
    axes[0].axis("off")

    axes[1].plot(trail[:, 0], trail[:, 1], linewidth=1.5)
    axes[1].scatter(trail[0, 0], trail[0, 1], c="green", s=30, label="start")
    axes[1].scatter(trail[-1, 0], trail[-1, 1], c="red", s=30, label="end")
    axes[1].set_title("Trajectory")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
