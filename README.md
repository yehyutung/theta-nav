# theta-nav

## Virtual environment

- Python virtual environment: `.venv`
- Create it with: `python3 -m venv .venv`
- Activate it on macOS or Linux with: `. .venv/bin/activate`
- Upgrade pip inside the environment with: `python -m pip install --upgrade pip`
- Install baseline dependencies with:
  - `python -m pip install mujoco numpy matplotlib`

## MuJoCo world scaffold

- Reusable world module: `src/theta_nav/world.py`
- Quick smoke test script: `scripts/smoke_test_world.py`
- Run the smoke test with:
  - `python scripts/smoke_test_world.py`
- Save quick world artifacts (trajectory + camera frames):
  - `python scripts/check_world_artifacts.py`

## Mapping scaffold

- Occupancy grid module: `src/theta_nav/occupancy.py`
- Map generators:
  - open map: `make_open_map()`
  - semi-cluttered map: `make_semi_cluttered_map(...)`
- Save mapping artifacts for open and semi-cluttered worlds:
  - `python scripts/check_mapping_artifacts.py`

## Episode protocol scaffold

- Rollout module: `src/theta_nav/rollout.py`
- Protocol choices implemented:
  - fixed horizon episodes
  - episode continues after collisions
  - collision metrics always logged
  - optional collision penalty in reward
- Save protocol-check artifacts:
  - `python scripts/check_episode_protocol.py`

## Policies scaffold

- Policy module: `src/theta_nav/policies.py`
- Implemented policies:
  - `ThetaSweepPolicy` (first version)
  - `LevyWalkPolicy`
  - `FrontierGreedyPolicy` (simple frontier baseline)
- Compare all policies with one artifact:
  - `python scripts/check_policies_artifacts.py`

### Notebook import snippet

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from theta_nav import MujocoNavWorld

world = MujocoNavWorld()
world.reset()
for _ in range(100):
    world.step(forward_speed=1.0, turn_speed=0.2)
img = world.render_topdown_view()
```

## Git setup

- This folder is initialized as a Git repository on the `main` branch.
- The local repository is ready for commits once files are added and staged.
- A GitHub remote has not been attached yet because GitHub authentication is not available in this session.
- To link the repository later, add the remote with `git remote add origin <github-repo-url>` or authenticate with `gh auth login` and create the repo with `gh repo create`.

## Commit and push flow

1. Activate the virtual environment: `. .venv/bin/activate`
2. Check changes: `git status`
3. Stage files: `git add .`
4. Commit: `git commit -m "Initial project setup"`
5. Push after a remote is configured: `git push -u origin main`
