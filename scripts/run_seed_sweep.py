from __future__ import annotations

import argparse
import csv
import json
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_obstacles(cfg: dict, map_seed: int):
    map_cfg = cfg["map"]
    room_half_size = cfg["world"]["room_half_size"]
    name = map_cfg["name"]
    if name == "open":
        return make_open_map()
    if name == "semi":
        mc = MapConfig(
            room_half_size=room_half_size,
            obstacle_min_half_size=map_cfg["obstacle_min_half_size"],
            obstacle_max_half_size=map_cfg["obstacle_max_half_size"],
            obstacle_margin_to_wall=map_cfg["obstacle_margin_to_wall"],
            obstacle_margin_to_origin=map_cfg["obstacle_margin_to_origin"],
        )
        return make_semi_cluttered_map(
            num_obstacles=map_cfg["num_obstacles"],
            seed=map_seed,
            cfg=mc,
        )
    raise ValueError(f"Unsupported map name: {name}")


def _build_policy(policy_name: str, params: dict, policy_seed: int):
    if policy_name == "theta":
        return ThetaSweepPolicy(**params)
    if policy_name == "frontier":
        return FrontierGreedyPolicy(**params)
    if policy_name == "levy":
        return LevyWalkPolicy(seed=policy_seed, **params)
    raise ValueError(f"Unknown policy: {policy_name}")


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _save_raw_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "policy",
        "seed",
        "map_seed",
        "policy_seed",
        "final_explored_fraction",
        "total_collisions",
        "collision_rate",
        "total_reward",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)

    out = {"n_total_runs": len(rows), "policies": {}}
    for policy, rr in by_policy.items():
        explored = np.array([x["final_explored_fraction"] for x in rr], dtype=np.float64)
        collisions = np.array([x["total_collisions"] for x in rr], dtype=np.float64)
        collision_rate = np.array([x["collision_rate"] for x in rr], dtype=np.float64)
        reward = np.array([x["total_reward"] for x in rr], dtype=np.float64)
        out["policies"][policy] = {
            "n_runs": int(len(rr)),
            "final_explored_fraction_mean": float(np.mean(explored)),
            "final_explored_fraction_sem": _sem(explored),
            "total_collisions_mean": float(np.mean(collisions)),
            "total_collisions_sem": _sem(collisions),
            "collision_rate_mean": float(np.mean(collision_rate)),
            "collision_rate_sem": _sem(collision_rate),
            "total_reward_mean": float(np.mean(reward)),
            "total_reward_sem": _sem(reward),
        }
    return out


def _save_summary_plot(summary: dict, out_path: Path) -> None:
    policies = list(summary["policies"].keys())
    x = np.arange(len(policies))
    w = 0.35

    explored_mean = np.array([summary["policies"][p]["final_explored_fraction_mean"] for p in policies])
    explored_sem = np.array([summary["policies"][p]["final_explored_fraction_sem"] for p in policies])
    coll_mean = np.array([summary["policies"][p]["collision_rate_mean"] for p in policies])
    coll_sem = np.array([summary["policies"][p]["collision_rate_sem"] for p in policies])
    rew_mean = np.array([summary["policies"][p]["total_reward_mean"] for p in policies])
    rew_sem = np.array([summary["policies"][p]["total_reward_sem"] for p in policies])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(x, explored_mean, yerr=explored_sem, capsize=3, color="tab:green")
    axes[0].set_title("Final explored fraction")
    axes[0].set_xticks(x, policies)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, coll_mean, yerr=coll_sem, capsize=3, color="tab:red")
    axes[1].set_title("Collision rate")
    axes[1].set_xticks(x, policies)
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(x, rew_mean, yerr=rew_sem, capsize=3, color="tab:purple")
    axes[2].set_title("Total reward")
    axes[2].set_xticks(x, policies)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed policy sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/policy_sweep.default.json",
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = _load_json(cfg_path)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_cfg = EpisodeConfig(**cfg["episode"])
    sensor_cfg = SensorConfig(**cfg["sensor"])
    world_cfg = WorldConfig(room_half_size=cfg["world"]["room_half_size"])

    num_seeds = int(cfg["num_seeds"])
    base_seed = int(cfg["base_seed"])
    policy_items = [(k, v) for k, v in cfg["policies"].items() if v.get("enabled", False)]

    rows: list[dict] = []
    for i in range(num_seeds):
        seed = base_seed + i
        map_seed = 1000 + seed
        obstacles = _make_obstacles(cfg, map_seed=map_seed)
        for policy_name, policy_cfg in policy_items:
            policy_seed = 2000 + seed
            policy = _build_policy(policy_name, policy_cfg.get("params", {}), policy_seed=policy_seed)
            world = MujocoNavWorld(config=world_cfg, obstacles=obstacles)
            grid = OccupancyGrid(world_half_size=world.config.room_half_size, resolution=0.1)
            result = run_episode(world, grid, policy, sensor_cfg, episode_cfg)
            rows.append(
                {
                    "policy": policy_name,
                    "seed": seed,
                    "map_seed": map_seed,
                    "policy_seed": policy_seed,
                    "final_explored_fraction": result.final_explored_fraction,
                    "total_collisions": result.total_collisions,
                    "collision_rate": float(np.mean(result.collisions_per_step)),
                    "total_reward": result.total_reward,
                }
            )
            print(
                f"seed={seed} policy={policy_name} "
                f"explored={result.final_explored_fraction:.3f} "
                f"collisions={result.total_collisions} reward={result.total_reward:.1f}"
            )

    raw_csv = out_dir / "seed_sweep_raw.csv"
    summary_json = out_dir / "seed_sweep_summary.json"
    summary_plot = out_dir / "seed_sweep_summary.png"

    _save_raw_csv(rows, raw_csv)
    summary = _summarize(rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _save_summary_plot(summary, summary_plot)

    print(f"saved_raw={raw_csv}")
    print(f"saved_summary={summary_json}")
    print(f"saved_plot={summary_plot}")


if __name__ == "__main__":
    main()
