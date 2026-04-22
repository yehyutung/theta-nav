from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from theta_nav import (
    OccupancyGrid,
    SensorConfig,
    VollanGreedySweepPolicy,
    VollanGreedySweepPolicyV2,
    WorldConfig,
    make_open_map,
)
from theta_nav.world import MujocoNavWorld


def _wrap_to_pi_vec(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class EnvSpec:
    name: str
    obstacles: list[tuple[float, float, float, float]]
    difficulty_rank: int


def _build_structured_envs() -> list[EnvSpec]:
    open_env = EnvSpec(name="open", obstacles=make_open_map(), difficulty_rank=1)

    corridor = EnvSpec(
        name="corridor",
        obstacles=[
            (-2.0, 0.0, 0.5, 3.8),
            (2.0, 0.0, 0.5, 3.8),
            (0.0, -3.2, 1.2, 0.35),
            (0.0, 3.2, 1.2, 0.35),
        ],
        difficulty_rank=2,
    )

    staggered = EnvSpec(
        name="staggered_gates",
        obstacles=[
            (-2.6, -1.8, 0.45, 1.0),
            (-0.8, 1.8, 0.45, 1.0),
            (0.8, -1.8, 0.45, 1.0),
            (2.6, 1.8, 0.45, 1.0),
            (0.0, 0.0, 0.35, 1.0),
        ],
        difficulty_rank=3,
    )

    dense = EnvSpec(
        name="dense_grid",
        obstacles=[
            (-2.4, -2.4, 0.35, 0.35),
            (-2.4, 0.0, 0.35, 0.35),
            (-2.4, 2.4, 0.35, 0.35),
            (0.0, -2.4, 0.35, 0.35),
            (0.0, 2.4, 0.35, 0.35),
            (2.4, -2.4, 0.35, 0.35),
            (2.4, 0.0, 0.35, 0.35),
            (2.4, 2.4, 0.35, 0.35),
            (-1.2, -1.2, 0.3, 0.3),
            (-1.2, 1.2, 0.3, 0.3),
            (1.2, -1.2, 0.3, 0.3),
            (1.2, 1.2, 0.3, 0.3),
        ],
        difficulty_rank=4,
    )

    return [open_env, corridor, staggered, dense]


def _one_sided_pvalue_alt_gt_chance(alternation_ratio: float, n_transitions: int) -> float:
    if n_transitions <= 0:
        return 1.0
    # Normal approximation of Binomial(n, 0.5), one-sided p-value P(X >= observed)
    std = math.sqrt(0.25 / n_transitions)
    z = (alternation_ratio - 0.5) / max(1e-12, std)
    # 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _run_episode_trace(
    env: EnvSpec,
    steps: int,
    policy_params: dict,
    sensor_cfg: SensorConfig,
    room_half_size: float,
    move_eps: float,
    policy_type: str = "vollan_v1",
    pose_dither: dict | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    world = MujocoNavWorld(config=WorldConfig(room_half_size=room_half_size), obstacles=env.obstacles)
    grid = OccupancyGrid(world_half_size=room_half_size, resolution=0.1)

    if policy_type == "vollan_v2":
        policy = VollanGreedySweepPolicyV2(seed=seed, **policy_params)
    else:
        policy = VollanGreedySweepPolicy(**policy_params)

    # Pose dither: randomize start position and heading based on seed so that different
    # seeds produce genuinely distinct trajectories (especially important for V2 with dither).
    if pose_dither is not None:
        rng = np.random.default_rng(seed + 9973)  # offset to avoid overlap with policy RNG
        pos_scale = float(pose_dither.get("pos_m", 0.0))
        yaw_scale = float(pose_dither.get("yaw_rad", 0.0))
        start_x = rng.uniform(-pos_scale, pos_scale) if pos_scale > 0.0 else 0.0
        start_y = rng.uniform(-pos_scale, pos_scale) if pos_scale > 0.0 else 0.0
        start_yaw = rng.uniform(-yaw_scale, yaw_scale) if yaw_scale > 0.0 else 0.0
        world.reset(start_x, start_y, start_yaw)
    else:
        world.reset()

    grid.reset()
    grid.update_from_world(world, sensor_cfg=sensor_cfg, mark_agent_free_radius=0.2)

    trail = np.zeros((steps, 2), dtype=np.float64)
    yaw_hist = np.zeros(steps, dtype=np.float64)
    move_heading = np.zeros(steps, dtype=np.float64)
    delta = np.zeros(steps, dtype=np.float64)
    valid = np.zeros(steps, dtype=bool)
    explored = np.zeros(steps, dtype=np.float64)
    chosen_offsets = np.zeros(steps, dtype=np.float64)

    x_prev, y_prev, yaw_prev = world.pose
    for t in range(steps):
        policy.observe(t, world, grid)
        # For V2, record chosen offset and mark only decision steps as valid.
        # Between decisions last_chosen_offset is held, but valid=False prevents
        # repeated identical values from inflating the alternation count.
        if policy_type == "vollan_v2":
            chosen_offsets[t] = policy.last_chosen_offset
            valid[t] = policy.last_decision_step
        forward, turn = policy.action(t, world.pose)
        x, y, yaw = world.step(forward, turn)
        trail[t] = (x, y)
        yaw_hist[t] = yaw

        dx = x - x_prev
        dy = y - y_prev
        disp2 = dx * dx + dy * dy
        if disp2 > move_eps * move_eps:
            mh = float(np.arctan2(dy, dx))
            if policy_type != "vollan_v2":
                valid[t] = True
        else:
            mh = yaw_prev
        move_heading[t] = mh
        delta[t] = float(np.arctan2(np.sin(yaw - mh), np.cos(yaw - mh)))
        x_prev, y_prev, yaw_prev = x, y, yaw

        grid.update_from_world(world, sensor_cfg=sensor_cfg, mark_agent_free_radius=0.2)
        explored[t] = grid.explored_fraction

    out: dict[str, np.ndarray] = {
        "trail": trail,
        "yaw": _wrap_to_pi_vec(yaw_hist),
        "move_heading": _wrap_to_pi_vec(move_heading),
        "delta": _wrap_to_pi_vec(delta),
        "valid": valid,
        "explored": explored,
        "chosen_offsets": chosen_offsets,
    }
    if policy_type == "vollan_v2":
        out["coverage_memory"] = policy._memory.copy()
        out["bin_centers"] = policy._bin_centers.copy()
    return out


def _compute_metrics(
    delta: np.ndarray,
    valid: np.ndarray,
    burn_in_steps: int,
    deadzone_deg: float,
    alt_signal: np.ndarray | None = None,
) -> dict[str, float]:
    # alt_signal overrides delta as the alternation signal (used for V2 chosen_offsets).
    signal = alt_signal if alt_signal is not None else delta
    keep = np.where(valid)[0]
    keep = keep[keep >= burn_in_steps]
    if keep.size < 5:
        return {
            "alternation_ratio": 0.0,
            "alternation_pvalue": 1.0,
            "n_transitions": 0.0,
            "left_right_balance": 0.0,
            "mean_abs_delta_deg": 0.0,
            "usable_steps": float(keep.size),
            "emergence_score": 0.0,
        }

    d = signal[keep]
    dz = np.deg2rad(deadzone_deg)
    side = np.sign(d)
    side[np.abs(d) < dz] = 0.0
    side = side[side != 0.0]
    if side.size < 2:
        return {
            "alternation_ratio": 0.0,
            "alternation_pvalue": 1.0,
            "n_transitions": 0.0,
            "left_right_balance": 0.0,
            "mean_abs_delta_deg": float(np.rad2deg(np.mean(np.abs(d)))),
            "usable_steps": float(keep.size),
            "emergence_score": 0.0,
        }

    flips = np.sum(side[1:] * side[:-1] < 0)
    n_transitions = int(side.size - 1)
    alt_ratio = float(flips / max(1, n_transitions))
    pval = _one_sided_pvalue_alt_gt_chance(alt_ratio, n_transitions)

    n_left = int(np.sum(side < 0))
    n_right = int(np.sum(side > 0))
    balance = 1.0 - abs(n_left - n_right) / max(1, n_left + n_right)
    mean_abs_delta_deg = float(np.rad2deg(np.mean(np.abs(d))))

    significant = 1.0 if (alt_ratio > 0.5 and pval < 0.01) else 0.0
    emergence_score = (
        0.55 * alt_ratio
        + 0.25 * balance
        + 0.10 * min(1.0, mean_abs_delta_deg / 45.0)
        + 0.10 * significant
    )
    return {
        "alternation_ratio": alt_ratio,
        "alternation_pvalue": float(pval),
        "n_transitions": float(n_transitions),
        "left_right_balance": float(balance),
        "mean_abs_delta_deg": mean_abs_delta_deg,
        "usable_steps": float(keep.size),
        "emergence_score": float(emergence_score),
    }


def _plot_top_run(
    out_path: Path,
    run_title: str,
    trace: dict[str, np.ndarray],
    metrics: dict[str, float],
    room_half_size: float,
) -> None:
    chosen = trace.get("chosen_offsets")
    use_chosen = chosen is not None and np.any(np.abs(chosen) > 1e-6)

    if use_chosen:
        _plot_v2(out_path, run_title, trace, metrics, room_half_size)
    else:
        _plot_v1(out_path, run_title, trace, metrics, room_half_size)


def _plot_v1(
    out_path: Path,
    run_title: str,
    trace: dict[str, np.ndarray],
    metrics: dict[str, float],
    room_half_size: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    signal = trace["delta"]
    signal_valid = trace["valid"]

    ax = axes[0]
    ax.plot(np.rad2deg(signal), color="tab:blue", linewidth=0.9)
    ax.set_title("Movement-frame heading offset")
    ax.set_xlabel("step")
    ax.set_ylabel("delta = head - move (deg)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    vals = signal[signal_valid]
    if vals.size > 0:
        ax.hist(np.rad2deg(vals), bins=48, color="tab:purple", alpha=0.75)
    ax.set_title("Histogram of delta")
    ax.set_xlabel("delta (deg)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    trail = trace["trail"]
    ax.plot(trail[:, 0], trail[:, 1], color="tab:green", linewidth=1.0)
    ax.scatter(trail[0, 0], trail[0, 1], c="tab:blue", s=18, label="start")
    ax.scatter(trail[-1, 0], trail[-1, 1], c="tab:red", s=18, label="end")
    ax.set_xlim(-room_half_size, room_half_size)
    ax.set_ylim(-room_half_size, room_half_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Trajectory")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{run_title}\nalt={metrics['alternation_ratio']:.3f}, p={metrics['alternation_pvalue']:.2e}, "
        f"balance={metrics['left_right_balance']:.3f}, score={metrics['emergence_score']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_v2(
    out_path: Path,
    run_title: str,
    trace: dict[str, np.ndarray],
    metrics: dict[str, float],
    room_half_size: float,
) -> None:
    """4-panel V2 figure: decision time series, polar histogram, coloured trajectory, coverage memory."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    chosen = trace["chosen_offsets"]
    valid = trace["valid"]  # True only at decision steps
    decision_idx = np.where(valid)[0]
    decision_offsets_deg = np.rad2deg(chosen[decision_idx])

    # Panel 1 (top-left): decision-level time series coloured by side
    ax1 = fig.add_subplot(gs[0, 0])
    if decision_idx.size > 0:
        colors_pts = ["tab:green" if o > 0 else "tab:red" for o in decision_offsets_deg]
        ax1.scatter(decision_idx, decision_offsets_deg, c=colors_pts, s=12, zorder=3)
        ax1.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax1.set_title("Sweep decisions (one per theta cycle)")
    ax1.set_xlabel("physics step")
    ax1.set_ylabel("chosen offset (deg)")
    ax1.grid(alpha=0.3)

    # Panel 2 (top-right): polar rose plot of chosen offsets at decision steps
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    if decision_offsets_deg.size > 0:
        bins_polar = np.linspace(-180, 180, 37)
        counts, edges = np.histogram(decision_offsets_deg, bins=bins_polar)
        centers_rad = np.deg2rad(0.5 * (edges[:-1] + edges[1:]))
        width_rad = np.deg2rad(edges[1] - edges[0])
        bar_colors = ["tab:green" if c > 0 else "tab:red" if c < 0 else "gray" for c in centers_rad]
        ax2.bar(centers_rad, counts, width=width_rad, color=bar_colors, alpha=0.8)
    ax2.set_title("Polar histogram of sweep offsets", pad=14)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    # Panel 3 (bottom-left): trajectory coloured by current sweep side
    ax3 = fig.add_subplot(gs[1, 0])
    trail = trace["trail"]
    # Propagate chosen_offsets forward so every step has the side of the last decision.
    side_color = np.full(len(trail), "gray", dtype=object)
    last_offset = 0.0
    for t in range(len(trail)):
        if valid[t]:
            last_offset = chosen[t]
        side_color[t] = "tab:green" if last_offset > 0 else "tab:red"
    for t in range(len(trail) - 1):
        ax3.plot(trail[t:t + 2, 0], trail[t:t + 2, 1], color=side_color[t], linewidth=1.0, solid_capstyle="round")
    ax3.scatter(trail[0, 0], trail[0, 1], c="tab:blue", s=22, zorder=5, label="start")
    ax3.scatter(trail[-1, 0], trail[-1, 1], c="k", s=22, zorder=5, label="end")
    ax3.set_xlim(-room_half_size, room_half_size)
    ax3.set_ylim(-room_half_size, room_half_size)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_title("Trajectory (green=right-offset, red=left-offset)")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    # Panel 4 (bottom-right): coverage memory at episode end
    ax4 = fig.add_subplot(gs[1, 1])
    mem = trace.get("coverage_memory")
    bin_centers = trace.get("bin_centers")
    if mem is not None and bin_centers is not None:
        bin_deg = np.rad2deg(bin_centers)
        bar_colors_mem = ["tab:green" if d > 0 else "tab:red" if d < 0 else "gray" for d in bin_deg]
        ax4.bar(bin_deg, mem, width=np.rad2deg(bin_centers[1] - bin_centers[0]) * 0.9,
                color=bar_colors_mem, alpha=0.75)
        ax4.axvline(0, color="k", linewidth=0.6, linestyle="--")
    ax4.set_title("Coverage memory at episode end")
    ax4.set_xlabel("relative offset (deg)")
    ax4.set_ylabel("accumulated coverage")
    ax4.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"{run_title}\n"
        f"alt={metrics['alternation_ratio']:.3f}  p={metrics['alternation_pvalue']:.2e}  "
        f"balance={metrics['left_right_balance']:.3f}  mean|offset|={metrics['mean_abs_delta_deg']:.1f}°  "
        f"score={metrics['emergence_score']:.3f}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_figure_filename(rank: int, param_id: str, env_name: str) -> str:
    """Replace decimal points in param_id only (e.g. 0.940 → 0_940); keep ``.png``."""
    safe_param = param_id.replace(".", "_")
    return f"rank{rank:02d}_{safe_param}_{env_name}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan VollanGreedySweepPolicy for left-right alternation emergence.")
    parser.add_argument("--config", type=str, default="configs/vollan_alternation_scan.quick.json")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "scan_raw.json"
    summary_path = out_dir / "scan_summary.json"
    top_dir = out_dir / "top_runs"
    top_dir.mkdir(parents=True, exist_ok=True)

    policy_type = cfg.get("policy_type", "vollan_v1")
    pose_dither = cfg.get("pose_dither", None)

    env_lookup = {e.name: e for e in _build_structured_envs()}
    envs = [env_lookup[name] for name in cfg["environments"]]

    sensor_cfg = SensorConfig(**cfg["sensor"])
    room_half_size = float(cfg["world"]["room_half_size"])
    steps = int(cfg["episode"]["steps"])
    burn_in_frac = float(cfg["metrics"]["burn_in_fraction"])
    burn_in_steps = int(steps * burn_in_frac)
    move_eps = float(cfg["metrics"]["movement_eps"])
    deadzone_deg = float(cfg["metrics"]["side_deadzone_deg"])
    n_seeds = int(cfg["scan"]["num_seeds"])
    base_seed = int(cfg["scan"]["base_seed"])
    top_n = int(cfg["scan"]["top_n"])

    # Param grid keys are read dynamically so the same script handles both V1 and V2 configs.
    param_grid = cfg["param_grid"]
    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))

    raw_runs: list[dict] = []
    by_param: dict[str, list[dict]] = {}

    total = len(combos) * len(envs) * n_seeds
    done = 0
    for combo_vals in combos:
        # Build param dict dynamically from the sorted key list.
        param = {}
        for k, v in zip(keys, combo_vals):
            param[k] = int(v) if isinstance(v, (int,)) or (isinstance(v, float) and v == int(v) and k in ("num_candidates", "num_memory_bins", "num_angle_bins")) else float(v)

        param_id = "__".join(f"{k}_{float(v):.3f}" for k, v in zip(keys, combo_vals))

        for env in envs:
            for i in range(n_seeds):
                seed = base_seed + i
                # Merge fixed_policy defaults with the scanned param values.
                policy_params = {k: (int(v) if isinstance(v, float) and v == int(v) and k in ("num_candidates", "num_memory_bins", "num_angle_bins") else v)
                                 for k, v in cfg["fixed_policy"].items()}
                policy_params.update(param)

                trace = _run_episode_trace(
                    env=env,
                    steps=steps,
                    policy_params=policy_params,
                    sensor_cfg=sensor_cfg,
                    room_half_size=room_half_size,
                    move_eps=move_eps,
                    policy_type=policy_type,
                    pose_dither=pose_dither,
                    seed=seed,
                )
                # For V2, use chosen_offsets as the alternation signal.
                alt_signal = trace["chosen_offsets"] if policy_type == "vollan_v2" else None
                metrics = _compute_metrics(
                    delta=trace["delta"],
                    valid=trace["valid"],
                    burn_in_steps=burn_in_steps,
                    deadzone_deg=deadzone_deg,
                    alt_signal=alt_signal,
                )
                run = {
                    "param_id": param_id,
                    "param": param,
                    "env": env.name,
                    "seed": seed,
                    "difficulty_rank": env.difficulty_rank,
                    "metrics": metrics,
                }
                raw_runs.append(run)
                by_param.setdefault(param_id, []).append(run)
                done += 1
                print(
                    f"[{done}/{total}] param={param_id} env={env.name} seed={seed} "
                    f"alt={metrics['alternation_ratio']:.3f} p={metrics['alternation_pvalue']:.2e} "
                    f"score={metrics['emergence_score']:.3f}"
                )

    summary_rows: list[dict] = []
    for param_id, rows in by_param.items():
        # Require success in both easy and hard environments by averaging over selected env set.
        env_names = sorted({r["env"] for r in rows})
        env_stats = {}
        for env_name in env_names:
            rr = [r for r in rows if r["env"] == env_name]
            alt = np.array([r["metrics"]["alternation_ratio"] for r in rr], dtype=np.float64)
            pv = np.array([r["metrics"]["alternation_pvalue"] for r in rr], dtype=np.float64)
            bal = np.array([r["metrics"]["left_right_balance"] for r in rr], dtype=np.float64)
            scr = np.array([r["metrics"]["emergence_score"] for r in rr], dtype=np.float64)
            sig = np.array(
                [1.0 if (r["metrics"]["alternation_ratio"] > 0.5 and r["metrics"]["alternation_pvalue"] < 0.01) else 0.0 for r in rr],
                dtype=np.float64,
            )
            env_stats[env_name] = {
                "alternation_mean": float(np.mean(alt)),
                "alternation_std": float(np.std(alt)),
                "pvalue_median": float(np.median(pv)),
                "balance_mean": float(np.mean(bal)),
                "score_mean": float(np.mean(scr)),
                "significant_rate": float(np.mean(sig)),
            }

        all_scores = np.array([r["metrics"]["emergence_score"] for r in rows], dtype=np.float64)
        all_sig = np.array(
            [1.0 if (r["metrics"]["alternation_ratio"] > 0.5 and r["metrics"]["alternation_pvalue"] < 0.01) else 0.0 for r in rows],
            dtype=np.float64,
        )
        hard_ok = float(np.mean([env_stats[e]["significant_rate"] for e in env_stats if e in {"staggered_gates", "dense_grid"}])) if any(
            e in env_stats for e in {"staggered_gates", "dense_grid"}
        ) else 0.0
        easy_ok = float(np.mean([env_stats[e]["significant_rate"] for e in env_stats if e in {"open", "corridor"}])) if any(
            e in env_stats for e in {"open", "corridor"}
        ) else 0.0
        both_requirement_score = min(easy_ok, hard_ok) if (easy_ok > 0.0 and hard_ok > 0.0) else float(np.mean(all_sig))
        final_rank_score = 0.75 * float(np.mean(all_scores)) + 0.25 * both_requirement_score

        summary_rows.append(
            {
                "param_id": param_id,
                "param": rows[0]["param"],
                "n_runs": len(rows),
                "final_rank_score": float(final_rank_score),
                "overall_significant_rate": float(np.mean(all_sig)),
                "both_requirement_score": float(both_requirement_score),
                "env_stats": env_stats,
            }
        )

    summary_rows.sort(key=lambda x: x["final_rank_score"], reverse=True)
    top_rows = summary_rows[:top_n]

    # Generate figures for top-N parameter sets from best run in each env.
    for rank, row in enumerate(top_rows, start=1):
        param_id = row["param_id"]
        rr = [r for r in raw_runs if r["param_id"] == param_id]
        for env_name in sorted({r["env"] for r in rr}):
            best = max(
                [r for r in rr if r["env"] == env_name],
                key=lambda x: (x["metrics"]["emergence_score"], x["metrics"]["alternation_ratio"]),
            )
            policy_params = {k: (int(v) if isinstance(v, float) and v == int(v) and k in ("num_candidates", "num_memory_bins", "num_angle_bins") else v)
                             for k, v in cfg["fixed_policy"].items()}
            policy_params.update(best["param"])
            env = env_lookup[env_name]
            best_seed = best["seed"]
            trace = _run_episode_trace(
                env=env,
                steps=steps,
                policy_params=policy_params,
                sensor_cfg=sensor_cfg,
                room_half_size=room_half_size,
                move_eps=move_eps,
                policy_type=policy_type,
                pose_dither=pose_dither,
                seed=best_seed,
            )
            alt_signal = trace["chosen_offsets"] if policy_type == "vollan_v2" else None
            metrics = _compute_metrics(
                delta=trace["delta"],
                valid=trace["valid"],
                burn_in_steps=burn_in_steps,
                deadzone_deg=deadzone_deg,
                alt_signal=alt_signal,
            )
            fig_path = top_dir / _safe_figure_filename(rank, param_id, env_name)
            _plot_top_run(
                out_path=fig_path,
                run_title=f"rank={rank} env={env_name} ({policy_type})",
                trace=trace,
                metrics=metrics,
                room_half_size=room_half_size,
            )

    summary = {
        "config_path": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
        "n_parameter_points": len(combos),
        "n_runs": len(raw_runs),
        "top_n": top_n,
        "best": top_rows,
        "scan_settings": cfg,
    }
    _save_json(raw_path, {"runs": raw_runs})
    _save_json(summary_path, summary)

    print(f"saved_raw={raw_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_top_figures_dir={top_dir}")
    if top_rows:
        print(f"best_param={top_rows[0]['param_id']} score={top_rows[0]['final_rank_score']:.3f}")


if __name__ == "__main__":
    main()
