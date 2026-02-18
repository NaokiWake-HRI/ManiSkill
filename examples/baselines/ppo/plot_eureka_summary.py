"""Combined Eureka (LLM-only) summary plot for all tasks.

For each task (latest run):
  Left:  per-iteration training curves (plot_outer_loop_iterations style)
         Legend shows only Iter 1 (first) and Iter N (last).
  Right: outer_iter/success_once vs iteration number (1-indexed: 1, 2, 3, ..., N).

Usage:
    python plot_eureka_summary.py                        # latest run per task (any seed)
    python plot_eureka_summary.py --seed 1788             # latest run per task for seed 1788
    python plot_eureka_summary.py --seeds 1788 4796 9351  # aggregate across 3 seeds (mean + individual points)
    python plot_eureka_summary.py --seed 9351 --iters 1 3 5  # show only iterations 1, 3, 5
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 13})


EUREKA_DIR = Path(__file__).parent / "runs" / "eureka"

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "AnymalC-Reach-v1",
]


def _extract_timestamp(name: str) -> str:
    """Extract YYYYMMDD_HHMMSS timestamp from run directory name."""
    m = re.search(r"(\d{8}_\d{6})$", name)
    return m.group(1) if m else "00000000_000000"


def _extract_seed(name: str) -> str | None:
    """Extract seed from run directory name like '...-{seed}-{task}-{timestamp}'."""
    m = re.search(r"-(\d+)-\S+-\d{8}_\d{6}$", name)
    return m.group(1) if m else None


def latest_run(task: str, seed: str | None = None) -> Path:
    task_dir = EUREKA_DIR / task
    runs = list(task_dir.iterdir())
    if seed is not None:
        runs = [r for r in runs if _extract_seed(r.name) == seed]
        if not runs:
            raise FileNotFoundError(f"No runs found for seed {seed}")
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def all_latest_runs(task: str, seeds: list[str]) -> list[Path]:
    """Get latest run for each seed."""
    runs = []
    for seed in seeds:
        try:
            runs.append(latest_run(task, seed=seed))
        except FileNotFoundError:
            print(f"  WARNING: No run found for seed {seed}, skipping")
    return runs


def load_tb_scalar(run_dir: str, tag: str):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def detect_iter_boundaries(run_dir: str):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    weight_tags = [t for t in tags if t.startswith("outer_iter/weights/")]
    if weight_tags:
        events = ea.Scalars(weight_tags[0])
        return sorted(set(e.step for e in events))
    history_path = Path(run_dir) / "outer_loop_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        return [entry["learning_curve"][0]["step"] for entry in history]
    return []


def rolling_mean(values, window=50):
    window = min(window, len(values))
    if window <= 1:
        return values.copy()
    cumsum = np.cumsum(values)
    out = np.empty_like(values)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def extract_iteration(steps, values, start, end):
    mask = (steps > start) & (steps < end)
    return steps[mask] - start, values[mask]


def plot_task_aggregated(ax_left, ax_right, run_dirs: list[Path], task_name: str,
                         iter_filter: list[int] | None = None):
    """Plot both panels for a single task, aggregating across multiple seeds."""
    if not run_dirs:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    metric = "train/success_once"
    n_seeds = len(run_dirs)

    # --- Left panel: averaged per-iteration training curves ---
    # Collect data from all seeds
    all_iter_starts = []
    all_steps_values = []
    for run_dir in run_dirs:
        run_str = str(run_dir)
        iter_starts = detect_iter_boundaries(run_str)
        steps, values = load_tb_scalar(run_str, metric)
        if iter_starts and steps is not None:
            all_iter_starts.append(iter_starts)
            all_steps_values.append((steps, values))

    if not all_iter_starts:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
    else:
        # Use the first seed's iteration boundaries as reference
        iter_starts = all_iter_starts[0]
        n_iters = len(iter_starts)
        iter_ends = iter_starts[1:] + [np.inf]

        if iter_filter is not None:
            plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
        else:
            plot_indices = list(range(n_iters))[::-1]  # Show all iterations (reversed order)

        for i in plot_indices:
            start, end = iter_starts[i], iter_ends[i]

            # Collect curves from all seeds for this iteration
            seed_curves = []
            for steps, values in all_steps_values:
                ix, iv = extract_iteration(steps, values, start, end)
                if len(ix) > 0 and len(iv) > 1:
                    smoothed = rolling_mean(iv, window=50)
                    seed_curves.append((ix, smoothed))

            if not seed_curves:
                continue

            # Compute mean curve (interpolate to common x-axis)
            # Use the first seed's x-axis as reference
            ref_x = seed_curves[0][0]
            interp_curves = []
            for ix, iv in seed_curves:
                # Interpolate to reference x-axis
                interp_y = np.interp(ref_x, ix, iv, left=np.nan, right=np.nan)
                interp_curves.append(interp_y)

            mean_curve = np.nanmean(interp_curves, axis=0)

            # Color: gray for Iter 1, red→blue gradient for Iter 2+
            if i == 0:
                color = (0.5, 0.5, 0.5)  # Gray for baseline
                alpha_smooth = 0.7
            else:
                # Gradient from red (Iter 2) to blue (last iter)
                progress = (i - 1) / max(n_iters - 2, 1) if n_iters > 1 else 0
                color = plt.cm.coolwarm(progress)
                alpha_smooth = 0.3 + 0.6 * progress  # 0.3 → 0.9

            label = f"Iter {i+1}"
            ax_left.plot(ref_x, mean_curve, color=color, alpha=alpha_smooth,
                         linewidth=1.8, label=label)

        ax_left.set_xlabel("Step (within iteration)")
        ax_left.set_ylabel("success_once")
        ax_left.set_xlim(0, None)
        ax_left.set_ylim(-0.05, 1.05)
        leg = ax_left.legend(loc="lower right")
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        ax_left.grid(True, alpha=0.3)

    ax_left.set_title(f"{task_name} (n={n_seeds})", fontweight="bold")

    # --- Right panel: outer_iter/success_once vs iteration (all seeds + mean) ---
    all_iters = []
    all_success = []

    for run_dir in run_dirs:
        history_path = run_dir / "outer_loop_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            iters = [e["outer_iter"] + 1 for e in history]  # 1-indexed
            success = [e["success_once"] for e in history]
            all_iters.append(iters)
            all_success.append(success)
        else:
            run_str = str(run_dir)
            outer_steps, outer_values = load_tb_scalar(run_str, "outer_iter/success_once")
            if outer_steps is not None and len(outer_steps) > 0:
                iters = np.arange(1, len(outer_values) + 1)  # 1-indexed
                all_iters.append(iters.tolist())
                all_success.append(outer_values.tolist())

    if all_iters:
        # Plot individual seeds as scatter points
        for iters, success in zip(all_iters, all_success):
            ax_right.plot(iters, success, 'o', alpha=0.5, markersize=6,
                         color='gray', zorder=1)

        # Compute and plot mean
        # Assume all seeds have same iteration count
        iters_ref = all_iters[0]
        success_array = np.array(all_success)  # shape: (n_seeds, n_iters)
        mean_success = np.mean(success_array, axis=0)

        ax_right.plot(iters_ref, mean_success, 'o-', color='black',
                     linewidth=2.5, markersize=8, label=f'Mean (n={n_seeds})',
                     zorder=2)
        ax_right.set_xticks(iters_ref)
        ax_right.legend(loc='best')

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success_once")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def plot_task(ax_left, ax_right, run_dir: Path, task_name: str,
              iter_filter: list[int] | None = None):
    """Plot both panels for a single task."""
    run_str = str(run_dir)
    metric = "train/success_once"

    # --- Left panel: per-iteration training curves ---
    iter_starts = detect_iter_boundaries(run_str)
    if not iter_starts:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    iter_ends = iter_starts[1:] + [np.inf]
    steps, values = load_tb_scalar(run_str, metric)
    if steps is None:
        ax_left.text(0.5, 0.5, f"No {metric}", ha="center", va="center",
                     transform=ax_left.transAxes)
    else:
        n_iters = len(iter_starts)
        if iter_filter is not None:
            plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
        else:
            plot_indices = list(range(n_iters))[::-1]  # Show all iterations (reversed order)

        for i in plot_indices:
            start, end = iter_starts[i], iter_ends[i]
            ix, iv = extract_iteration(steps, values, start, end)
            if len(ix) == 0:
                continue

            # Color: gray for Iter 1, red→blue gradient for Iter 2+
            if i == 0:
                color = (0.5, 0.5, 0.5)  # Gray for baseline
                alpha_raw = 0.06
                alpha_smooth = 0.7
            else:
                # Gradient from red (Iter 2) to blue (last iter)
                progress = (i - 1) / max(n_iters - 2, 1) if n_iters > 1 else 0
                color = plt.cm.coolwarm(progress)
                # Alpha increases with iteration (more dramatic gradient)
                alpha_raw = 0.08 + 0.1 * progress
                alpha_smooth = 0.3 + 0.6 * progress  # 0.3 → 0.9

            label = f"Iter {i+1}"

            # Raw data line (disabled for cleaner visualization)
            # ax_left.plot(ix, iv, color=color, alpha=alpha_raw, linewidth=0.4)

            if len(iv) > 1:
                smoothed = rolling_mean(iv, window=50)
                ax_left.plot(ix, smoothed, color=color, alpha=alpha_smooth,
                             linewidth=1.8, label=label)

        ax_left.set_xlabel("Step (within iteration)")
        ax_left.set_ylabel("success_once")
        ax_left.set_xlim(0, None)
        ax_left.set_ylim(-0.05, 1.05)
        leg = ax_left.legend(loc="lower right")
        # Make legend lines thicker for better visibility
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        ax_left.grid(True, alpha=0.3)

    ax_left.set_title(task_name, fontweight="bold")

    # --- Right panel: outer_iter/success_once vs iteration step ---
    # Prefer outer_loop_history.json (always complete) over TB (may be partial)
    history_path = run_dir / "outer_loop_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        iters = [e["outer_iter"] + 1 for e in history]  # Convert to 1-indexed
        success = [e["success_once"] for e in history]
        ax_right.plot(iters, success, "o-", color="black",
                      linewidth=1.8, markersize=5)
        ax_right.set_xticks(iters)
    else:
        outer_steps, outer_values = load_tb_scalar(run_str, "outer_iter/success_once")
        if outer_steps is not None and len(outer_steps) > 0:
            iter_indices = np.arange(1, len(outer_values) + 1)  # 1-indexed
            ax_right.plot(iter_indices, outer_values, "o-", color="black",
                          linewidth=1.8, markersize=5)
            ax_right.set_xticks(iter_indices)

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success_once")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Eureka (LLM-only) summary plot")
    parser.add_argument("--seed", type=str, default=None,
                        help="Filter runs by seed (e.g. 1788)")
    parser.add_argument("--seeds", nargs="+", type=str, default=None,
                        help="Aggregate across multiple seeds (e.g. 1788 4796 9351)")
    parser.add_argument("--iters", nargs="+", type=int, default=None,
                        help="Show only specific iterations (1-indexed, e.g. --iters 1 3 5)")
    args = parser.parse_args()

    iter_filter = args.iters
    if iter_filter:
        print(f"Filtering to iterations: {iter_filter}")

    # Determine mode: aggregate or single seed
    if args.seeds:
        mode = "aggregate"
        seeds = args.seeds
        print(f"Aggregating across {len(seeds)} seeds: {seeds}")
    elif args.seed:
        mode = "single"
        seeds = None
        print(f"Plotting single seed: {args.seed}")
    else:
        mode = "latest"
        seeds = None
        print("Plotting latest run per task (any seed)")

    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 3.2 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for i, task in enumerate(TASKS):
        print(f"Processing {task}...")
        try:
            if mode == "aggregate":
                run_dirs = all_latest_runs(task, seeds)
                if run_dirs:
                    print(f"  Found {len(run_dirs)} runs")
                    plot_task_aggregated(axes[i, 0], axes[i, 1], run_dirs, task,
                                        iter_filter=iter_filter)
                else:
                    raise FileNotFoundError("No runs found for specified seeds")
            else:
                run_dir = latest_run(task, seed=args.seed if mode == "single" else None)
                print(f"  Latest run: {run_dir.name}")
                plot_task(axes[i, 0], axes[i, 1], run_dir, task,
                         iter_filter=iter_filter)
        except Exception as e:
            print(f"  ERROR: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error: {e}", ha="center",
                            va="center", transform=axes[i, 0].transAxes,
                            fontsize=7, color="red")
            axes[i, 0].set_title(task, fontweight="bold")
            axes[i, 1].set_title(task, fontweight="bold")

    # Title and filename
    if mode == "aggregate":
        title_suffix = f" (n={len(seeds)} seeds)"
        file_suffix = f"_agg_n{len(seeds)}"
    elif mode == "single":
        title_suffix = f" (seed {args.seed})"
        file_suffix = f"_seed{args.seed}"
    else:
        title_suffix = ""
        file_suffix = ""

    fig.suptitle(f"Eureka Results (LLM-only, latest run per task{title_suffix})",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_path = EUREKA_DIR / f"eureka_summary{file_suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
