"""Combined outer-loop summary plot for all tasks (Eureka full / outer-loop_full mode).

For each task (latest run):
  Left:  per-iteration training curves (plot_outer_loop_iterations style)
         Uses TB train/success_once within best candidate's step range.
  Right: best candidate success_once vs iteration number (1-indexed).
         Light-blue scatter = all K candidates; black line = best.

Usage:
    python plot_outer_loop_summary_full.py                           # latest run per task (any seed)
    python plot_outer_loop_summary_full.py --seed 1788               # latest run per task for seed 1788
    python plot_outer_loop_summary_full.py --seeds 1788 4796 9351    # aggregate across 3 seeds
    python plot_outer_loop_summary_full.py --mode eureka_full        # only eureka_full runs
    python plot_outer_loop_summary_full.py --mode outer-loop_full    # only outer-loop_full runs
    python plot_outer_loop_summary_full.py --seed 9351 --iters 1 3 5 # show only iterations 1, 3, 5
    python plot_outer_loop_summary_full.py --seed 9351 --include-incomplete
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 13})


RUNS_DIR = Path(__file__).parent / "runs"

# Supported run directories (checked in order)
FULL_DIRS = {
    "eureka_full": RUNS_DIR / "eureka_full",
    "outer-loop_full": RUNS_DIR / "outer-loop_full",
}

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PegInsertionSide-v1",
    "PushT-v1",
    "AnymalC-Reach-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
]


def _extract_timestamp(name: str) -> str:
    """Extract YYYYMMDD_HHMMSS timestamp from run directory name."""
    m = re.search(r"(\d{8}_\d{6})$", name)
    return m.group(1) if m else "00000000_000000"


def _extract_seed(name: str) -> str | None:
    """Extract seed from run directory name like '...-{seed}-{task}-{timestamp}'."""
    m = re.search(r"-(\d+)-\S+-\d{8}_\d{6}$", name)
    return m.group(1) if m else None


def _is_complete_run(run: Path) -> bool:
    """A completed run must have both history and final best candidate."""
    return (run / "outer_loop_history.json").exists() and (run / "final_best_candidate.json").exists()


def _find_task_runs(
    task: str,
    mode: str | None = None,
    include_incomplete: bool = False,
) -> list[Path]:
    """Find all run directories for a task across full dirs."""
    dirs = []
    search_dirs = {mode: FULL_DIRS[mode]} if mode and mode in FULL_DIRS else FULL_DIRS
    for dir_path in search_dirs.values():
        task_dir = dir_path / task
        if task_dir.is_dir():
            for run in task_dir.iterdir():
                if not run.is_dir() or not (run / "outer_loop_history.json").exists():
                    continue
                if not include_incomplete and not _is_complete_run(run):
                    continue
                dirs.append(run)
    return dirs


def latest_run(
    task: str,
    seed: str | None = None,
    mode: str | None = None,
    include_incomplete: bool = False,
) -> Path:
    runs = _find_task_runs(task, mode, include_incomplete=include_incomplete)
    if seed is not None:
        runs = [r for r in runs if _extract_seed(r.name) == seed]
    if not runs:
        raise FileNotFoundError(
            f"No {'(complete) ' if not include_incomplete else ''}runs found for {task}"
            + (f" (seed={seed})" if seed else "")
        )
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def all_latest_runs(
    task: str,
    seeds: list[str],
    mode: str | None = None,
    include_incomplete: bool = False,
) -> list[Path]:
    """Get latest run for each seed."""
    runs = []
    for seed in seeds:
        try:
            runs.append(
                latest_run(
                    task,
                    seed=seed,
                    mode=mode,
                    include_incomplete=include_incomplete,
                )
            )
        except FileNotFoundError:
            print(f"  WARNING: No run found for seed {seed}, skipping")
    return runs


def load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


def load_tb_scalar(run_dir: str, tag: str):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


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
    """Extract data within [start, end) and shift steps to be relative."""
    mask = (steps >= start) & (steps < end)
    return steps[mask] - start, values[mask]


def _iter_step_range(history_entry: dict) -> tuple[int, int]:
    """Get (start_step, end_step) for the best candidate from its learning_curve."""
    lc = history_entry["best_candidate"]["learning_curve"]
    if not lc:
        return 0, 0
    return lc[0]["step"], lc[-1]["step"]


def _iter_color(i: int, n_iters: int):
    """Return (color, alpha) matching the existing summary style."""
    if i == 0:
        return (0.5, 0.5, 0.5), 0.7
    progress = (i - 1) / max(n_iters - 2, 1) if n_iters > 1 else 0
    return plt.cm.coolwarm(progress), 0.3 + 0.6 * progress


def plot_task_aggregated(ax_left, ax_right, run_dirs: list[Path], task_name: str,
                         iter_filter: list[int] | None = None):
    """Plot both panels for a single task, aggregating across multiple seeds."""
    if not run_dirs:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    n_seeds = len(run_dirs)
    all_histories = [load_history(d) for d in run_dirs]
    all_histories = [h for h in all_histories if h]

    if not all_histories:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    # Load TB data for each seed
    metric = "train/success_once"
    all_tb = []
    for run_dir in run_dirs:
        steps, values = load_tb_scalar(str(run_dir), metric)
        all_tb.append((steps, values))

    # --- Left panel: averaged per-iteration training curves (from TB) ---
    n_iters = len(all_histories[0])
    if iter_filter is not None:
        plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
    else:
        plot_indices = list(range(n_iters))[::-1]

    for i in plot_indices:
        seed_curves = []
        for history, (tb_steps, tb_values) in zip(all_histories, all_tb):
            if i >= len(history) or tb_steps is None:
                continue
            start, end = _iter_step_range(history[i])
            if start == end == 0:
                continue
            ix, iv = extract_iteration(tb_steps, tb_values, start, end + 1)
            if len(ix) > 0 and len(iv) > 1:
                smoothed = rolling_mean(iv, window=50)
                seed_curves.append((ix, smoothed))

        if not seed_curves:
            continue

        ref_x = seed_curves[0][0]
        interp_curves = []
        for sx, sv in seed_curves:
            interp_y = np.interp(ref_x, sx, sv, left=np.nan, right=np.nan)
            interp_curves.append(interp_y)
        mean_curve = np.nanmean(interp_curves, axis=0)

        color, alpha = _iter_color(i, n_iters)
        ax_left.plot(ref_x, mean_curve, color=color, alpha=alpha,
                     linewidth=1.8, label=f"Iter {i+1}")

    ax_left.set_xlabel("Step (within iteration)")
    ax_left.set_ylabel("success_once")
    ax_left.set_xlim(0, None)
    ax_left.set_ylim(-0.05, 1.05)
    leg = ax_left.legend(loc="lower right")
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_title(f"{task_name} (n={n_seeds})", fontweight="bold")

    # --- Right panel: success_once vs iteration (all seeds + mean) ---
    all_iters = []
    all_success = []

    for history in all_histories:
        iters = [e["outer_iter"] + 1 for e in history]
        success = [
            e["best_candidate"]["eval_metrics"].get("success_once", 0.0)
            for e in history
        ]
        all_iters.append(iters)
        all_success.append(success)

        # All candidates scatter (light blue)
        for entry in history:
            it = entry["outer_iter"] + 1
            for cand in entry.get("all_candidates", []):
                cs = cand.get("eval_metrics", {}).get("success_once", 0.0)
                ax_right.plot(it, cs, "o", alpha=0.15, markersize=4,
                              color="tab:cyan", zorder=0)

    if all_iters:
        # Individual seeds
        for iters, success in zip(all_iters, all_success):
            ax_right.plot(iters, success, "o", alpha=0.5, markersize=6,
                          color="gray", zorder=1)

        iters_ref = all_iters[0]
        success_array = np.array(all_success)
        mean_success = np.mean(success_array, axis=0)
        ax_right.plot(iters_ref, mean_success, "o-", color="black",
                      linewidth=2.5, markersize=8,
                      label=f"Mean (n={n_seeds})", zorder=2)
        ax_right.set_xticks(iters_ref)
        ax_right.legend(loc="best")

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success_once")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def plot_task(ax_left, ax_right, run_dir: Path, task_name: str,
              iter_filter: list[int] | None = None):
    """Plot both panels for a single task."""
    history = load_history(run_dir)
    if not history:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    n_iters = len(history)
    metric = "train/success_once"

    # --- Left panel: per-iteration training curves (best candidate, from TB) ---
    tb_steps, tb_values = load_tb_scalar(str(run_dir), metric)
    if tb_steps is None:
        ax_left.text(0.5, 0.5, f"No {metric}", ha="center", va="center",
                     transform=ax_left.transAxes)
    else:
        if iter_filter is not None:
            plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
        else:
            plot_indices = list(range(n_iters))[::-1]

        for i in plot_indices:
            start, end = _iter_step_range(history[i])
            if start == end == 0:
                continue
            ix, iv = extract_iteration(tb_steps, tb_values, start, end + 1)
            if len(ix) == 0:
                continue

            color, alpha = _iter_color(i, n_iters)
            if len(iv) > 1:
                smoothed = rolling_mean(iv, window=50)
                ax_left.plot(ix, smoothed, color=color, alpha=alpha,
                             linewidth=1.8, label=f"Iter {i+1}")

        ax_left.set_xlabel("Step (within iteration)")
        ax_left.set_ylabel("success_once")
        ax_left.set_xlim(0, None)
        ax_left.set_ylim(-0.05, 1.05)
        leg = ax_left.legend(loc="lower right")
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        ax_left.grid(True, alpha=0.3)

    ax_left.set_title(task_name, fontweight="bold")

    # --- Right panel: success_once vs iteration ---
    iters = [e["outer_iter"] + 1 for e in history]
    success = [
        e["best_candidate"]["eval_metrics"].get("success_once", 0.0)
        for e in history
    ]

    # All candidates scatter (show diversity)
    for entry in history:
        it = entry["outer_iter"] + 1
        for cand in entry.get("all_candidates", []):
            cs = cand.get("eval_metrics", {}).get("success_once", 0.0)
            ax_right.plot(it, cs, "o", alpha=0.3, markersize=5,
                          color="tab:cyan", zorder=0)

    # Best candidate line
    ax_right.plot(iters, success, "o-", color="black",
                  linewidth=1.8, markersize=5, zorder=2)
    ax_right.set_xticks(iters)

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success_once")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Outer-loop summary plot (full mode)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Filter runs by seed (e.g. 1788)")
    parser.add_argument("--seeds", nargs="+", type=str, default=None,
                        help="Aggregate across multiple seeds (e.g. 1788 4796 9351)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["eureka_full", "outer-loop_full"],
                        help="Only show runs from this directory (default: both)")
    parser.add_argument("--iters", nargs="+", type=int, default=None,
                        help="Show only specific iterations (1-indexed, e.g. --iters 1 3 5)")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs missing final_best_candidate.json (possibly in-progress/incomplete)",
    )
    args = parser.parse_args()

    iter_filter = args.iters
    if iter_filter:
        print(f"Filtering to iterations: {iter_filter}")

    # Determine mode: aggregate or single seed
    if args.seeds:
        plot_mode = "aggregate"
        seeds = args.seeds
        print(f"Aggregating across {len(seeds)} seeds: {seeds}")
    elif args.seed:
        plot_mode = "single"
        seeds = None
        print(f"Plotting single seed: {args.seed}")
    else:
        plot_mode = "latest"
        seeds = None
        print("Plotting latest run per task (any seed)")

    # Filter to tasks that have data
    available_tasks = []
    for task in TASKS:
        runs = _find_task_runs(task, args.mode, include_incomplete=args.include_incomplete)
        if args.seeds:
            runs = [r for r in runs if _extract_seed(r.name) in args.seeds]
        elif args.seed:
            runs = [r for r in runs if _extract_seed(r.name) == args.seed]
        if runs:
            available_tasks.append(task)

    if not available_tasks:
        print("No runs found. Check run directories and try --include-incomplete if runs are still in progress.")
        return

    n_tasks = len(available_tasks)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 3.2 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for i, task in enumerate(available_tasks):
        print(f"Processing {task}...")
        try:
            if plot_mode == "aggregate":
                run_dirs = all_latest_runs(
                    task,
                    seeds,
                    mode=args.mode,
                    include_incomplete=args.include_incomplete,
                )
                if run_dirs:
                    print(f"  Found {len(run_dirs)} runs")
                    plot_task_aggregated(axes[i, 0], axes[i, 1], run_dirs, task,
                                        iter_filter=iter_filter)
                else:
                    raise FileNotFoundError("No runs found for specified seeds")
            else:
                run_dir = latest_run(
                    task,
                    seed=args.seed if plot_mode == "single" else None,
                    mode=args.mode,
                    include_incomplete=args.include_incomplete,
                )
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
    mode_label = args.mode or "full"
    if plot_mode == "aggregate":
        title_suffix = f" (n={len(seeds)} seeds)"
        file_suffix = f"_agg_n{len(seeds)}"
    elif plot_mode == "single":
        title_suffix = f" (seed {args.seed})"
        file_suffix = f"_seed{args.seed}"
    else:
        title_suffix = ""
        file_suffix = ""

    fig.suptitle(f"Outer-Loop Full Results ({mode_label}{title_suffix})",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_dir = FULL_DIRS.get(args.mode, RUNS_DIR / "outer-loop_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"outer_loop_summary{file_suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
