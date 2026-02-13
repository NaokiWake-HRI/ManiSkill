"""Combined outer-loop summary plot for all tasks.

For each task (latest run):
  Left:  per-iteration training curves (plot_outer_loop_iterations style)
         Legend shows only Iter 1 (first) and Iter N (last).
  Right: outer_iter/success_once vs iteration number (1-indexed: 1, 2, 3, ..., N).

Usage:
    python plot_outer_loop_summary.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 13})


OUTER_LOOP_DIR = Path(__file__).parent / "runs" / "outer-loop"

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


def latest_run(task: str) -> Path:
    task_dir = OUTER_LOOP_DIR / task
    runs = sorted(task_dir.iterdir(), key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


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


def plot_task(ax_left, ax_right, run_dir: Path, task_name: str):
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
        # Only plot first and last iterations (reversed: last then first)
        plot_indices = [0] if n_iters == 1 else [n_iters - 1, 0]
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
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 3.2 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for i, task in enumerate(TASKS):
        print(f"Processing {task}...")
        try:
            run_dir = latest_run(task)
            print(f"  Latest run: {run_dir.name}")
            plot_task(axes[i, 0], axes[i, 1], run_dir, task)
        except Exception as e:
            print(f"  ERROR: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error: {e}", ha="center",
                            va="center", transform=axes[i, 0].transAxes,
                            fontsize=7, color="red")
            axes[i, 0].set_title(task, fontweight="bold")
            axes[i, 1].set_title(task, fontweight="bold")

    fig.suptitle("Outer-Loop Results (latest run per task)",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_path = OUTER_LOOP_DIR / "outer_loop_summary.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
