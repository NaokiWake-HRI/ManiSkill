"""Plot outer loop iterations overlaid on the same axes.

Detects iteration boundaries automatically from tensorboard data
(outer_iter/weights/* logged at each iteration start).

Single run:  rolling-mean line + faint raw data per iteration.
Multi seed:  mean line + shaded SEM across seeds per iteration.

Usage:
    # Single run
    python plot_outer_loop_iterations.py runs/run_a

    # Multiple seeds (mean +/- SEM)
    python plot_outer_loop_iterations.py runs/run_a runs/run_b runs/run_c
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalar(run_dir: str, tag: str):
    """Load a scalar tag from a tensorboard events file."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        raise ValueError(f"Tag '{tag}' not found. Available: {available}")
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def detect_iter_boundaries(run_dir: str):
    """Detect iteration start steps from outer_iter/weights/* in tensorboard."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    # outer_iter/weights/* is logged at each iteration START
    weight_tags = [t for t in tags if t.startswith("outer_iter/weights/")]
    if weight_tags:
        events = ea.Scalars(weight_tags[0])
        iter_starts = sorted(set(e.step for e in events))
        print(f"Detected {len(iter_starts)} iterations from {weight_tags[0]}: {iter_starts}")
        return iter_starts

    # Fallback: outer_loop_history.json
    history_path = Path(run_dir) / "outer_loop_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        iter_starts = [entry["learning_curve"][0]["step"] for entry in history]
        print(f"Detected {len(iter_starts)} iterations from history JSON: {iter_starts}")
        return iter_starts

    raise FileNotFoundError(f"Cannot detect iterations: no outer_iter/weights/* tags "
                            f"and no outer_loop_history.json in {run_dir}")


def rolling_mean(values, window=50):
    """Causal rolling mean (uses only past values, no lookahead)."""
    cumsum = np.cumsum(values)
    out = np.empty_like(values)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def extract_iteration(steps, values, start, end):
    """Extract data for one iteration, offset steps to 0."""
    mask = (steps > start) & (steps < end)
    return steps[mask] - start, values[mask]


def resample_to_grid(x, y, grid, window=50):
    """Interpolate (x, y) onto a common grid via rolling-mean-smoothed values."""
    if len(x) == 0:
        return np.full_like(grid, np.nan)
    smoothed = rolling_mean(y, window=window)
    return np.interp(grid, x, smoothed, left=np.nan, right=np.nan)


# ---- Single run mode ----

def plot_single(run_dir, metric, smooth_window, output, max_iters=None):
    iter_starts = detect_iter_boundaries(str(run_dir))
    iter_ends = iter_starts[1:] + [np.inf]
    if max_iters is not None:
        iter_starts = iter_starts[:max_iters]
        iter_ends = iter_ends[:max_iters]

    steps, values = load_tb_scalar(str(run_dir), metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(iter_starts)))

    for i, (start, end) in enumerate(zip(iter_starts, iter_ends)):
        ix, iv = extract_iteration(steps, values, start, end)
        if len(ix) == 0:
            continue

        ax.plot(ix, iv, color=colors[i], alpha=0.15, linewidth=0.5)

        if len(iv) > 1:
            smoothed = rolling_mean(iv, window=smooth_window)
            ax.plot(ix, smoothed, color=colors[i], linewidth=2,
                    label=f"Iter {i+1}")

        print(f"Iter {i+1}: steps {start} -> {end}, {len(ix)} pts")

    ax.set_xlabel("Step (within iteration)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per outer-loop iteration\n{run_dir.name}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, None)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    metric_slug = metric.replace("/", "_")
    out = Path(output) if output else run_dir / f"iterations_overlay_{metric_slug}.png"
    fig.savefig(str(out), dpi=150)
    print(f"Saved to {out.resolve()}")
    plt.close()


# ---- Multi seed mode ----

def plot_multi(run_dirs, metric, num_bins, output, max_iters=None):
    all_boundaries = []
    all_steps_values = []
    for rd in run_dirs:
        all_boundaries.append(detect_iter_boundaries(str(rd)))
        all_steps_values.append(load_tb_scalar(str(rd), metric))

    num_iters = min(len(b) for b in all_boundaries)
    if max_iters is not None:
        num_iters = min(num_iters, max_iters)
    num_seeds = len(run_dirs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_iters))

    for i in range(num_iters):
        seed_curves = []
        max_step = 0
        for s in range(num_seeds):
            starts = all_boundaries[s]
            ends = starts[1:] + [np.inf]
            ix, iv = extract_iteration(
                all_steps_values[s][0], all_steps_values[s][1],
                starts[i], ends[i])
            if len(ix) > 0:
                seed_curves.append((ix, iv))
                max_step = max(max_step, ix[-1])

        if not seed_curves or max_step == 0:
            continue

        grid = np.linspace(0, max_step, num_bins)
        resampled = np.array([resample_to_grid(x, y, grid)
                              for x, y in seed_curves])

        with np.errstate(all="ignore"):
            mean = np.nanmean(resampled, axis=0)
            sem = np.nanstd(resampled, axis=0) / np.sqrt(
                np.sum(~np.isnan(resampled), axis=0))

        valid = ~np.isnan(mean)
        gv, mv, sv = grid[valid], mean[valid], sem[valid]

        ax.plot(gv, mv, color=colors[i], linewidth=2, label=f"Iter {i+1}")
        ax.fill_between(gv, mv - sv, mv + sv, color=colors[i], alpha=0.2)

        print(f"Iter {i+1}: {len(seed_curves)} seeds, {int(max_step)} max steps")

    ax.set_xlabel("Step (within iteration)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per outer-loop iteration  ({num_seeds} seeds)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(output) if output else run_dirs[0] / "iterations_overlay_multi.png"
    fig.savefig(str(out), dpi=150)
    print(f"Saved to {out.resolve()}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Run directory (1=single, 2+=multi-seed)")
    parser.add_argument("--metric", default="train/success_once")
    parser.add_argument("--smooth_window", type=int, default=50,
                        help="Rolling mean window size in data points (single-run mode)")
    parser.add_argument("--num_bins", type=int, default=200,
                        help="Grid resolution (multi-seed mode only)")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Show only the first N iterations (default: all)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]

    if len(run_dirs) == 1:
        plot_single(run_dirs[0], args.metric, args.smooth_window, args.output,
                    max_iters=args.max_iters)
    else:
        plot_multi(run_dirs, args.metric, args.num_bins, args.output,
                   max_iters=args.max_iters)


if __name__ == "__main__":
    main()
