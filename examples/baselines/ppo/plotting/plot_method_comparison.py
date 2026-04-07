"""Compare outer-loop vs eureka: success_once vs outer iteration for tasks.

Usage:
    python plot_method_comparison.py                              # standard mode, all seeds
    python plot_method_comparison.py --mode full                  # full (multi-candidate) mode
    python plot_method_comparison.py --seeds 1788 4796 9351       # use specific seeds
    python plot_method_comparison.py --mode full --seeds 1788 4796 # full mode with specific seeds
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13})

BASE_DIR = Path(__file__).resolve().parent.parent / "runs"

METHODS_STANDARD = {
    "eureka": {"label": "Eureka (LLM-only)", "color": "#1a5276", "marker": "s"},
    "outer-loop": {"label": "Outer-Loop", "color": "#e67e22", "marker": "o"},
}

METHODS_FULL_FAILURE = {
    "eureka_full_k_16": {"label": "Eureka K=16", "color": "#2980b9", "marker": "s"},
    "outer-loop_full_failureselection_k_16": {"label": "VLM+LLM FailSel K=16", "color": "#8e44ad", "marker": "D"},
}

METHODS_FULL_FNM = {
    "eureka_full_failure_and_near_miss_k_16": {"label": "Eureka K=16 (F+NM)", "color": "#2980b9", "marker": "s"},
    "outer-loop_full_failureselection_failure_and_near_miss_k_16": {"label": "VLM+LLM F+NM K=16", "color": "#27ae60", "marker": "^"},
}

METHODS_FULL_GROUPS = [
    ("failure", METHODS_FULL_FAILURE),
    ("failure_and_near_miss", METHODS_FULL_FNM),
]

TASKS_STANDARD = [
    "PushCube-v1",
    "PickCube-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "AnymalC-Reach-v1",
]

TASKS_FULL = [
    "PushCube-v1",
    "PickCube-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PegInsertionSide-v1",
    "PushT-v1",
    "RotateValveLevel0-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "UnitreeG1TransportBox-v1",
    "AnymalC-Reach-v1",
]


def _extract_timestamp(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})$", name)
    return m.group(1) if m else "00000000_000000"


def _extract_seed(name: str) -> str | None:
    # Strip all _resumeN_TIMESTAMP suffixes (may be chained, e.g. _resume1_..._resume1_...)
    stripped = name
    while re.search(r"_resume\d+_\d{8}_\d{6}$", stripped):
        stripped = re.sub(r"_resume\d+_\d{8}_\d{6}$", "", stripped)
    m = re.search(r"-(\d+)-\S+-\d{8}_\d{6}$", stripped)
    return m.group(1) if m else None


def _history_length(run_dir: Path) -> int:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return 0
    try:
        with open(history_path) as f:
            return len(json.load(f))
    except (json.JSONDecodeError, ValueError):
        return 0


def _best_success(run_dir: Path) -> float:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return 0.0
    try:
        with open(history_path) as f:
            history = json.load(f)
        return max((e["best_candidate"]["eval_metrics"].get("success_once", 0.0) for e in history), default=0.0)
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0.0


def latest_run(method: str, task: str, seed: str) -> Path | None:
    task_dir = BASE_DIR / method / task
    if not task_dir.exists():
        return None
    runs = [r for r in task_dir.iterdir() if _extract_seed(r.name) == seed]
    if not runs:
        return None
    # Prefer run with best results; break ties by timestamp
    runs_with_history = [r for r in runs if _history_length(r) > 0]
    if runs_with_history:
        runs_with_history.sort(key=lambda p: (_best_success(p), _extract_timestamp(p.name)))
        return runs_with_history[-1]
    # No run has history yet
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def load_success(run_dir: Path, is_full: bool, metric: str = "success_once", max_iters: int = 5) -> list[float] | None:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        history = json.load(f)
    history = history[:max_iters]
    if is_full:
        return [e["best_candidate"]["eval_metrics"].get(metric, 0.0) for e in history]
    else:
        return [e.get(metric, 0.0) for e in history]


def main():
    parser = argparse.ArgumentParser(description="Outer-loop vs Eureka comparison")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "full"],
                        help="standard: single-candidate, full: multi-candidate")
    parser.add_argument("--seeds", nargs="+", type=str, default=["1788", "4796", "9351"])
    args = parser.parse_args()

    is_full = args.mode == "full"
    if is_full:
        method_groups = METHODS_FULL_GROUPS
    else:
        method_groups = [("standard", METHODS_STANDARD)]
    tasks = TASKS_FULL if is_full else TASKS_STANDARD
    seeds = args.seeds

    for group_name, methods in method_groups:
        # Filter to tasks that have data for at least one method in this group
        filtered_tasks = []
        for task in tasks:
            has_any = any(any(latest_run(m, task, s) is not None for s in seeds) for m in methods)
            if has_any:
                filtered_tasks.append(task)
                missing = [m for m in methods if not any(latest_run(m, task, s) is not None for s in seeds)]
                if missing:
                    print(f"Note: {task} missing data for {missing} (plotting available methods only)")
            else:
                print(f"Skipping {task}: no data for any method")
        group_tasks = filtered_tasks

        n_tasks = len(group_tasks)
        if n_tasks == 0:
            print(f"No tasks with data for group {group_name}.")
            continue
        n_cols = min(4, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        metrics = [("success_once", "Success Once"), ("success_at_end", "Success At End")]

        for metric_key, metric_label in metrics:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=True)
            axes_flat = np.atleast_1d(axes.flatten() if hasattr(axes, 'flatten') else axes)

            for ax, task in zip(axes_flat, group_tasks):
                n_methods = len(methods)
                for mi, (method, style) in enumerate(methods.items()):
                    all_success = []
                    for seed in seeds:
                        run_dir = latest_run(method, task, seed)
                        if run_dir is None:
                            continue
                        success = load_success(run_dir, is_full, metric=metric_key)
                        if success is not None:
                            all_success.append(success)

                    if not all_success:
                        continue

                    n_iters = len(all_success[0])
                    # Jitter x-positions to avoid overlap
                    jitter = (mi - (n_methods - 1) / 2) * 0.08
                    iters = np.arange(1, n_iters + 1) + jitter

                    # Individual seeds as faint scatter
                    for s in all_success:
                        ax.plot(iters, s, style["marker"], alpha=0.3, markersize=5,
                                color=style["color"], zorder=1)

                    # Mean line
                    arr = np.array(all_success)
                    mean = np.mean(arr, axis=0)
                    ax.plot(iters, mean, f'{style["marker"]}-', color=style["color"],
                            linewidth=2.5, markersize=8,
                            label=style["label"] if (is_full and len(all_success) == 1) else f'{style["label"]} (n={len(all_success)})', zorder=2)

                ax.set_xlabel("Outer iteration")
                ax.set_xticks(np.arange(1, 6))
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
                ax.set_title(task, fontweight="bold")
                ax.legend(loc="lower right", fontsize=10)

            # Hide unused axes
            for ax in axes_flat[n_tasks:]:
                ax.set_visible(False)

            # y-label on leftmost column only
            for row in range(n_rows):
                axes_flat[row * n_cols].set_ylabel(metric_key)

            title = f"Outer-Loop vs Eureka — {metric_label} [{group_name}]"
            fig.suptitle(title, fontsize=15, fontweight="bold")
            plt.tight_layout()

            suffix = f"_full_{group_name}" if is_full else ""
            out_filename = f"method_comparison{suffix}_{metric_key}.png"
            out_path = BASE_DIR / out_filename
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            print(f"Saved to {out_path.resolve()}")
            plt.close()


if __name__ == "__main__":
    main()
