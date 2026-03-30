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

METHODS_FULL = {
    # "eureka_full": {"label": "Eureka Full (LLM-only)", "color": "#1a5276", "marker": "s"},
    "eureka_full_k_16": {"label": "Eureka Full K=16 (LLM-only)", "color": "#1a5276", "marker": "s"},
    # "outer-loop_full": {"label": "VLM+LLM Full", "color": "#e67e22", "marker": "o"},
    # "outer-loop_full_failureselection": {"label": "VLM+LLM Failure Selection", "color": "#27ae60", "marker": "^"},
    "outer-loop_full_failureselection_k_16": {"label": "VLM+LLM FailSel K=16", "color": "#8e44ad", "marker": "D"},
}

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
    # Strip _resumeN_TIMESTAMP suffix if present (e.g. _resume1_20260302_185045)
    stripped = re.sub(r"_resume\d+_\d{8}_\d{6}$", "", name)
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


def latest_run(method: str, task: str, seed: str) -> Path | None:
    task_dir = BASE_DIR / method / task
    if not task_dir.exists():
        return None
    runs = [r for r in task_dir.iterdir() if _extract_seed(r.name) == seed]
    if not runs:
        return None
    # Prefer latest run by timestamp (early-stopped runs have fewer iters but are still valid)
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def load_success(run_dir: Path, is_full: bool, metric: str = "success_once") -> list[float] | None:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        history = json.load(f)
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
    methods = METHODS_FULL if is_full else METHODS_STANDARD
    tasks = TASKS_FULL if is_full else TASKS_STANDARD
    seeds = args.seeds

    # Filter to tasks that have data for at least one method
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
    tasks = filtered_tasks

    n_tasks = len(tasks)
    if n_tasks == 0:
        print("No tasks with data for all methods.")
        return
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    metrics = [("success_once", "Success Once"), ("success_at_end", "Success At End")]

    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=True)
        axes_flat = axes.flatten() if n_rows > 1 else axes

        for ax, task in zip(axes_flat, tasks):
            for method, style in methods.items():
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
                iters = np.arange(1, n_iters + 1)

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

        title = f"Outer-Loop vs Eureka — {metric_label}" + (" [Full]" if is_full else "")
        fig.suptitle(title, fontsize=15, fontweight="bold")
        plt.tight_layout()

        out_filename = f"method_comparison{'_full' if is_full else ''}_{metric_key}.png"
        out_path = BASE_DIR / out_filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path.resolve()}")
        plt.close()


if __name__ == "__main__":
    main()
