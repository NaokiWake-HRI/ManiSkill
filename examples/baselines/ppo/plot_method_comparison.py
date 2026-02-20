"""Compare outer-loop vs eureka: success_once vs outer iteration for 4 tasks.

Usage:
    python plot_method_comparison.py                        # aggregate all available seeds
    python plot_method_comparison.py --seeds 1788 4796 9351 # use specific seeds
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13})

BASE_DIR = Path(__file__).parent / "runs"

METHODS = {
    "outer-loop": {"label": "Outer-Loop", "color": "#e67e22", "marker": "o"},
    "eureka": {"label": "Eureka (LLM-only)", "color": "#1a5276", "marker": "s"},
}

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "AnymalC-Reach-v1",
]


def _extract_timestamp(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})$", name)
    return m.group(1) if m else "00000000_000000"


def _extract_seed(name: str) -> str | None:
    m = re.search(r"-(\d+)-\S+-\d{8}_\d{6}$", name)
    return m.group(1) if m else None


def latest_run(method: str, task: str, seed: str) -> Path | None:
    task_dir = BASE_DIR / method / task
    if not task_dir.exists():
        return None
    runs = [r for r in task_dir.iterdir() if _extract_seed(r.name) == seed]
    if not runs:
        return None
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def load_success(run_dir: Path) -> list[float] | None:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        history = json.load(f)
    return [e["success_once"] for e in history]


def main():
    parser = argparse.ArgumentParser(description="Outer-loop vs Eureka comparison")
    parser.add_argument("--seeds", nargs="+", type=str, default=["1788", "4796", "9351"])
    args = parser.parse_args()

    seeds = args.seeds
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4.5 * n_tasks, 4), sharey=True)

    for ax, task in zip(axes, TASKS):
        for method, style in METHODS.items():
            all_success = []
            for seed in seeds:
                run_dir = latest_run(method, task, seed)
                if run_dir is None:
                    continue
                success = load_success(run_dir)
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
                    label=f'{style["label"]} (n={len(all_success)})', zorder=2)

        ax.set_xlabel("Outer iteration")
        ax.set_xticks(np.arange(1, 6))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(task, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)

    axes[0].set_ylabel("success_once")

    fig.suptitle("Outer-Loop vs Eureka (LLM-only)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = BASE_DIR / "method_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
