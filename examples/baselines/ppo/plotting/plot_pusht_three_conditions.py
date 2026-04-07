"""Plot PushT-v1 three-condition comparison: Eureka vs VLM+LLM failure vs VLM+LLM F+NM.

All three share the same iter 0 (failure mode, success_at_end=0.3125).

Usage:
    python plot_pusht_three_conditions.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# All share iter 0 from: outer-loop_full_failureselection_k_16/PushT-v1/...20260401_093535
RUNS = {
    "Eureka K=16": {
        "path": RUNS_DIR / "eureka_full_k_16/PushT-v1",
        "color": "#2980b9", "marker": "s",
    },
    "VLM+LLM failure K=16": {
        "path": RUNS_DIR / "outer-loop_full_failureselection_k_16/PushT-v1",
        "color": "#8e44ad", "marker": "D",
    },
    "VLM+LLM F+NM K=16": {
        "path": RUNS_DIR / "outer-loop_full_failureselection_failure_and_near_miss_k_16/PushT-v1",
        "resume_from": "20260401_093535_resume1_20260406",  # match the F2 run
        "color": "#27ae60", "marker": "^",
    },
}

MAX_ITERS = 5


def find_latest_run(task_dir: Path, resume_from: str | None = None) -> Path | None:
    if not task_dir.exists():
        return None
    candidates = list(task_dir.iterdir())
    if resume_from:
        candidates = [d for d in candidates if resume_from in d.name]
    candidates = [d for d in candidates if (d / "outer_loop_history.json").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name)
    return candidates[-1]


def load_metric(history_path: Path, metric: str, max_iters: int) -> list[float]:
    history = json.loads(history_path.read_text())[:max_iters]
    return [e["best_candidate"]["eval_metrics"].get(metric, 0.0) for e in history]


def main():
    plt.rcParams.update({"font.size": 13})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_key, metric_label, ax in [
        ("success_at_end", "success_at_end", axes[0]),
        ("success_once", "success_once", axes[1]),
    ]:
        for i, (label, cfg) in enumerate(RUNS.items()):
            run_dir = find_latest_run(cfg["path"], cfg.get("resume_from"))
            if run_dir is None:
                print(f"  {label}: no data")
                continue
            vals = load_metric(run_dir / "outer_loop_history.json", metric_key, MAX_ITERS)
            jitter = (i - 1) * 0.06
            iters = np.arange(1, len(vals) + 1) + jitter
            ax.plot(iters, vals, f'{cfg["marker"]}-', color=cfg["color"],
                    linewidth=2.5, markersize=9, label=label)
            print(f"  {label} [{metric_key}]: {vals}")

        ax.set_xlabel("Outer iteration")
        ax.set_ylabel(metric_label)
        ax.set_title(f"PushT-v1 — {metric_label}", fontweight="bold")
        ax.set_xticks(range(1, MAX_ITERS + 1))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = RUNS_DIR / "pusht_three_conditions.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
