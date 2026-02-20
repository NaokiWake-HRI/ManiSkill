"""Outer-loop full (VLM+LLM reward generation) summary plot. Works mid-run.

Usage:
    python plot_outer_loop_full_summary.py                          # latest PushCube run
    python plot_outer_loop_full_summary.py --env PushCube-v1        # specific env
    python plot_outer_loop_full_summary.py --run_dir runs/outer-loop_full/PushCube-v1/...  # specific run
"""

import argparse
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTER_LOOP_DIR = Path(__file__).parent / "runs" / "outer-loop_full"


def latest_run(task: str) -> Path:
    task_dir = OUTER_LOOP_DIR / task
    runs = sorted(task_dir.iterdir(), key=lambda p: p.name)
    return runs[-1]


def rolling_mean(values, window=50):
    window = min(window, len(values))
    if window <= 1:
        return values.copy()
    cumsum = np.cumsum(values)
    out = np.empty_like(values)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def load_tb_scalar(tb_dir: str, tag: str):
    """Load a scalar from tensorboard events."""
    try:
        ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None, None
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    except Exception:
        return None, None


def detect_iter_ranges_from_lc(entry: dict):
    """Get step range for an iteration from learning_curve (works for both history entry and result JSON)."""
    # History entry: best_candidate.learning_curve
    if "best_candidate" in entry:
        lc = entry["best_candidate"].get("learning_curve", [])
    else:
        # Result JSON: learning_curve directly
        lc = entry.get("learning_curve", [])
    if not lc:
        return None, None
    return lc[0]["step"], lc[-1]["step"]


def _plot_candidate_tb(ax, run_dir, cand_id, step_start, step_end, metric, color,
                       linewidth, alpha, label):
    """Plot a single candidate's TB data. Returns True if successful."""
    tb_dir = str(run_dir / f"cand_{cand_id}")
    if not Path(tb_dir).exists():
        return False
    steps, values = load_tb_scalar(tb_dir, metric)
    if steps is None or len(steps) == 0:
        return False
    mask = (steps >= step_start) & (steps <= step_end + 100000)
    steps_iter = steps[mask]
    values_iter = values[mask]
    if len(steps_iter) == 0:
        return False
    steps_rel = steps_iter - steps_iter[0]
    smoothed = rolling_mean(values_iter, window=50)
    ax.plot(steps_rel, smoothed, color=color, linewidth=linewidth, alpha=alpha, label=label)
    return True


def _plot_candidate_lc(ax, lc, color, linewidth, alpha, label, marker="o-"):
    """Fallback: plot from learning_curve JSON data."""
    if not lc:
        return
    steps = [p["step"] for p in lc]
    success = [p["success_once"] for p in lc]
    s0 = steps[0]
    steps_rel = [s - s0 for s in steps]
    ax.plot(steps_rel, success, marker, color=color, linewidth=linewidth,
            markersize=4, alpha=alpha, label=label)


def plot_summary(run_dir: Path, out_path: Path | None = None):
    run_dir = Path(run_dir)
    env_name = run_dir.parent.name

    # Load history (completed iters)
    history_path = run_dir / "outer_loop_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # Load current iter's result JSONs (may not yet be in history)
    result_files = sorted(glob.glob(str(run_dir / "_cand_*_result.json")))
    current_results = []
    for rf in result_files:
        with open(rf) as f:
            current_results.append(json.load(f))

    # Determine step range of the last completed iteration to filter stale results
    if history:
        last_lc = history[-1]["best_candidate"].get("learning_curve", [])
        last_hist_start = last_lc[0]["step"] if last_lc else 0
    else:
        last_hist_start = 0

    # Filter current results: only keep those whose step range is BEYOND the last history entry
    # (i.e., belongs to an iteration not yet recorded in history)
    fresh_results = []
    for cr in current_results:
        lc = cr.get("learning_curve", [])
        if lc and lc[0]["step"] > last_hist_start and cr.get("success"):
            fresh_results.append(cr)

    current_is_new = len(fresh_results) > 0

    print(f"History iterations: {len(history)}")
    print(f"Current result JSONs: {len(current_results)} total, {len(fresh_results)} fresh (new_iter={current_is_new})")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Left: Learning curves from tensorboard (dense) ===
    ax = axes[0]
    n_total = len(history) + (1 if current_is_new else 0)
    metric = "train/success_once"

    # --- History iterations: best candidate (thick) + other candidates (thin) ---
    for i, entry in enumerate(history):
        bc = entry["best_candidate"]
        best_cand_id = bc["candidate_id"]
        step_start, step_end = detect_iter_ranges_from_lc(entry)

        if i == 0:
            base_color = (0.5, 0.5, 0.5)
        else:
            progress = (i - 1) / max(n_total - 2, 1)
            base_color = plt.cm.coolwarm(progress)

        # Plot all candidates thin first
        all_cands = entry.get("all_candidates", [])
        for c in all_cands:
            cid = c["candidate_id"]
            if cid == best_cand_id:
                continue  # plot best separately
            c_step_start, c_step_end = None, None
            c_lc = c.get("learning_curve", [])
            if c_lc:
                c_step_start = c_lc[0]["step"]
                c_step_end = c_lc[-1]["step"]
            if c_step_start is not None:
                plotted = _plot_candidate_tb(
                    ax, run_dir, cid, c_step_start, c_step_end, metric,
                    color=base_color, linewidth=0.6, alpha=0.25, label=None)
                if not plotted:
                    _plot_candidate_lc(ax, c_lc, color=base_color, linewidth=0.6,
                                       alpha=0.25, label=None, marker="-")

        # Plot best candidate thick
        if step_start is not None:
            plotted = _plot_candidate_tb(
                ax, run_dir, best_cand_id, step_start, step_end, metric,
                color=base_color, linewidth=1.8, alpha=0.8,
                label=f"Iter {i+1} (cand {best_cand_id})")
            if not plotted:
                lc = bc.get("learning_curve", [])
                _plot_candidate_lc(ax, lc, color=base_color, linewidth=2,
                                   alpha=0.8, label=f"Iter {i+1} (cand {best_cand_id})")

    # --- Current iter: all fresh candidates (thin), best (thick) ---
    if current_is_new and fresh_results:
        best_current = max(fresh_results, key=lambda r: r["eval_metrics"]["success_once"])
        best_cand_id = best_current["candidate_id"]
        cur_iter_label = len(history) + 1

        for cr in fresh_results:
            cid = cr["candidate_id"]
            lc = cr.get("learning_curve", [])
            if not lc:
                continue
            step_start = lc[0]["step"]
            step_end = lc[-1]["step"]
            is_best = (cid == best_cand_id)

            lw = 1.8 if is_best else 0.6
            alpha = 0.9 if is_best else 0.3
            label = f"Iter {cur_iter_label}* (cand {cid})" if is_best else None

            plotted = _plot_candidate_tb(
                ax, run_dir, cid, step_start, step_end, metric,
                color="blue", linewidth=lw, alpha=alpha, label=label)
            if not plotted:
                _plot_candidate_lc(ax, lc, color="blue", linewidth=lw,
                                   alpha=alpha, label=label, marker="-")

    ax.set_xlabel("Step (within iteration)")
    ax.set_ylabel("success_once")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{env_name}: Learning Curves (bold=best, thin=others)", fontweight="bold")
    leg = ax.legend(loc="lower right", fontsize=9)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    ax.grid(True, alpha=0.3)

    # === Right: success_once per outer iteration ===
    ax2 = axes[1]
    iters_done = [e["outer_iter"] + 1 for e in history]
    success_done = [e["best_candidate"]["eval_metrics"]["success_once"] for e in history]

    # All candidates as scatter
    for entry in history:
        all_cands = entry.get("all_candidates", [])
        for c in all_cands:
            em = c.get("eval_metrics", {})
            s = em.get("success_once", 0)
            ax2.plot(entry["outer_iter"] + 1, s, "o", color="lightgray",
                     markersize=6, alpha=0.6, zorder=1)

    # Best line
    if iters_done:
        ax2.plot(iters_done, success_done, "o-", color="black", linewidth=2.5,
                 markersize=8, label="Best candidate", zorder=3)

    # Current iter candidates (fresh only)
    if current_is_new and fresh_results:
        cur_iter = len(history) + 1
        for cr in fresh_results:
            s = cr["eval_metrics"]["success_once"]
            ax2.plot(cur_iter, s, "o", color="lightblue", markersize=6,
                     alpha=0.7, zorder=1)
        best_s = max(cr["eval_metrics"]["success_once"] for cr in fresh_results)
        ax2.plot(cur_iter, best_s, "s", color="blue", markersize=10,
                 label=f"Iter {cur_iter}* (in-progress)", zorder=3)

    ax2.set_xlabel("Outer Iteration")
    ax2.set_ylabel("success_once (best)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"{env_name}: Outer Loop Progress", fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    max_iter = len(history) + (1 if current_is_new else 0)
    ax2.set_xticks(range(1, max_iter + 1))

    fig.suptitle(f"Outer-Loop Full - {env_name} ({run_dir.name})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if out_path is None:
        out_path = run_dir / "outer_loop_full_summary.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Outer-loop full summary plot")
    parser.add_argument("--env", type=str, default="PushCube-v1")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Specific run directory (overrides --env)")
    parser.add_argument("--out", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run(args.env)
        print(f"Latest run: {run_dir}")

    plot_summary(run_dir, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
