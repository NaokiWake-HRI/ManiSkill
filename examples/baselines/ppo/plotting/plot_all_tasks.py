"""All-tasks summary plot for outer-loop / eureka experiments.

For each task (latest run):
  Left:  per-iteration training curves
  Right: success_once vs iteration number (1-indexed)

Usage:
    python plot_all_tasks.py --mode outer-loop                           # single-candidate VLM+LLM
    python plot_all_tasks.py --mode eureka                               # single-candidate LLM-only
    python plot_all_tasks.py --mode outer-loop_full                      # multi-candidate VLM+LLM
    python plot_all_tasks.py --mode eureka_full                          # multi-candidate LLM-only
    python plot_all_tasks.py --mode outer-loop --seed 1788               # filter by seed
    python plot_all_tasks.py --mode outer-loop --seeds 1788 4796 9351    # aggregate across seeds
    python plot_all_tasks.py --mode outer-loop --iters 1 3 5             # show only specific iterations
    python plot_all_tasks.py --mode eureka_full --include-incomplete      # include in-progress runs
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 13})

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

MODES = ["outer-loop", "eureka", "outer-loop_full", "eureka_full"]
FULL_MODES = {"outer-loop_full", "eureka_full"}

MODE_LABELS = {
    "outer-loop": "Outer-Loop",
    "eureka": "Eureka (LLM-only)",
    "outer-loop_full": "Outer-Loop Full",
    "eureka_full": "Eureka Full",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_timestamp(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})$", name)
    return m.group(1) if m else "00000000_000000"


def _extract_seed(name: str) -> str | None:
    m = re.search(r"-(\d+)-\S+-\d{8}_\d{6}$", name)
    return m.group(1) if m else None


def _is_complete_run(run: Path) -> bool:
    return (run / "outer_loop_history.json").exists() and (run / "final_best_candidate.json").exists()


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
    mask = (steps >= start) & (steps < end)
    return steps[mask] - start, values[mask]


def _iter_color(i: int, n_iters: int):
    if i == 0:
        return (0.5, 0.5, 0.5), 0.7
    progress = (i - 1) / max(n_iters - 2, 1) if n_iters > 1 else 0
    return plt.cm.coolwarm(progress), 0.3 + 0.6 * progress


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _find_task_runs(task: str, mode: str, include_incomplete: bool = False) -> list[Path]:
    task_dir = RUNS_DIR / mode / task
    if not task_dir.is_dir():
        return []
    dirs = []
    for run in task_dir.iterdir():
        if not run.is_dir():
            continue
        is_full = mode in FULL_MODES
        if is_full:
            if not (run / "outer_loop_history.json").exists():
                continue
            if not include_incomplete and not _is_complete_run(run):
                continue
        dirs.append(run)
    return dirs


def latest_run(task: str, mode: str, seed: str | None = None,
               include_incomplete: bool = False) -> Path:
    runs = _find_task_runs(task, mode, include_incomplete=include_incomplete)
    if seed is not None:
        runs = [r for r in runs if _extract_seed(r.name) == seed]
    if not runs:
        raise FileNotFoundError(
            f"No {'(complete) ' if not include_incomplete and mode in FULL_MODES else ''}runs found for {task}"
            + (f" (seed={seed})" if seed else ""))
    runs.sort(key=lambda p: _extract_timestamp(p.name))
    return runs[-1]


def all_latest_runs(task: str, seeds: list[str], mode: str,
                    include_incomplete: bool = False) -> list[Path]:
    runs = []
    for seed in seeds:
        try:
            runs.append(latest_run(task, mode, seed=seed, include_incomplete=include_incomplete))
        except FileNotFoundError:
            print(f"  WARNING: No run found for seed {seed}, skipping")
    return runs


# ---------------------------------------------------------------------------
# TB loading
# ---------------------------------------------------------------------------

def load_tb_scalar(run_dir: str, tag: str):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    order = np.argsort(steps)
    return steps[order], values[order]


def load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "outer_loop_history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-candidate mode helpers (outer-loop, eureka)
# TB data lives in run_dir root; history has flat structure.
# ---------------------------------------------------------------------------

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


def _get_success_single(entry: dict) -> float:
    """Extract success_once from single-candidate history entry."""
    return entry.get("success_once", 0.0)


# ---------------------------------------------------------------------------
# Multi-candidate (full) mode helpers
# TB data lives in cand_X/ subdirs; history has nested best_candidate structure.
# ---------------------------------------------------------------------------

def _iter_step_range(history_entry: dict) -> tuple[int, int]:
    lc = history_entry["best_candidate"]["learning_curve"]
    if not lc:
        return 0, 0
    return lc[0]["step"], lc[-1]["step"]


def _best_cand_dir(run_dir: Path, history_entry: dict) -> Path:
    """Return the cand_X directory for the best candidate of an iteration."""
    cand_id = history_entry["best_candidate"]["candidate_id"]
    return run_dir / f"cand_{cand_id}"


def _load_tb_cached(cache: dict, cand_dir: Path, tag: str):
    """Load TB scalar from cand_dir, using cache to avoid re-reading."""
    key = str(cand_dir)
    if key not in cache:
        steps, values = load_tb_scalar(key, tag)
        cache[key] = (steps, values)
    return cache[key]


def _get_success_full(entry: dict) -> float:
    """Extract success_once from full-mode history entry."""
    return entry["best_candidate"]["eval_metrics"].get("success_once", 0.0)


# ---------------------------------------------------------------------------
# Plot functions: single-candidate mode
# ---------------------------------------------------------------------------

def plot_task_single(ax_left, ax_right, run_dir: Path, task_name: str,
                     iter_filter: list[int] | None = None):
    run_str = str(run_dir)
    metric = "train/success_once"

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
            plot_indices = list(range(n_iters))[::-1]

        for i in plot_indices:
            start, end = iter_starts[i], iter_ends[i]
            ix, iv = extract_iteration(steps, values, start, end)
            if len(ix) == 0:
                continue

            color, alpha_smooth = _iter_color(i, n_iters)
            if len(iv) > 1:
                smoothed = rolling_mean(iv, window=50)
                ax_left.plot(ix, smoothed, color=color, alpha=alpha_smooth,
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

    # Right panel
    history_path = run_dir / "outer_loop_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        iters = [e["outer_iter"] + 1 for e in history]
        success = [_get_success_single(e) for e in history]
        ax_right.plot(iters, success, "o-", color="black", linewidth=1.8, markersize=5)
        ax_right.set_xticks(iters)
    else:
        outer_steps, outer_values = load_tb_scalar(run_str, "outer_iter/success_once")
        if outer_steps is not None and len(outer_steps) > 0:
            iter_indices = np.arange(1, len(outer_values) + 1)
            ax_right.plot(iter_indices, outer_values, "o-", color="black",
                          linewidth=1.8, markersize=5)
            ax_right.set_xticks(iter_indices)

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success_once")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def plot_task_single_aggregated(ax_left, ax_right, run_dirs: list[Path], task_name: str,
                                iter_filter: list[int] | None = None):
    if not run_dirs:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    metric = "train/success_once"
    n_seeds = len(run_dirs)

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
        iter_starts = all_iter_starts[0]
        n_iters = len(iter_starts)
        iter_ends = iter_starts[1:] + [np.inf]

        if iter_filter is not None:
            plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
        else:
            plot_indices = list(range(n_iters))[::-1]

        for i in plot_indices:
            start, end = iter_starts[i], iter_ends[i]
            seed_curves = []
            for steps, values in all_steps_values:
                ix, iv = extract_iteration(steps, values, start, end)
                if len(ix) > 0 and len(iv) > 1:
                    smoothed = rolling_mean(iv, window=50)
                    seed_curves.append((ix, smoothed))

            if not seed_curves:
                continue

            ref_x = seed_curves[0][0]
            interp_curves = []
            for ix, iv in seed_curves:
                interp_y = np.interp(ref_x, ix, iv, left=np.nan, right=np.nan)
                interp_curves.append(interp_y)
            mean_curve = np.nanmean(interp_curves, axis=0)

            color, alpha_smooth = _iter_color(i, n_iters)
            ax_left.plot(ref_x, mean_curve, color=color, alpha=alpha_smooth,
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

    # Right panel
    all_iters = []
    all_success = []
    for run_dir in run_dirs:
        history_path = run_dir / "outer_loop_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            iters = [e["outer_iter"] + 1 for e in history]
            success = [_get_success_single(e) for e in history]
            all_iters.append(iters)
            all_success.append(success)
        else:
            run_str = str(run_dir)
            outer_steps, outer_values = load_tb_scalar(run_str, "outer_iter/success_once")
            if outer_steps is not None and len(outer_steps) > 0:
                iters = np.arange(1, len(outer_values) + 1)
                all_iters.append(iters.tolist())
                all_success.append(outer_values.tolist())

    if all_iters:
        for iters, success in zip(all_iters, all_success):
            ax_right.plot(iters, success, 'o', alpha=0.5, markersize=6,
                         color='gray', zorder=1)
        iters_ref = all_iters[0]
        success_array = np.array(all_success)
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


# ---------------------------------------------------------------------------
# Plot functions: multi-candidate (full) mode
# ---------------------------------------------------------------------------

def plot_task_full(ax_left, ax_right, run_dir: Path, task_name: str,
                   iter_filter: list[int] | None = None):
    history = load_history(run_dir)
    if not history:
        ax_left.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_right.transAxes)
        return

    n_iters = len(history)
    metric = "train/success_once"

    # Left panel: per-iteration training curves (best candidate, from TB)
    if iter_filter is not None:
        plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
    else:
        plot_indices = list(range(n_iters))[::-1]

    tb_cache = {}
    has_any_curve = False
    for i in plot_indices:
        start, end = _iter_step_range(history[i])
        if start == end == 0:
            continue
        cand_dir = _best_cand_dir(run_dir, history[i])
        tb_steps, tb_values = _load_tb_cached(tb_cache, cand_dir, metric)
        if tb_steps is None:
            continue
        ix, iv = extract_iteration(tb_steps, tb_values, start, end + 1)
        if len(ix) == 0:
            continue

        color, alpha = _iter_color(i, n_iters)
        if len(iv) > 1:
            smoothed = rolling_mean(iv, window=50)
            ax_left.plot(ix, smoothed, color=color, alpha=alpha,
                         linewidth=1.8, label=f"Iter {i+1}")
            has_any_curve = True

    if not has_any_curve:
        ax_left.text(0.5, 0.5, f"No {metric}", ha="center", va="center",
                     transform=ax_left.transAxes)
    else:
        ax_left.set_xlabel("Step (within iteration)")
        ax_left.set_ylabel("success_once")
        ax_left.set_xlim(0, None)
        ax_left.set_ylim(-0.05, 1.05)
        leg = ax_left.legend(loc="lower right")
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        ax_left.grid(True, alpha=0.3)

    ax_left.set_title(task_name, fontweight="bold")

    # Right panel: best candidate line + all candidates scatter
    iters = [e["outer_iter"] + 1 for e in history]
    success_once = [_get_success_full(e) for e in history]
    success_at_end = [e["best_candidate"]["eval_metrics"].get("success_at_end")
                      for e in history]
    has_success_at_end = any(v is not None for v in success_at_end)

    # All candidates scatter (success_at_end as pink squares)
    for entry in history:
        it = entry["outer_iter"] + 1
        for cand in entry.get("all_candidates", []):
            em = cand.get("eval_metrics", {})
            sae = em.get("success_at_end")
            if sae is not None:
                ax_right.plot(it, sae, "s", color="mistyrose",
                              markersize=5, alpha=0.5, zorder=1)

    # Best lines
    ax_right.plot(iters, success_once, "o-", color="black",
                  linewidth=1.8, markersize=5, label="success_once (best)", zorder=2)
    if has_success_at_end:
        ax_right.plot(iters, success_at_end, "s--", color="tab:red",
                      linewidth=1.5, markersize=5, label="success_at_end (best)", zorder=2)
    ax_right.set_xticks(iters)

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success (best)")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.legend(loc="best", fontsize=8)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


def plot_task_full_aggregated(ax_left, ax_right, run_dirs: list[Path], task_name: str,
                              iter_filter: list[int] | None = None):
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

    metric = "train/success_once"

    # Left panel
    n_iters = len(all_histories[0])
    if iter_filter is not None:
        plot_indices = [i for i in range(n_iters) if (i + 1) in iter_filter][::-1]
    else:
        plot_indices = list(range(n_iters))[::-1]

    # Per-run TB cache (keyed by cand_dir path)
    tb_caches = [{} for _ in run_dirs]

    for i in plot_indices:
        seed_curves = []
        for history, run_dir, tb_cache in zip(all_histories, run_dirs, tb_caches):
            if i >= len(history):
                continue
            start, end = _iter_step_range(history[i])
            if start == end == 0:
                continue
            cand_dir = _best_cand_dir(run_dir, history[i])
            tb_steps, tb_values = _load_tb_cached(tb_cache, cand_dir, metric)
            if tb_steps is None:
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

    # Right panel
    all_iters = []
    all_success_once = []
    all_success_at_end = []
    for history in all_histories:
        iters = [e["outer_iter"] + 1 for e in history]
        success_once = [_get_success_full(e) for e in history]
        success_at_end = [e["best_candidate"]["eval_metrics"].get("success_at_end")
                          for e in history]
        all_iters.append(iters)
        all_success_once.append(success_once)
        all_success_at_end.append(success_at_end)

        for entry in history:
            it = entry["outer_iter"] + 1
            for cand in entry.get("all_candidates", []):
                em = cand.get("eval_metrics", {})
                sae = em.get("success_at_end")
                if sae is not None:
                    ax_right.plot(it, sae, "s", alpha=0.1, markersize=3,
                                  color="mistyrose", zorder=0)

    has_sae = any(v is not None for vals in all_success_at_end for v in vals)

    if all_iters:
        for iters, s_once in zip(all_iters, all_success_once):
            ax_right.plot(iters, s_once, "o", alpha=0.5, markersize=6,
                          color="gray", zorder=1)
        iters_ref = all_iters[0]
        once_array = np.array(all_success_once)
        mean_once = np.mean(once_array, axis=0)
        ax_right.plot(iters_ref, mean_once, "o-", color="black",
                      linewidth=2.5, markersize=8,
                      label=f"success_once (n={n_seeds})", zorder=2)
        if has_sae:
            sae_array = np.array([[v if v is not None else np.nan for v in vals]
                                  for vals in all_success_at_end])
            mean_sae = np.nanmean(sae_array, axis=0)
            ax_right.plot(iters_ref, mean_sae, "s--", color="tab:red",
                          linewidth=2, markersize=7,
                          label=f"success_at_end (n={n_seeds})", zorder=2)
        ax_right.set_xticks(iters_ref)
        ax_right.legend(loc="best", fontsize=8)

    ax_right.set_xlabel("Outer iteration")
    ax_right.set_ylabel("success (best)")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title(f"{task_name} (eval)", fontweight="bold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="All-tasks summary plot")
    parser.add_argument("--mode", type=str, required=True, choices=MODES,
                        help="Run mode: outer-loop, eureka, outer-loop_full, eureka_full")
    parser.add_argument("--seed", type=str, default=None,
                        help="Filter runs by seed (e.g. 1788)")
    parser.add_argument("--seeds", nargs="+", type=str, default=None,
                        help="Aggregate across multiple seeds (e.g. 1788 4796 9351)")
    parser.add_argument("--iters", nargs="+", type=int, default=None,
                        help="Show only specific iterations (1-indexed, e.g. --iters 1 3 5)")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Include runs missing final_best_candidate.json (full modes only)")
    args = parser.parse_args()

    mode = args.mode
    is_full = mode in FULL_MODES
    iter_filter = args.iters
    if iter_filter:
        print(f"Filtering to iterations: {iter_filter}")

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
        runs = _find_task_runs(task, mode, include_incomplete=args.include_incomplete)
        if args.seeds:
            runs = [r for r in runs if _extract_seed(r.name) in args.seeds]
        elif args.seed:
            runs = [r for r in runs if _extract_seed(r.name) == args.seed]
        if runs:
            available_tasks.append(task)

    if not available_tasks:
        print(f"No runs found for mode={mode}. Check runs/{mode}/ directory.")
        if is_full:
            print("Try --include-incomplete if runs are still in progress.")
        return

    n_tasks = len(available_tasks)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 3.2 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for i, task in enumerate(available_tasks):
        print(f"Processing {task}...")
        try:
            if plot_mode == "aggregate":
                run_dirs = all_latest_runs(task, seeds, mode,
                                           include_incomplete=args.include_incomplete)
                if not run_dirs:
                    raise FileNotFoundError("No runs found for specified seeds")
                print(f"  Found {len(run_dirs)} runs")
                if is_full:
                    plot_task_full_aggregated(axes[i, 0], axes[i, 1], run_dirs, task,
                                             iter_filter=iter_filter)
                else:
                    plot_task_single_aggregated(axes[i, 0], axes[i, 1], run_dirs, task,
                                               iter_filter=iter_filter)
            else:
                run_dir = latest_run(task, mode,
                                     seed=args.seed if plot_mode == "single" else None,
                                     include_incomplete=args.include_incomplete)
                print(f"  Latest run: {run_dir.name}")
                if is_full:
                    plot_task_full(axes[i, 0], axes[i, 1], run_dir, task,
                                  iter_filter=iter_filter)
                else:
                    plot_task_single(axes[i, 0], axes[i, 1], run_dir, task,
                                    iter_filter=iter_filter)
        except Exception as e:
            print(f"  ERROR: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error: {e}", ha="center",
                            va="center", transform=axes[i, 0].transAxes,
                            fontsize=7, color="red")
            axes[i, 0].set_title(task, fontweight="bold")
            axes[i, 1].set_title(task, fontweight="bold")

    # Title and filename
    mode_label = MODE_LABELS[mode]
    if plot_mode == "aggregate":
        title_suffix = f" (n={len(seeds)} seeds)"
        file_suffix = f"_agg_n{len(seeds)}"
    elif plot_mode == "single":
        title_suffix = f" (seed {args.seed})"
        file_suffix = f"_seed{args.seed}"
    else:
        title_suffix = ""
        file_suffix = ""

    fig.suptitle(f"{mode_label} Results{title_suffix}",
                 fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    out_dir = RUNS_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary{file_suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
