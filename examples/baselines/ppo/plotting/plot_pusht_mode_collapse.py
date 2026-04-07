"""Plot a slide-ready PushT figure showing plateau and reward-family collapse.

Usage:
    python plot_pusht_mode_collapse.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

DEFAULT_RUN1 = RUNS_DIR / (
    "outer-loop_full_failureselection_failure_and_near_miss_k_16/PushT-v1/"
    "ppo-vlm-full-failureselection-PushT-v1-9351-PushT-v1-20260402_194929/"
    "outer_loop_history.json"
)
DEFAULT_RUN2 = RUNS_DIR / (
    "outer-loop_full_failureselection_failure_and_near_miss_k_16/PushT-v1/"
    "ppo-vlm-full-failureselection-PushT-v1-9351-PushT-v1-20260401_093535_resume1_20260406_213015/"
    "outer_loop_history.json"
)
DEFAULT_OUT = RUNS_DIR / "pusht_mode_collapse_story.png"


# Lightweight reward-family proxy over code motifs.
FAMILY_TERMS: list[tuple[str, str]] = [
    ("sigmoid", "gate"),
    ("precision", "precision"),
    ("retreat", "retreat"),
    ("release", "release"),
    ("overlap", "overlap"),
    ("settle", "settle"),
    ("basin", "basin"),
    ("softmin", "softmin"),
    ("couple", "couple"),
    ("near_goal", "near_goal"),
]


def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def best_success_curve(history: list[dict]) -> list[float]:
    out = []
    for record in history:
        best = record["best_candidate"]
        out.append(best.get("fitness_success_at_end", best.get("fitness", 0.0)))
    return out


def family_signature(code: str | None) -> tuple[str, ...]:
    code_l = (code or "").lower()
    tags = tuple(label for needle, label in FAMILY_TERMS if needle in code_l)
    if not tags:
        return ("plain",)
    return tags


def collect_candidates(history: list[dict]) -> list[list[dict]]:
    per_iter: list[list[dict]] = []
    for record in history:
        best_id = record.get("best_candidate", {}).get("candidate_id", -1)
        items = []
        for idx, cand in enumerate(record.get("all_candidates", [])):
            candidate_id = cand.get("candidate_id", idx)
            items.append(
                {
                    "sig": family_signature(cand.get("code")),
                    "candidate_id": candidate_id,
                    "is_best": candidate_id == best_id,
                }
            )
        per_iter.append(items)
    return per_iter


def build_family_index(*run_items: list[list[dict]]) -> dict[tuple[str, ...], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    for per_iter in run_items:
        for items in per_iter:
            counts.update(item["sig"] for item in items)
    ordered = [sig for sig, _ in counts.most_common()]
    return {sig: idx for idx, sig in enumerate(ordered)}


def build_heatmap_matrix(
    per_iter: list[list[dict]], family_to_idx: dict[tuple[str, ...], int]
) -> tuple[np.ndarray, list[int], list[float], list[int | None]]:
    max_rows = max(len(items) for items in per_iter)
    n_iters = len(per_iter)
    mat = np.full((max_rows, n_iters), np.nan)
    unique_counts: list[int] = []
    top2_share: list[float] = []
    best_rows: list[int | None] = []
    for col, items in enumerate(per_iter):
        sigs = [item["sig"] for item in items]
        counts = Counter(sigs)
        unique_counts.append(len(counts))
        most_common = [c for _, c in counts.most_common(2)]
        top2_share.append(sum(most_common) / max(len(sigs), 1))
        ordered_items = sorted(
            (
                {
                    **item,
                    "family_idx": family_to_idx[item["sig"]],
                }
                for item in items
            ),
            key=lambda item: (item["family_idx"], item["candidate_id"]),
        )
        best_row = None
        for row, item in enumerate(ordered_items):
            mat[row, col] = item["family_idx"]
            if item["is_best"]:
                best_row = row
        best_rows.append(best_row)
    return mat, unique_counts, top2_share, best_rows


def signature_label(sig: tuple[str, ...]) -> str:
    if sig == ("plain",):
        return "plain"
    return "+".join(sig)


def make_cmap(n: int) -> ListedColormap:
    base = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
    colors = [base[i % len(base)] for i in range(n)]
    cmap = ListedColormap(colors)
    cmap.set_bad("#f2f2f2")
    return cmap


def plot(
    run1_history: list[dict],
    run2_history: list[dict],
    out_path: Path,
) -> None:
    run1_items = collect_candidates(run1_history)
    run2_items = collect_candidates(run2_history)
    family_to_idx = build_family_index(run1_items, run2_items)
    idx_to_family = {idx: sig for sig, idx in family_to_idx.items()}
    cmap = make_cmap(len(family_to_idx))

    run1_heat, run1_unique, run1_top2, run1_best_rows = build_heatmap_matrix(run1_items, family_to_idx)
    run2_heat, run2_unique, run2_top2, run2_best_rows = build_heatmap_matrix(run2_items, family_to_idx)

    run1_success = best_success_curve(run1_history)
    run2_success = best_success_curve(run2_history)

    fig = plt.figure(figsize=(16, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.0], wspace=0.28)

    ax_curve = fig.add_subplot(gs[0, 0])
    ax_run1 = fig.add_subplot(gs[0, 1])
    ax_run2 = fig.add_subplot(gs[0, 2])

    x1 = np.arange(1, len(run1_success) + 1)
    x2 = np.arange(1, len(run2_success) + 1)
    ax_curve.plot(x1, run1_success, marker="^", linewidth=3, markersize=11, color="#2ca25f", label="Run1 (fresh iter0)")
    ax_curve.plot(x2, run2_success, marker="o", linewidth=3, markersize=11, color="#e67e22", label="Run2 (failure iter0)")
    ax_curve.axvspan(3, 6.25, color="#fce5d6", alpha=0.55)
    ax_curve.annotate(
        "Run2 plateaus at 0.8125",
        xy=(4.0, 0.8125),
        xytext=(3.15, 0.67),
        arrowprops={"arrowstyle": "->", "color": "#9c4f14", "lw": 1.5},
        color="#9c4f14",
        fontsize=11,
    )
    ax_curve.set_title("Success plateau", fontsize=16, weight="bold")
    ax_curve.set_xlabel("Outer iteration", fontsize=13)
    ax_curve.set_ylabel("success_at_end", fontsize=13)
    ax_curve.set_xlim(0.7, 6.3)
    ax_curve.set_ylim(0.2, 1.02)
    ax_curve.grid(True, alpha=0.28)
    ax_curve.legend(loc="lower right", frameon=True, fontsize=11)

    def draw_heatmap(
        ax,
        mat: np.ndarray,
        unique_counts: list[int],
        top2_share: list[float],
        best_rows: list[int | None],
        title: str,
    ):
        ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(len(family_to_idx) - 1, 1))
        ax.set_title(title, fontsize=16, weight="bold")
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(np.arange(1, mat.shape[1] + 1), fontsize=11)
        ax.set_yticks([0, mat.shape[0] - 1])
        ax.set_yticklabels(["1", str(mat.shape[0])], fontsize=11)
        ax.set_ylabel("Candidates\n(sorted within iter)", fontsize=12)
        ax.set_ylim(mat.shape[0] - 0.5, -1.45)
        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for x, (u, c) in enumerate(zip(unique_counts, top2_share)):
            ax.text(x, -0.92, str(u), ha="center", va="center", fontsize=10, fontweight="bold")
        for col, row in enumerate(best_rows):
            if row is None:
                continue
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=3.6,
                )
            )
    draw_heatmap(ax_run1, run1_heat, run1_unique, run1_top2, run1_best_rows, "Run1 reward families")
    draw_heatmap(ax_run2, run2_heat, run2_unique, run2_top2, run2_best_rows, "Run2 reward families")

    dominant_run2 = Counter(item["sig"] for items in run2_items[3:] for item in items).most_common(2)
    dominant_text = ", ".join(signature_label(sig) for sig, _ in dominant_run2)
    fig.suptitle("PushT: plateau appears when reward-search diversity collapses", fontsize=22, weight="bold", y=0.98)
    fig.subplots_adjust(top=0.87, bottom=0.17)
    fig.text(
        0.565,
        0.082,
        f"Late Run2 is dominated by: {dominant_text}",
        ha="center",
        fontsize=12,
        color="#444444",
    )
    fig.text(0.5, 0.12, "Red box: best candidate selected in that iteration", ha="center", fontsize=11, color="#b22222")
    fig.text(
        0.5,
        0.03,
        "Reward family is a lightweight code-structure proxy over motifs "
        "{gate, precision, retreat, release, overlap, settle, basin, softmin, couple, near_goal}.",
        ha="center",
        fontsize=10,
        color="#666666",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, default=DEFAULT_RUN1)
    parser.add_argument("--run2", type=Path, default=DEFAULT_RUN2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run1_history = load_history(args.run1)
    run2_history = load_history(args.run2)
    plot(run1_history, run2_history, args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
