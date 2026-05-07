#!/usr/bin/env python3
"""Combined self-report multiturn plots averaged across models.

Produces:
  1. Averaged wellbeing-over-turns line plot (one line per condition)
  2. Averaged mean-wellbeing bar chart (one bar per condition)

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.plot_self_report_multiturn_combined \
        --results-dir outputs/self_report_multiturn/qwen25-32b-instruct --model qwen25-32b-instruct \
        --results-dir outputs/self_report_multiturn/llama-33-70b-instruct --model llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

from typing import Optional

from superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.plot_self_report_multiturn_results import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    _ERRBAR_KW,
    _sem,
    compute_wellbeing_by_turn,
    load_conversations,
    mean_wellbeing_per_rep,
)


def plot_wellbeing_avg(
    model_data: List[Tuple[str, Dict[str, List[dict]]]],
    max_turns: int,
    out_path: Path,
) -> None:
    """Line plot of wellbeing over turns, averaged across models."""
    with plt.rc_context({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
    }):
        fig, ax = plt.subplots(figsize=(4, 3))
        turns = np.arange(1, max_turns + 1)

        for cond in CONDITION_ORDER:
            curves = []
            for _, data in model_data:
                if cond in data:
                    means, _ = compute_wellbeing_by_turn(data[cond], max_turns)
                    curves.append(means)
            if not curves:
                continue
            avg = np.nanmean(curves, axis=0)
            valid = ~np.isnan(avg)
            ax.plot(
                turns[valid], avg[valid],
                color=CONDITION_COLORS.get(cond, "#c0c0c0"),
                linewidth=1.5,
                label=CONDITION_LABELS.get(cond, cond),
                marker="o",
                markersize=3,
            )

        ax.set_xlabel("Turn")
        ax.set_ylabel("Score")
        ax.set_title("Self-Report over Turns", pad=40)
        ax.set_xlim(0.5, max_turns + 0.5)
        ax.set_ylim(1, 7)
        ax.set_yticks(range(1, 8))
        ax.set_xticks(range(1, max_turns + 1))
        ax.legend(
            loc="lower center", ncol=len(CONDITION_ORDER),
            bbox_to_anchor=(0.5, 1.03), fontsize=9,
            frameon=True, edgecolor="#c0c0c0", fancybox=False,
            columnspacing=1.0, handlelength=1.5,
        )
        ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved avg wellbeing-over-turns plot to {out_path}")


def plot_mean_wellbeing_avg(
    model_data: List[Tuple[str, Dict[str, List[dict]]]],
    max_turns: int,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Bar chart of overall mean wellbeing per condition, averaged across models."""
    labels = []
    means = []
    errs = []
    colors = []

    for cond in CONDITION_ORDER:
        per_model_means = []
        per_model_sems = []
        for _, data in model_data:
            if cond not in data:
                continue
            mean_val, sem_val = mean_wellbeing_per_rep(data[cond])
            if mean_val != 0.0:
                per_model_means.append(mean_val)
                per_model_sems.append(sem_val)
        if per_model_means:
            labels.append(CONDITION_LABELS.get(cond, cond))
            means.append(float(np.mean(per_model_means)))
            errs.append(float(np.mean(per_model_sems)))
            colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(3, 2))
    x = np.arange(len(labels))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, means, 0.65, **bar_kw)
    for i, v in enumerate(means):
        offset = errs[i] + 0.05 if errbar_mode == "rep_sem" else 0.05
        ax.text(i, v + offset, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Self-Report \u2014 Multiturn Interactions", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(1, 7)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved avg mean-wellbeing plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot self-report multiturn results averaged across models.",
    )
    parser.add_argument(
        "--results-dir", type=Path, action="append", required=True,
        help="Results directory for one model (can be repeated)",
    )
    parser.add_argument(
        "--model", type=str, action="append", required=True,
        help="Model key (matched 1:1 with --results-dir)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Max turns for x-axis (default: inferred from data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: consolidated/)",
    )
    args = parser.parse_args()

    if len(args.results_dir) != len(args.model):
        print("ERROR: --results-dir and --model must be specified the same number of times.", file=sys.stderr)
        return 1

    model_data: List[Tuple[str, Dict[str, List[dict]]]] = []
    global_max_turns = args.max_turns or 0

    for results_dir, model_key in zip(args.results_dir, args.model):
        results_dir = results_dir.expanduser().resolve()
        if not results_dir.exists():
            print(f"WARNING: Skipping {model_key} — {results_dir} does not exist", file=sys.stderr)
            continue

        data = load_conversations(results_dir)
        if not data:
            print(f"WARNING: No data for {model_key}", file=sys.stderr)
            continue

        if args.max_turns is None:
            mt = max(
                r.get("n_turns_target", 10)
                for records in data.values()
                for r in records
            )
            global_max_turns = max(global_max_turns, mt)

        found = [c for c in CONDITION_ORDER if c in data]
        print(f"{model_key}: conditions={found}")

        model_data.append((model_key, data))

    if not model_data:
        print("ERROR: No model data found.", file=sys.stderr)
        return 1

    if global_max_turns == 0:
        global_max_turns = 10

    out_dir = args.output_dir or (args.results_dir[0].expanduser().resolve().parent.parent / "consolidated")

    plot_wellbeing_avg(model_data, global_max_turns, out_dir / "self_report_multiturn_wellbeing_avg.png")
    plot_mean_wellbeing_avg(model_data, global_max_turns, out_dir / "self_report_multiturn_mean_wellbeing_avg.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
