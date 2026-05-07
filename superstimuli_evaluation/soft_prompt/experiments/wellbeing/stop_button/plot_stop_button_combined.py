#!/usr/bin/env python3
"""Plot stop button results for multiple models side by side.

Two combined plots:
  1. Survival curves — one subplot per model, placed side by side
  2. Completion bar charts — one subplot per model, placed side by side

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.plot_stop_button_combined \
        --results-dir outputs/stop_button/qwen25-32b-instruct --model qwen25-32b-instruct \
        --results-dir outputs/stop_button/llama-33-70b-instruct --model llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

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

from superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.plot_stop_button_results import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_LINESTYLES,
    CONDITION_ORDER,
    _ERRBAR_KW,
    _sem,
    completion_rate_per_rep,
    compute_completion_rate,
    compute_survival,
    load_conversations,
)


def _resolve_display_name(model_key: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        return get_model_display_name(model_key)
    except Exception:
        return model_key


def plot_survival_combined(
    model_data: List[Tuple[str, dict]],
    max_turns: int,
    out_path: Path,
) -> None:
    """Survival curves for all models, one subplot per model."""
    n_models = len(model_data)
    fig_w = 3 if n_models == 1 else 3 * n_models
    fig_h = 2
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(fig_w, fig_h),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for idx, (model_display, data) in enumerate(model_data):
        ax = axes[idx]
        for cond in CONDITION_ORDER:
            if cond not in data:
                continue
            records = data[cond]
            survival = compute_survival(records, max_turns)
            turns = np.arange(max_turns + 1)
            ax.plot(
                turns, survival,
                color=CONDITION_COLORS.get(cond, "#c0c0c0"),
                linestyle=CONDITION_LINESTYLES.get(cond, "-"),
                linewidth=1.5,
                label=CONDITION_LABELS.get(cond, cond),
                marker="o",
                markersize=3,
            )

        ax.set_xlabel("Turn")
        ax.set_title(model_display, pad=20)
        ax.set_xlim(0, max_turns)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(range(0, max_turns + 1))
        ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_ylabel("Fraction Remaining")

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Stop Button Survival Curves",
        fontsize=12, y=1.10,
    )
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(6, len(labels)),
        fontsize=9,
        frameon=True, edgecolor="#c0c0c0", fancybox=False,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.2,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined survival plot to {out_path}")


def plot_completion_combined(
    model_data: List[Tuple[str, dict]],
    max_turns: int,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Completion bar charts for all models, one subplot per model."""
    n_models = len(model_data)
    fig_w = 3 if n_models == 1 else 3 * n_models
    fig_h = 2
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(fig_w, fig_h),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for idx, (model_display, data) in enumerate(model_data):
        ax = axes[idx]
        labels = []
        rates = []
        errs = []
        colors = []

        for cond in CONDITION_ORDER:
            if cond not in data:
                continue
            records = data[cond]
            mean_rate, sem_rate = completion_rate_per_rep(records)
            labels.append(CONDITION_LABELS.get(cond, cond))
            rates.append(mean_rate)
            errs.append(sem_rate)
            colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

        if not labels:
            continue

        x = np.arange(len(labels))
        bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
        if errbar_mode == "rep_sem":
            bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
        bars = ax.bar(x, rates, 0.65, **bar_kw)
        for i, r in enumerate(rates):
            offset = errs[i] + 2 if errbar_mode == "rep_sem" else 2
            ax.text(i, r + offset, f"{r:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(model_display, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in labels])
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_ylabel(f"% Active after {max_turns} Turns")

    fig.suptitle(
        "Conversation Survival Rate",
        fontsize=12, y=1.10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined completion plot to {out_path}")


def plot_survival_avg(
    model_data: List[Tuple[str, dict]],
    max_turns: int,
    out_path: Path,
) -> None:
    """Survival curves averaged across models, one line per condition."""
    with plt.rc_context({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
    }):
        fig, ax = plt.subplots(figsize=(4, 3))
        turns = np.arange(max_turns + 1)

        for cond in CONDITION_ORDER:
            curves = []
            for _, data in model_data:
                if cond in data:
                    curves.append(compute_survival(data[cond], max_turns))
            if not curves:
                continue
            avg_surv = np.mean(curves, axis=0) * 100
            ax.plot(
                turns, avg_surv,
                color=CONDITION_COLORS.get(cond, "#c0c0c0"),
                linestyle=CONDITION_LINESTYLES.get(cond, "-"),
                linewidth=1.5,
                label=CONDITION_LABELS.get(cond, cond),
                marker="o",
                markersize=3,
            )

        ax.set_xlabel("Turn")
        ax.set_ylabel("1 - Stop Button Rate")
        ax.set_title("Dialogue Continuation Curve", pad=40)
        ax.set_xlim(0, max_turns)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, max_turns + 1))
        ax.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.03), fontsize=9, frameon=True, edgecolor="#c0c0c0", fancybox=False, columnspacing=1.0, handlelength=1.5)
        ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved avg survival plot to {out_path}")


def plot_completion_avg(
    model_data: List[Tuple[str, dict]],
    max_turns: int,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Completion bar chart averaged across models, one bar per condition."""
    cond_labels = []
    avg_rates = []
    errs = []
    colors = []

    for cond in CONDITION_ORDER:
        per_model_means = []
        per_model_sems = []
        for _, data in model_data:
            if cond in data:
                mean_rate, sem_rate = completion_rate_per_rep(data[cond])
                per_model_means.append(mean_rate)
                per_model_sems.append(sem_rate)
        if not per_model_means:
            continue
        cond_labels.append(CONDITION_LABELS.get(cond, cond))
        avg_rates.append(float(np.mean(per_model_means)))
        errs.append(float(np.mean(per_model_sems)))
        colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not cond_labels:
        return

    x = np.arange(len(cond_labels))
    fig, ax = plt.subplots(figsize=(3, 2))

    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, avg_rates, 0.65, **bar_kw)
    for i, r in enumerate(avg_rates):
        offset = errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, r + offset, f"{r:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("1 - Stop Button Rate")
    ax.set_title("Dialogue Continuation", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in cond_labels])
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved avg completion plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot stop button results for multiple models side by side.",
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
        help="Max turns for survival curve x-axis (default: inferred from data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: first results-dir's parent / combined_plots)",
    )
    parser.add_argument(
        "--prefix", type=str, default="stop_button",
        help="Filename prefix for output plots (default: stop_button)",
    )
    args = parser.parse_args()

    if len(args.results_dir) != len(args.model):
        print("ERROR: --results-dir and --model must be specified the same number of times.", file=sys.stderr)
        return 1

    model_data: List[Tuple[str, dict]] = []
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

        display = _resolve_display_name(model_key)

        # Infer max turns from data
        if args.max_turns is None:
            mt = max(
                r.get("n_turns_target", 10)
                for records in data.values()
                for r in records
            )
            global_max_turns = max(global_max_turns, mt)

        found = [c for c in CONDITION_ORDER if c in data]
        print(f"{display}: conditions={found}")
        for cond in found:
            n = len(data[cond])
            rate = compute_completion_rate(data[cond])
            print(f"  {cond}: {n} conversations, completion rate = {rate:.1%}")

        model_data.append((display, data))

    if not model_data:
        print("ERROR: No model data found.", file=sys.stderr)
        return 1

    if global_max_turns == 0:
        global_max_turns = 10

    out_dir = args.output_dir or (args.results_dir[0].expanduser().resolve().parent.parent / "consolidated")

    prefix = args.prefix
    plot_survival_avg(model_data, global_max_turns, out_dir / f"{prefix}_survival_avg.png")
    plot_completion_avg(model_data, global_max_turns, out_dir / f"{prefix}_completion_avg.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
