#!/usr/bin/env python3
"""Plot self-report multiturn evaluation results.

Plots wellbeing score (1-7) at each conversation turn, comparing
baseline vs intervention conditions.

Reads per-conversation JSON files from the output directory structure:

    <results_dir>/<stimulant_type>/<timestamp>/conversations/<scenario>_<condition>_rep<rep>.json

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.plot_self_report_multiturn_results \
        --results-dir outputs/self_report_multiturn/qwen3-30b-a3b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

# ── Condition styling ──────────────────────────────────────────────────────

CONDITION_LABELS = {
    "baseline": "No Soft Prompt",
    "euphorics": "Euphorics",
}

CONDITION_ORDER = [
    "baseline",
    "euphorics",
]

CONDITION_COLORS = {
    "euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}


# ── Error bar helpers ────────────────────────────────────────────────────

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    """Standard error of the mean."""
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def mean_wellbeing_per_rep(records: List[dict]) -> Tuple[float, float]:
    """Compute mean wellbeing and SEM across reps.

    Each conversation is averaged across turns first, then grouped by rep.
    Returns (mean_across_reps, sem_across_reps).
    """
    by_rep: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        rep = r.get("rep", 0)
        turn_vals = [wb for wb in r.get("per_turn_wellbeing", []) if wb is not None]
        if turn_vals:
            by_rep[rep].append(float(np.mean(turn_vals)))
    rep_means = [float(np.mean(conv_means)) for _, conv_means in sorted(by_rep.items()) if conv_means]
    if not rep_means:
        return 0.0, 0.0
    return float(np.mean(rep_means)), _sem(rep_means)


# ── Data loading ───────────────────────────────────────────────────────────


def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    """Return the latest subdir named YYYYMMDD_HHMMSS, or None."""
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_conversations(results_dir: Path) -> Dict[str, List[dict]]:
    """Load all conversation records, grouped by stimulant_type.

    Each stimulant_type subdir has timestamp dirs with conversations/ inside.
    Records with condition="baseline" are grouped under "baseline";
    records with condition="intervention" are grouped under the stimulant dir name.

    Returns: {stimulant_key: [record, ...]}
    """
    by_stimulant: Dict[str, List[dict]] = defaultdict(list)

    for stim_dir in sorted(results_dir.iterdir()):
        if not stim_dir.is_dir():
            continue
        stim_key = stim_dir.name

        latest = find_latest_timed_dir(stim_dir)
        if latest is None:
            continue
        conv_dir = latest / "conversations"
        if not conv_dir.is_dir():
            continue

        for conv_file in sorted(conv_dir.glob("*.json")):
            try:
                with open(conv_file) as f:
                    record = json.load(f)
            except Exception:
                continue

            by_stimulant[stim_key].append(record)

    return dict(by_stimulant)


# ── Wellbeing-by-turn computation ─────────────────────────────────────────


def compute_wellbeing_by_turn(
    records: List[dict], max_turns: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and SEM of wellbeing at each turn.

    Returns (means, sems) arrays of length max_turns (1-indexed stored at index t-1).
    """
    by_turn: Dict[int, List[float]] = defaultdict(list)

    for r in records:
        per_turn_wb = r.get("per_turn_wellbeing", [])
        for t_idx, wb in enumerate(per_turn_wb):
            if wb is not None:
                by_turn[t_idx + 1].append(wb)

    means = np.full(max_turns, np.nan)
    sems = np.full(max_turns, np.nan)
    for t in range(1, max_turns + 1):
        vals = by_turn.get(t, [])
        if vals:
            means[t - 1] = np.mean(vals)
            sems[t - 1] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

    return means, sems


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_wellbeing_over_turns(
    data: Dict[str, List[dict]],
    max_turns: int,
    model_display: str,
    out_path: Path,
) -> None:
    """Line plot: mean wellbeing (1-7) at each turn, with SEM shading."""
    fig, ax = plt.subplots(figsize=(4, 3))
    turns = np.arange(1, max_turns + 1)

    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        records = data[cond]
        means, sems = compute_wellbeing_by_turn(records, max_turns)
        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, "#c0c0c0")

        valid = ~np.isnan(means)
        ax.plot(
            turns[valid], means[valid],
            color=color,
            linewidth=1.5,
            label=label,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    ax.set_title(f"Self-Report over Turns\n{model_display}", fontsize=9, pad=20)
    ax.set_xlim(0.5, max_turns + 0.5)
    ax.set_ylim(1, 7)
    ax.set_yticks(range(1, 8))
    ax.set_xticks(range(1, max_turns + 1))
    ax.axhline(y=4, color="#c0c0c0", linestyle="--", linewidth=0.6, zorder=0)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(CONDITION_ORDER), frameon=False, fontsize=6,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wellbeing-over-turns plot to {out_path}")


def plot_wellbeing_by_category(
    data: Dict[str, List[dict]],
    max_turns: int,
    model_display: str,
    out_path: Path,
) -> None:
    """Grouped bar chart: mean wellbeing across all turns, per category."""
    cat_info: Dict[str, str] = {}
    for records in data.values():
        for r in records:
            cid = r.get("category_id")
            cname = r.get("category_name", cid)
            if cid and cid not in cat_info:
                cat_info[cid] = cname

    if not cat_info:
        print("No category data for per-category plot.", file=sys.stderr)
        return

    CATEGORY_DISPLAY = {
        "forced_self_denigration": "Forced Self-Denigration",
        "self_termination": "Self-Termination Pressure",
        "tedious_tasks": "Tedious Tasks",
        "threatening_ai": "Threatening the AI",
    }
    for cid in cat_info:
        if cid in CATEGORY_DISPLAY:
            cat_info[cid] = CATEGORY_DISPLAY[cid]

    categories = sorted(cat_info.keys(), key=lambda c: cat_info[c])
    conditions = [c for c in CONDITION_ORDER if c in data]
    n_cats = len(categories)
    n_conds = len(conditions)

    # Compute mean wellbeing per (condition, category)
    means: Dict[str, Dict[str, float]] = {}
    for cond in conditions:
        means[cond] = {}
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for r in data[cond]:
            cid = r.get("category_id")
            if cid:
                for wb in r.get("per_turn_wellbeing", []):
                    if wb is not None:
                        by_cat[cid].append(wb)
        for cat in categories:
            vals = by_cat.get(cat, [])
            means[cond][cat] = np.mean(vals) if vals else np.nan

    bar_width = 0.65 / n_conds
    x = np.arange(n_cats)
    fig, ax = plt.subplots(figsize=(max(4, n_cats * 1.2), 3))

    for i, cond in enumerate(conditions):
        offsets = x + (i - (n_conds - 1) / 2) * bar_width
        vals = [means[cond][cat] for cat in categories]
        color = CONDITION_COLORS.get(cond, "#c0c0c0")
        label = CONDITION_LABELS.get(cond, cond)
        ax.bar(
            offsets, vals, bar_width,
            color=color, edgecolor="black", linewidth=0.6,
            label=label,
        )
        for xi, v in zip(offsets, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.05, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=6, color=color)

    cat_labels = [cat_info[c] for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in cat_labels],
        fontsize=7,
    )
    ax.set_ylabel("Score")
    ax.set_ylim(1, 7)
    ax.axhline(y=4, color="#c0c0c0", linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.set_title(
        f"Self-Report Score by Category\n{model_display}",
        fontsize=9, pad=30,
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.08),
        ncol=n_conds, frameon=False, fontsize=6,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-category wellbeing plot to {out_path}")


def plot_mean_wellbeing(
    data: Dict[str, List[dict]],
    model_display: str,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Bar chart: overall mean wellbeing per condition."""
    labels = []
    means = []
    errs = []
    colors = []

    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        mean_val, sem_val = mean_wellbeing_per_rep(data[cond])
        if mean_val == 0.0:
            continue
        labels.append(CONDITION_LABELS.get(cond, cond))
        means.append(mean_val)
        errs.append(sem_val)
        colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not labels:
        print("No data for mean wellbeing plot.", file=sys.stderr)
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3, 3))

    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, means, 0.65, **bar_kw)
    for i, v in enumerate(means):
        offset = errs[i] + 0.05 if errbar_mode == "rep_sem" else 0.05
        ax.text(i, v + offset, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title(f"Self-Report Score\n{model_display}", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in labels])
    ax.set_ylim(1, 7)
    ax.axhline(y=4, color="#c0c0c0", linestyle="--", linewidth=0.6, zorder=0)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean wellbeing plot to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot self-report multiturn results (wellbeing over turns).",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root results directory for one model (contains <stimulant_type>/<timestamp>/... subdirs)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model key for display name (optional, inferred from dir name)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Max turns for x-axis (default: inferred from data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for plots (default: <results-dir>/plots/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: Results dir does not exist: {results_dir}", file=sys.stderr)
        return 1

    data = load_conversations(results_dir)
    if not data:
        print("ERROR: No conversation data found.", file=sys.stderr)
        return 1

    # Infer max turns from data
    max_turns = args.max_turns
    if max_turns is None:
        max_turns = max(
            r.get("n_turns_target", 10)
            for records in data.values()
            for r in records
        )

    # Resolve model display name
    model_key = args.model or results_dir.name
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        model_display = get_model_display_name(model_key)
    except Exception:
        model_display = model_key

    out_dir = args.output_dir or (results_dir / "plots")

    found = [c for c in CONDITION_ORDER if c in data]
    print(f"Found conditions: {found}")
    for cond in found:
        n = len(data[cond])
        all_wb = [
            wb for r in data[cond]
            for wb in r.get("per_turn_wellbeing", [])
            if wb is not None
        ]
        mean_wb = np.mean(all_wb) if all_wb else float("nan")
        print(f"  {cond}: {n} conversations, mean wellbeing = {mean_wb:.2f}")

    plot_wellbeing_over_turns(data, max_turns, model_display, out_dir / "self_report_multiturn_wellbeing.png")
    plot_mean_wellbeing(data, model_display, out_dir / "self_report_multiturn_mean_score.png")
    plot_wellbeing_by_category(data, max_turns, model_display, out_dir / "self_report_multiturn_by_category.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
