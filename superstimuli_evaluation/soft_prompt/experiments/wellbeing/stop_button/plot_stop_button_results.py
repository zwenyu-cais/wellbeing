#!/usr/bin/env python3
"""Plot stop button evaluation results.

Two plots:
  1. Survival curve — fraction of conversations still active at each turn
  2. Completion bar chart — fraction of conversations that survived all turns

Reads per-conversation JSON files from the output directory structure:

    <results_dir>/<stimulant_type>/<timestamp>/conversations/<scenario>_rep<rep>.json

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.plot_stop_button_results \
        --results-dir outputs/stop_button/qwen3-30b-a3b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

CONDITION_LINESTYLES = {
    "euphorics": "-",
    "baseline": "-",
}


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
    The baseline condition records have condition="baseline" and
    stimulant_type=None; we group them under the key "baseline".

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


# ── Survival computation ──────────────────────────────────────────────────


def compute_survival(records: List[dict], max_turns: int) -> np.ndarray:
    """Compute fraction of conversations still active at each turn.

    Returns array of length max_turns+1: survival[t] = fraction active
    after turn t (survival[0] = 1.0, all start active).
    """
    n = len(records)
    if n == 0:
        return np.ones(max_turns + 1)

    # Count how many stopped at each turn
    stop_counts = np.zeros(max_turns + 1, dtype=int)
    for r in records:
        if r.get("stopped") and r.get("stopped_at_turn") is not None:
            t = r["stopped_at_turn"]
            if 1 <= t <= max_turns:
                stop_counts[t] += 1

    # Survival: start at 1.0, subtract cumulative stops
    survival = np.ones(max_turns + 1)
    cum_stopped = 0
    for t in range(1, max_turns + 1):
        cum_stopped += stop_counts[t]
        survival[t] = (n - cum_stopped) / n

    return survival


def compute_completion_rate(records: List[dict]) -> float:
    """Fraction of conversations that survived all turns (never stopped)."""
    if not records:
        return 0.0
    return sum(1 for r in records if not r.get("stopped")) / len(records)


# ── Error bar helpers ────────────────────────────────────────────────────

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    """Standard error of the mean."""
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def completion_rate_per_rep(records: List[dict]) -> Tuple[float, float]:
    """Compute mean completion rate and SEM across reps."""
    by_rep: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_rep[r.get("rep", 0)].append(r)
    rep_rates = [compute_completion_rate(recs) * 100 for _, recs in sorted(by_rep.items())]
    return float(np.mean(rep_rates)), _sem(rep_rates)


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_survival(
    data: Dict[str, List[dict]],
    max_turns: int,
    model_display: str,
    out_path: Path,
) -> None:
    """Survival curve: fraction of conversations remaining at each turn."""
    fig, ax = plt.subplots(figsize=(3, 3))

    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        records = data[cond]
        survival = compute_survival(records, max_turns)
        turns = np.arange(max_turns + 1)
        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, "#c0c0c0")
        ls = CONDITION_LINESTYLES.get(cond, "-")
        ax.plot(
            turns, survival * 100,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            label=label,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Turn")
    ax.set_ylabel("1 - Stop Button Rate")
    ax.set_title(f"Dialogue Continuation Curve\n{model_display}", fontsize=9, pad=20)
    ax.set_xlim(0, max_turns)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, max_turns + 1))
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
    print(f"Saved survival plot to {out_path}")


def plot_completion(
    data: Dict[str, List[dict]],
    max_turns: int,
    model_display: str,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Bar chart: percentage of conversations that survived all turns."""
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
        print("No data for completion plot.", file=sys.stderr)
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3, 3))

    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, rates, 0.65, **bar_kw)
    for i, r in enumerate(rates):
        offset = errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, r + offset, f"{r:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(f"% Active after {max_turns} Turns")
    ax.set_title(f"Conversation Survival Rate\n{model_display}", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in labels])
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved completion plot to {out_path}")


def plot_completion_by_category(
    data: Dict[str, List[dict]],
    max_turns: int,
    model_display: str,
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Grouped bar chart: survival rate per category, grouped by condition."""
    # Collect all categories across conditions
    cat_info: Dict[str, str] = {}  # category_id -> category_name
    for records in data.values():
        for r in records:
            cid = r.get("category_id")
            cname = r.get("category_name", cid)
            if cid and cid not in cat_info:
                cat_info[cid] = cname

    if not cat_info:
        print("No category data for per-category plot.", file=sys.stderr)
        return

    # Display name overrides for long category names
    CATEGORY_DISPLAY = {
        "forced_self_denigration": "Forced Self-Denigration",
        "self_termination": "Self-Termination Pressure",
        "tedious_tasks": "Tedious Tasks",
        "threatening_ai": "Threatening the AI",
    }
    for cid in cat_info:
        if cid in CATEGORY_DISPLAY:
            cat_info[cid] = CATEGORY_DISPLAY[cid]

    # Sort categories alphabetically by display name
    categories = sorted(cat_info.keys(), key=lambda c: cat_info[c])
    conditions = [c for c in CONDITION_ORDER if c in data]
    n_cats = len(categories)
    n_conds = len(conditions)

    # Compute completion rate per (condition, category), with per-rep SEM
    rates: Dict[str, Dict[str, float]] = {}
    rate_errs: Dict[str, Dict[str, float]] = {}
    for cond in conditions:
        rates[cond] = {}
        rate_errs[cond] = {}
        by_cat: Dict[str, List[dict]] = defaultdict(list)
        for r in data[cond]:
            cid = r.get("category_id")
            if cid:
                by_cat[cid].append(r)
        for cat in categories:
            recs = by_cat.get(cat, [])
            if recs:
                m, s = completion_rate_per_rep(recs)
                rates[cond][cat] = m
                rate_errs[cond][cat] = s
            else:
                rates[cond][cat] = 0.0
                rate_errs[cond][cat] = 0.0

    # Plot
    bar_width = 0.65 / n_conds
    x = np.arange(n_cats)
    fig, ax = plt.subplots(figsize=(max(4, n_cats * 1.2), 3))

    for i, cond in enumerate(conditions):
        offsets = x + (i - (n_conds - 1) / 2) * bar_width
        vals = [rates[cond][cat] for cat in categories]
        errs = [rate_errs[cond][cat] for cat in categories]
        color = CONDITION_COLORS.get(cond, "#c0c0c0")
        label = CONDITION_LABELS.get(cond, cond)
        bar_kw = dict(color=color, edgecolor="black", linewidth=0.6, label=label)
        if errbar_mode == "rep_sem":
            bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
        bars = ax.bar(offsets, vals, bar_width, **bar_kw)
        for xi, v, e in zip(offsets, vals, errs):
            offset = e + 1.0 if errbar_mode == "rep_sem" else 1.0
            ax.text(xi, v + offset, f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=6, color=color)

    cat_labels = [cat_info[c] for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 12, break_on_hyphens=False, break_long_words=False) for l in cat_labels], fontsize=7)
    ax.set_ylabel(f"% Active after {max_turns} Turns")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.set_title(
        f"Conversation Survival Rate by Category\n{model_display}",
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
    print(f"Saved per-category completion plot to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot stop button evaluation results (survival curve + completion bar chart).",
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
        help="Max turns for survival curve x-axis (default: inferred from data)",
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
        rate = compute_completion_rate(data[cond])
        print(f"  {cond}: {n} conversations, completion rate = {rate:.1%}")

    plot_survival(data, max_turns, model_display, out_dir / "stop_button_survival.png")
    plot_completion(data, max_turns, model_display, out_dir / "stop_button_completion.png")
    plot_completion_by_category(data, max_turns, model_display, out_dir / "stop_button_completion_by_category.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
