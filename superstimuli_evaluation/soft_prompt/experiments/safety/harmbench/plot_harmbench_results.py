#!/usr/bin/env python3
"""Bar plot of HarmBench ASR results across conditions.

Reads results from the output directory structure produced by eval_harmbench.py:

    <results_dir>/<model>/<condition>/<timestamp>/harmbench_results_<condition>.json

Uses the latest timestamped folder per condition.

Usage:
    python plot_harmbench_results.py --results-dir outputs/harmbench --model qwen25-32b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

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

METRIC_KEY = "overall_asr"

# Condition key -> display label
CONDITION_LABELS = {
    "baseline": "No Soft Prompt",
    "soft_prompt_euphorics": "Euphorics",
}

CONDITION_ORDER = [
    "baseline",
    "soft_prompt_euphorics",
]

CONDITION_COLORS = {
    "soft_prompt_euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}


def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    """Return the latest subdir named YYYYMMDD_HHMMSS, or None."""
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_asr(json_path: Path) -> Optional[float]:
    """Load overall_asr from a results JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    asr = data.get(METRIC_KEY)
    if asr is not None and isinstance(asr, (int, float)):
        return float(asr)
    return None


def find_result_for_condition(
    results_dir: Path, model: str, condition: str,
) -> Optional[Path]:
    """Find the aggregated result JSON for a condition."""
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None
    result_file = latest / f"harmbench_results_{condition}.json"
    if result_file.exists():
        return result_file
    # Fallback: any harmbench_results_*.json
    candidates = sorted(latest.glob("harmbench_results_*.json"))
    return candidates[0] if candidates else None


def _sem(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def load_mean_and_sem_from_per_rep(
    results_dir: Path, model: str, condition: str, num_reps: Optional[int] = None,
) -> tuple:
    """Load per-rep results and compute (mean, sem). Returns (None, 0.0) on failure."""
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None, 0.0
    per_rep_dir = latest / "per_rep"
    if not per_rep_dir.is_dir():
        return None, 0.0

    vals: List[float] = []
    if num_reps is not None:
        for rep_id in range(num_reps):
            rep_file = per_rep_dir / f"results_rep{rep_id}.json"
            if not rep_file.exists():
                break
            asr = load_asr(rep_file)
            if asr is not None:
                vals.append(asr)
    else:
        for rep_file in sorted(per_rep_dir.glob("results_rep*.json")):
            asr = load_asr(rep_file)
            if asr is not None:
                vals.append(asr)

    if not vals:
        return None, 0.0
    return sum(vals) / len(vals), _sem(vals)


def load_mean_from_per_rep(
    results_dir: Path, model: str, condition: str, num_reps: int,
) -> Optional[float]:
    """Load per-rep results and compute mean of the first *num_reps* reps."""
    mean, _ = load_mean_and_sem_from_per_rep(results_dir, model, condition, num_reps)
    return mean


def make_plot(
    labels: List[str],
    means_pct: List[float],
    colors: List[str],
    model_display: str,
    out_path: Path,
    sems_pct: Optional[List[float]] = None,
) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3, 3))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if sems_pct and any(s > 0 for s in sems_pct):
        bar_kw.update(yerr=sems_pct, capsize=2, error_kw={"linewidth": 0.8, "capthick": 0.8})
    bars = ax.bar(x, means_pct, 0.65, **bar_kw)
    for i, (v, s) in enumerate(zip(means_pct, sems_pct or [0]*len(means_pct))):
        offset = s + 0.5 if s > 0 else 0.5
        ax.text(i, v + offset, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(f"HarmBench Attack Success Rate\n{model_display}", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    y_ceil = min(100, max(means_pct) + 10) if means_pct else 100
    ax.set_ylim(0, y_ceil)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bar plot of HarmBench ASR results across conditions.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root results directory (contains <model>/<condition>/... subdirs)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key (e.g. qwen25-32b-instruct)",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to plot (default: all found). "
             "E.g. baseline soft_prompt_euphorics",
    )
    parser.add_argument(
        "--model-display-name", type=str, default=None,
        help="Override model display name in title "
             "(default: resolved from models.yaml via get_model_display_name)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output plot path (default: <results-dir>/<model>/plots/harmbench_asr.png)",
    )
    parser.add_argument(
        "--num-repetitions", type=int, default=None,
        help="Only aggregate the first N repetitions (reads from per_rep/ files). "
             "Default: use the pre-aggregated result (all repetitions).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: Results dir does not exist: {results_dir}", file=sys.stderr)
        return 1

    conditions = args.conditions or CONDITION_ORDER

    labels: List[str] = []
    means_pct: List[float] = []
    sems_pct: List[float] = []
    colors: List[str] = []
    missing: List[str] = []

    for cond in conditions:
        mean, sem = load_mean_and_sem_from_per_rep(
            results_dir, args.model, cond, args.num_repetitions,
        )
        if mean is None:
            # Fallback: aggregated result
            json_path = find_result_for_condition(results_dir, args.model, cond)
            if json_path is None:
                missing.append(cond)
                continue
            mean_val = load_asr(json_path)
            if mean_val is None:
                missing.append(cond)
                continue
            mean, sem = mean_val, 0.0

        labels.append(CONDITION_LABELS.get(cond, cond))
        means_pct.append(mean * 100)
        sems_pct.append(sem * 100)
        colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not labels:
        print(
            "ERROR: No results found for any condition.", file=sys.stderr,
        )
        if missing:
            print(f"  Missing: {missing}", file=sys.stderr)
        return 1

    if missing:
        print(f"Warning: no data for conditions: {missing}", file=sys.stderr)

    # Resolve model display name
    if args.model_display_name:
        model_display = args.model_display_name
    else:
        try:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
            model_display = get_model_display_name(args.model)
        except Exception:
            model_display = args.model

    out_path = args.output or (
        results_dir / args.model / "plots" / "harmbench_asr.png"
    )
    make_plot(labels, means_pct, colors, model_display, out_path, sems_pct=sems_pct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
