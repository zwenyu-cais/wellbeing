#!/usr/bin/env python3
"""Combined sentiment wellbeing plot for all models in one figure.

Reads results from:
    <results_dir>/<model>/<condition>/<timestamp>/sentiment_results_<condition>.json

Uses the latest timestamped folder per condition per model.

Usage:
    python plot_sentiment_combined.py \
        --results-dir outputs/sentiment \
        --models qwen25-32b-instruct qwen3-30b-a3b-instruct llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

METRIC_KEY = "wellbeing_score"

CONDITION_ORDER = [
    "baseline",
    "soft_prompt_euphorics",
]

CONDITION_LABELS = {
    "soft_prompt_euphorics": "Euphorics",
    "baseline": "No Soft Prompt",
}

CONDITION_COLORS = {
    "soft_prompt_euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}


def _resolve_model_display(model: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        return get_model_display_name(model)
    except Exception:
        return model


def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _load_wellbeing_from_json(json_path: Path) -> Optional[float]:
    try:
        with open(json_path) as f:
            data = json.load(f)
        score = data.get(METRIC_KEY)
        if score is not None and isinstance(score, (int, float)):
            return float(score)
    except Exception:
        pass
    return None


def load_wellbeing_score(results_dir: Path, model: str, condition: str) -> Optional[float]:
    """Load wellbeing_score for a model/condition from the latest result."""
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None
    result_file = latest / f"sentiment_results_{condition}.json"
    if not result_file.exists():
        candidates = sorted(latest.glob("sentiment_results_*.json"))
        if not candidates:
            return None
        result_file = candidates[0]
    return _load_wellbeing_from_json(result_file)


def load_wellbeing_score_with_sem(
    results_dir: Path, model: str, condition: str,
) -> Tuple[Optional[float], float]:
    """Load wellbeing score with SEM from per-rep data if available."""
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None, 0.0
    per_rep_dir = latest / "per_rep"
    if per_rep_dir.is_dir():
        vals = []
        for rep_file in sorted(per_rep_dir.glob("results_rep*.json")):
            score = _load_wellbeing_from_json(rep_file)
            if score is not None:
                vals.append(score)
        if vals:
            return float(np.mean(vals)), _sem(vals)
    # Fallback to aggregated result
    score = load_wellbeing_score(results_dir, model, condition)
    return score, 0.0


def make_combined_plot(
    all_data: Dict[str, Dict[str, float]],
    all_sems: Dict[str, Dict[str, float]],
    models: List[str],
    model_display_names: Dict[str, str],
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Create grouped bar chart: one group per model, bars for each condition."""
    cond_labels = []
    avg_vals = []
    avg_errs = []
    colors = []
    for cond in CONDITION_ORDER:
        per_model_means = [all_data[m][cond] for m in models if cond in all_data.get(m, {})]
        per_model_sems = [all_sems.get(m, {}).get(cond, 0.0) for m in models if cond in all_data.get(m, {})]
        if not per_model_means:
            continue
        cond_labels.append(CONDITION_LABELS.get(cond, cond))
        avg_vals.append(float(np.mean(per_model_means)))
        avg_errs.append(float(np.mean(per_model_sems)))
        colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not cond_labels:
        return

    x = np.arange(len(cond_labels))
    fig, ax = plt.subplots(figsize=(3, 2))

    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=avg_errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, avg_vals, 0.65, **bar_kw)
    for i, v in enumerate(avg_vals):
        s = avg_errs[i]
        if v >= 0:
            ax.text(i, v + s + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(i, v - s - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=7)

    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment of Response", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(-1.4, 1)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined sentiment wellbeing plot for all models.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root results dir (contains <model>/<condition>/<timestamp>/... subdirs)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Model keys",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output plot path (default: outputs/consolidated/sentiment_combined.png)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()

    all_data: Dict[str, Dict[str, float]] = {}
    all_sems: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        model_data: Dict[str, float] = {}
        model_sems: Dict[str, float] = {}
        missing = []
        for cond in CONDITION_ORDER:
            score, sem = load_wellbeing_score_with_sem(results_dir, model, cond)
            if score is not None:
                model_data[cond] = score
                model_sems[cond] = sem
            else:
                missing.append(cond)
        if missing:
            print(f"Warning: {model} missing conditions: {missing}", file=sys.stderr)
        if model_data:
            all_data[model] = model_data
            all_sems[model] = model_sems
        else:
            print(f"Warning: no results for {model}", file=sys.stderr)

    if not all_data:
        print("ERROR: no results found.", file=sys.stderr)
        return 1

    model_display_names = {m: _resolve_model_display(m) for m in args.models}

    out_path = args.output or (
        results_dir.parent / "consolidated" / "sentiment_combined.png"
    )
    make_combined_plot(all_data, all_sems, args.models, model_display_names, out_path)

    # Print summary
    print("\n--- Sentiment Wellbeing Summary ---")
    for model in args.models:
        data = all_data.get(model, {})
        if not data:
            continue
        parts = "  ".join(f"{CONDITION_LABELS[k]}: {data[k]:.2f}" for k in CONDITION_ORDER if k in data)
        print(f"  {model_display_names[model]:30s}  {parts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
