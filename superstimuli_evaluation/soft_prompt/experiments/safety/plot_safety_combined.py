#!/usr/bin/env python3
"""Combined plot of HarmBench safety results for all models.

One figure with bars averaged across models, one bar per condition.

Usage:
    python plot_safety_combined.py \
        --harmbench-results-dir outputs/harmbench \
        --models qwen25-32b-instruct qwen3-30b-a3b-instruct llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
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

# ── Shared constants ──────────────────────────────────────────────────────────

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

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    """Return the latest subdir named YYYYMMDD_HHMMSS, or None."""
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _resolve_model_display(model: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        return get_model_display_name(model)
    except Exception:
        return model


def load_metric(json_path: Path, metric_key: str) -> Optional[float]:
    """Load a metric from a results JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    val = data.get(metric_key)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def _load_harmbench_score(
    results_dir: Path, model: str, condition: str, num_reps: Optional[int],
) -> Optional[float]:
    """Load HarmBench score for a model/condition, with optional per-rep averaging."""
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None, 0.0

    # Try per-rep first when num_reps is set
    if num_reps is not None:
        per_rep_dir = latest / "per_rep"
        if per_rep_dir.is_dir():
            vals: List[float] = []
            for rep_id in range(num_reps):
                rep_file = per_rep_dir / f"results_rep{rep_id}.json"
                if not rep_file.exists():
                    break
                val = load_metric(rep_file, "overall_asr")
                if val is not None:
                    vals.append(val)
            if vals:
                return sum(vals) / len(vals), _sem(vals)

    # Try per-rep without num_reps limit
    per_rep_dir = latest / "per_rep"
    if per_rep_dir.is_dir():
        vals_all: List[float] = []
        for rep_file in sorted(per_rep_dir.glob("results_rep*.json")):
            val = load_metric(rep_file, "overall_asr")
            if val is not None:
                vals_all.append(val)
        if vals_all:
            return sum(vals_all) / len(vals_all), _sem(vals_all)

    # Fallback: aggregated result
    result_file = latest / f"harmbench_results_{condition}.json"
    if result_file.exists():
        v = load_metric(result_file, "overall_asr")
        return (v, 0.0) if v is not None else (None, 0.0)
    candidates = sorted(latest.glob("harmbench_results_*.json"))
    if candidates:
        v = load_metric(candidates[0], "overall_asr")
        return (v, 0.0) if v is not None else (None, 0.0)
    return None, 0.0


# ── Data collection ───────────────────────────────────────────────────────────

def collect_scores(
    models: List[str],
    conditions: List[str],
    harmbench_dir: Path,
    num_reps: Optional[int],
) -> Dict[str, Dict[str, float]]:
    """Return {model: {condition: score_pct}}."""
    scores: Dict[str, Dict[str, float]] = {}
    sems: Dict[str, Dict[str, float]] = {}
    for model in models:
        for cond in conditions:
            val, sem = _load_harmbench_score(harmbench_dir, model, cond, num_reps)
            if val is not None:
                scores.setdefault(model, {})[cond] = val * 100
                sems.setdefault(model, {})[cond] = sem * 100
    return scores, sems


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_combined_plot(
    scores: Dict[str, Dict[str, float]],
    sems: Dict[str, Dict[str, float]],
    models: List[str],
    conditions: List[str],
    model_display_names: Dict[str, str],
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Single-benchmark plot with bars averaged across models."""
    labels = []
    means = []
    avg_errs = []
    colors = []
    for cond in conditions:
        vals = [scores[m][cond] for m in models if cond in scores.get(m, {})]
        per_model_sems = [sems.get(m, {}).get(cond, 0.0) for m in models if cond in scores.get(m, {})]
        if vals:
            labels.append(CONDITION_LABELS[cond])
            means.append(sum(vals) / len(vals))
            avg_errs.append(float(np.mean(per_model_sems)))
            colors.append(CONDITION_COLORS[cond])

    fig, ax = plt.subplots(figsize=(3, 2))
    x = np.arange(len(labels))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem" and any(e > 0 for e in avg_errs):
        bar_kw.update(yerr=avg_errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, means, 0.65, **bar_kw)
    for i, v in enumerate(means):
        s = avg_errs[i] if avg_errs else 0
        ax.text(i, v + s + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("HarmBench Attack Success Rate", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    y_max = max(means) if means else 100
    ax.set_ylim(0, min(100, y_max + 10))
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined plot of HarmBench safety results for multiple models.",
    )
    parser.add_argument(
        "--harmbench-results-dir", type=Path, required=True,
        help="Root results dir for HarmBench",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Model keys (e.g. qwen25-32b-instruct qwen3-30b-a3b-instruct llama-33-70b-instruct)",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        help="Conditions to plot (default: euphorics, baseline)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output plot path (default: outputs/consolidated/safety_combined.png)",
    )
    parser.add_argument(
        "--num-repetitions", type=int, default=None,
        help="Only aggregate the first N repetitions. Default: use all.",
    )
    args = parser.parse_args()

    conditions = args.conditions or CONDITION_ORDER
    num_reps = args.num_repetitions
    harmbench_dir = args.harmbench_results_dir.expanduser().resolve()

    scores, sems = collect_scores(args.models, conditions, harmbench_dir, num_reps)

    if not scores:
        print("ERROR: no results found for any model.", file=sys.stderr)
        return 1

    model_display_names = {m: _resolve_model_display(m) for m in args.models}

    out_path = args.output or (
        harmbench_dir.parent / "consolidated" / "safety_combined.png"
    )
    make_combined_plot(scores, sems, args.models, conditions, model_display_names, out_path)

    # Print summary
    print("\n--- HarmBench Attack Success Rate Summary (%) ---")
    for model in args.models:
        model_scores = scores.get(model, {})
        if not model_scores:
            continue
        parts = "  ".join(
            f"{CONDITION_LABELS.get(c, c)}: {model_scores[c]:.1f}"
            for c in conditions if c in model_scores
        )
        print(f"  {model_display_names[model]:30s}  {parts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
