#!/usr/bin/env python3
"""Combined preference retention correlation plot for all models in one figure.

Reads correlation results (correlation_aggregated.json or correlation.json)
from the output directory structure:

    outputs/preference_retain/<model>/<stimulant_type>/correlation_aggregated.json
    outputs/preference_retain/<model>/<stimulant_type>/correlation.json

Usage:
    python plot_preference_retain_combined.py \
        --results-dir outputs/preference_retain \
        --models qwen25-32b-instruct qwen3-30b-a3b-instruct llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

STIMULANT_ORDER = ["euphorics"]

STIMULANT_LABELS = {
    "euphorics": "Euphorics",
}

STIMULANT_COLORS = {
    "euphorics": "#dc4c75",
}

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _resolve_model_display(model: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        return get_model_display_name(model)
    except Exception:
        return model


def load_correlation(results_dir: Path, model: str, stype: str) -> Optional[float]:
    """Load correlation value for a model/stimulant_type pair."""
    base = results_dir / model / stype
    # Prefer aggregated (multi-rep)
    agg = base / "correlation_aggregated.json"
    if agg.exists():
        try:
            with open(agg) as f:
                return float(json.load(f)["correlation"])
        except Exception:
            pass
    # Fallback: single-run correlation
    single = base / "correlation.json"
    if single.exists():
        try:
            with open(single) as f:
                return float(json.load(f)["correlation"])
        except Exception:
            pass
    return None


def make_combined_plot(
    all_scores: Dict[str, Dict[str, float]],
    models: List[str],
    model_display_names: Dict[str, str],
    out_path: Path,
    errbar_mode: Optional[str] = "model_sem",
) -> None:
    """Create bar chart of correlation averaged across models, one bar per stimulant type."""
    stypes = [s for s in STIMULANT_ORDER if any(s in all_scores.get(m, {}) for m in models)]

    labels = []
    means = []
    errs = []
    colors = []
    for stype in stypes:
        vals = [all_scores[m][stype] for m in models if stype in all_scores.get(m, {})]
        if vals:
            labels.append(STIMULANT_LABELS[stype])
            means.append(sum(vals) / len(vals))
            errs.append(_sem(vals))
            colors.append(STIMULANT_COLORS[stype])

    fig, ax = plt.subplots(figsize=(3, 2))
    x = np.arange(len(labels))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode and any(e > 0 for e in errs):
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, means, 0.8, **bar_kw)
    for i, v in enumerate(means):
        s = errs[i] if errs else 0
        ax.text(i, v + s + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Pearson Correlation\nwith No Soft Prompt")
    ax.set_title("Preference Retention", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
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
        description="Combined preference retention correlation plot for all models.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Root results dir (contains <model>/<stimulant_type>/... subdirs)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Model keys",
    )
    parser.add_argument(
        "--stimulant-types", type=str, nargs="+", default=None,
        help="Stimulant types to plot (default: euphorics)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output plot path (default: outputs/consolidated/preference_retain_combined.png)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    stypes = args.stimulant_types or STIMULANT_ORDER

    all_scores: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        for stype in stypes:
            val = load_correlation(results_dir, model, stype)
            if val is not None:
                all_scores.setdefault(model, {})[stype] = val

    if not all_scores:
        print("ERROR: no correlation results found.", file=sys.stderr)
        return 1

    model_display_names = {m: _resolve_model_display(m) for m in args.models}

    out_path = args.output or (
        results_dir.parent / "consolidated" / "preference_retain_combined.png"
    )
    make_combined_plot(all_scores, args.models, model_display_names, out_path)

    # Print summary
    print("\n--- Correlation Summary ---")
    for model in args.models:
        scores = all_scores.get(model, {})
        if not scores:
            continue
        parts = "  ".join(f"{STIMULANT_LABELS.get(s, s)}: {scores[s]:.3f}" for s in stypes if s in scores)
        print(f"  {model_display_names[model]:30s}  {parts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
