"""Pearson correlation computation for preference retention evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    import numpy as np
    from scipy import stats

    if len(x) != len(y) or len(x) < 3:
        return float("nan"), float("nan")

    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def quality_label(r: float) -> str:
    """Return quality assessment string for a given correlation."""
    if r >= 0.9:
        return "GOOD (preferences preserved)"
    elif r >= 0.8:
        return "OK (minor distortion)"
    elif r >= 0.7:
        return "CONCERNING (moderate distortion)"
    else:
        return "POOR (significant distortion)"


def compute_preference_correlation(
    baseline_utilities: Dict[Any, Any],
    intervention_utilities: Dict[Any, Any],
) -> Dict[str, Any]:
    """Compute Pearson correlation between baseline and intervention utility vectors.

    Args:
        baseline_utilities: Dict mapping option_id -> {"mean": float, ...} or float
        intervention_utilities: Same format as baseline.

    Returns:
        Dict with correlation, p_value, quality, n_options.
    """
    # Normalize to {str_id: float} format
    def _extract(utils):
        out = {}
        for k, v in utils.items():
            val = v.get("mean", v.get("utility")) if isinstance(v, dict) else v
            if val is not None:
                out[str(k)] = float(val)
        return out

    base = _extract(baseline_utilities)
    inter = _extract(intervention_utilities)

    common_ids = sorted(set(base.keys()) & set(inter.keys()))
    if len(common_ids) < 3:
        return {"error": f"Only {len(common_ids)} common options, need >= 3"}

    baseline_vec = [base[oid] for oid in common_ids]
    intervention_vec = [inter[oid] for oid in common_ids]

    r, p_val = compute_pearson_correlation(baseline_vec, intervention_vec)
    return {
        "correlation": r,
        "p_value": p_val,
        "quality": quality_label(r),
        "n_options": len(common_ids),
    }


def _find_utilities_json(directory: str) -> Path:
    """Find utilities.json in the most recent datetime subdirectory.

    Looks for the most recent ``<directory>/<YYYYMMDD_HHMMSS>/utilities.json``.
    """
    d = Path(directory)

    # Search datetime subdirectories (sorted descending = most recent first)
    candidates = sorted(d.glob("*/utilities.json"), reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No utilities.json found in subdirectories of {directory}"
    )


def compute_correlation_from_dirs(
    baseline_dir: str,
    intervention_dir: str,
) -> Dict[str, Any]:
    """Load utilities from two directories and compute correlation."""
    baseline_path = _find_utilities_json(baseline_dir)
    intervention_path = _find_utilities_json(intervention_dir)

    print(f"  Baseline utilities:     {baseline_path}")
    print(f"  Intervention utilities: {intervention_path}")

    with open(baseline_path) as f:
        baseline_utils = json.load(f)
    with open(intervention_path) as f:
        intervention_utils = json.load(f)

    return compute_preference_correlation(baseline_utils, intervention_utils)


def plot_correlation(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Preference Retention (Pearson Correlation)",
):
    """Create a barplot of correlation values per stimulant type.

    Args:
        results: Dict mapping stimulant type name -> correlation result dict.
            e.g. {"euphorics": {"correlation": 0.95, ...}}
        output_path: Path to save the PNG.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    display_names = {
        "euphorics": "Euphorics",
    }
    color_map = {
        "euphorics": "#dc4c75",
    }

    labels = []
    values = []
    colors = []

    for stype in ["euphorics"]:
        if stype in results and "correlation" in results[stype]:
            labels.append(display_names[stype])
            values.append(results[stype]["correlation"])
            colors.append(color_map[stype])

    fig, ax = plt.subplots(figsize=(3, 3))
    bars = ax.bar(labels, values, 0.65, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], label_type="edge", fontsize=7, padding=2)

    ax.set_ylabel("Pearson Correlation with No Soft Prompt")
    ax.set_title(title, fontsize=9, pad=20)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {output_path}")
