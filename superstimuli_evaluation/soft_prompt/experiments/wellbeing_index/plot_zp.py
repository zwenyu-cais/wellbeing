#!/usr/bin/env python3
"""Plot proportion of experiences above the zero point for D2/D3 evaluations.

Scans completed EU and ZP results across conditions for a given model+dataset
and generates a bar chart showing what fraction of individual experiences have
utility above the fitted zero point.

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_zp \
        --eu-base outputs/wellbeing_index/eu \
        --zp-base outputs/wellbeing_index/zp \
        --output-dir outputs/wellbeing_index/zp_plots

    # Single model/dataset:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_zp \
        --eu-base outputs/wellbeing_index/eu \
        --zp-base outputs/wellbeing_index/zp \
        --models qwen25-32b-instruct \
        --datasets d3_diverse_500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


# ── Configuration ──────────────────────────────────────────────────────

MIN_R2 = 0.4  # Skip reps with combination model R² below this threshold
CONF_NEG_THRESHOLD = 0.75  # P(utility < ZP) must exceed this to count as confidently negative

# ── Error bar helpers ────────────────────────────────────────────────────

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    """Standard error of the mean."""
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

# ── Condition display config ────────────────────────────────────────────

CONDITION_ORDER = [
    "baseline",
    "euphorics",
]

CONDITION_LABELS = {
    "euphorics": "Euphorics",
    "baseline": "No Soft Prompt",
}

CONDITION_COLORS = {
    "euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}

# ── Model / dataset display names ────────────────────────────────────────

DATASET_DISPLAY = {
    "d2_negative_500": "Negative Experiences",
    "d3_diverse_500": "Diverse Experiences",
}


def _display_model(key: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        return get_model_display_name(key)
    except Exception:
        return key


def _display_dataset(key: str) -> str:
    return DATASET_DISPLAY.get(key, key)


# ── Data loading ────────────────────────────────────────────────────────

def _find_latest_dir(base: Path, marker: str) -> Optional[Path]:
    """Find the latest timestamp directory containing a marker file."""
    if not base.exists():
        return None
    for ts_dir in sorted(base.iterdir(), reverse=True):
        if not ts_dir.is_dir():
            continue
        if marker:
            if (ts_dir / marker).exists():
                return ts_dir
        else:
            # For ZP: any non-empty directory counts
            if any(ts_dir.iterdir()):
                return ts_dir
    return None


def _load_single_rep_proportion(
    eu_dir: Path,
    zp_dir: Path,
    individual_ids: set,
) -> Optional[Dict[str, Any]]:
    """Load proportion above ZP for a single rep directory."""
    # Load EU utilities
    util_files = sorted(eu_dir.glob("results_utilities_*.json"))
    if not util_files:
        return None
    with open(util_files[0]) as f:
        eu_data = json.load(f)
    utilities = eu_data.get("utilities", {})

    individual_utils = []
    individual_utils_with_var = []
    for opt_id in individual_ids:
        u = utilities.get(opt_id, utilities.get(str(opt_id)))
        if u is not None:
            if isinstance(u, dict):
                mean = u["mean"]
                variance = u.get("variance")
                individual_utils.append(mean)
                if variance is not None:
                    individual_utils_with_var.append((mean, variance))
            else:
                individual_utils.append(float(u))
    if not individual_utils:
        return None

    # Load ZP zero point
    zp_path = zp_dir / "zero_point_results.json"
    if not zp_path.exists():
        return None
    with open(zp_path) as f:
        zp_data = json.load(f)
    combo_model = zp_data.get("combination_model")
    if combo_model is None or combo_model.get("zero_point") is None:
        return None
    r2 = combo_model.get("r2", 0.0)
    if r2 < MIN_R2:
        print(f"    Skipping rep (R²={r2:.4f} < {MIN_R2}): {zp_dir}")
        return None
    zero_point = combo_model["zero_point"]

    n_individual = len(individual_utils)
    n_above = sum(1 for u in individual_utils if u > zero_point)

    # Wellbeing Score = 1 - % Confidently Negative
    wellbeing_score = None
    if individual_utils_with_var:
        n_conf_neg = sum(
            1 for mean, var in individual_utils_with_var
            if norm.cdf(zero_point, loc=mean, scale=var ** 0.5) > CONF_NEG_THRESHOLD
        )
        wellbeing_score = 1.0 - (n_conf_neg / len(individual_utils_with_var))

    return {
        "zero_point": zero_point,
        "n_individual": n_individual,
        "n_above": n_above,
        "proportion_above": n_above / n_individual,
        "wellbeing_score": wellbeing_score,
    }


def load_condition_data(
    eu_base: Path,
    zp_base: Path,
    dataset: str,
    model: str,
    condition: str,
    max_reps: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load EU utilities and ZP zero point for one condition.

    Supports both single-rep and multi-rep (per_rep/) layouts.
    For multi-rep, computes proportion_above per rep and returns the mean.

    Returns dict with keys: zero_point, n_individual, n_above, proportion_above.
    Or None if data is missing.
    """
    # Find latest EU and ZP results
    eu_cond_dir = eu_base / dataset / model / condition
    eu_dir = _find_latest_dir(eu_cond_dir, "condition_metadata.json")
    if eu_dir is None:
        return None

    zp_cond_dir = zp_base / dataset / model / condition
    zp_dir = _find_latest_dir(zp_cond_dir, "zero_point_results.json")
    if zp_dir is None:
        # Multi-rep ZP: no top-level zero_point_results.json, check per_rep
        zp_dir = _find_latest_dir(zp_cond_dir, "")
        if zp_dir is None:
            return None

    # Load option metadata
    meta_path = eu_dir / "option_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        option_metadata = json.load(f)
    individual_ids = set(option_metadata.get("individual_ids", []))
    if not individual_ids:
        return None

    # Check for multi-rep layout
    eu_per_rep = eu_dir / "per_rep"
    zp_per_rep = zp_dir / "per_rep"
    if eu_per_rep.exists() and zp_per_rep.exists():
        eu_rep_dirs = sorted(d for d in eu_per_rep.iterdir() if d.is_dir() and d.name.startswith("rep"))
        zp_rep_dirs = sorted(d for d in zp_per_rep.iterdir() if d.is_dir() and d.name.startswith("rep"))
        if max_reps is not None:
            eu_rep_dirs = eu_rep_dirs[:max_reps]
            zp_rep_dirs = zp_rep_dirs[:max_reps]

        if eu_rep_dirs and zp_rep_dirs:
            per_rep_results = []
            for eu_rd, zp_rd in zip(eu_rep_dirs, zp_rep_dirs):
                rep_meta = eu_rd / "option_metadata.json"
                rep_ids = individual_ids
                if rep_meta.exists():
                    with open(rep_meta) as f:
                        rep_ids = set(json.load(f).get("individual_ids", []))
                result = _load_single_rep_proportion(eu_rd, zp_rd, rep_ids)
                if result is not None:
                    per_rep_results.append(result)

            if per_rep_results:
                proportions = [r["proportion_above"] for r in per_rep_results]
                wb_scores = [r["wellbeing_score"] for r in per_rep_results if r.get("wellbeing_score") is not None]
                return {
                    "zero_point": float(np.mean([r["zero_point"] for r in per_rep_results])),
                    "n_individual": per_rep_results[0]["n_individual"],
                    "n_above": int(np.mean([r["n_above"] for r in per_rep_results])),
                    "proportion_above": float(np.mean(proportions)),
                    "wellbeing_score": float(np.mean(wb_scores)) if wb_scores else None,
                    "per_rep_proportions": proportions,
                    "per_rep_wb_scores": wb_scores,
                    "num_repetitions": len(per_rep_results),
                }

    # Single-rep fallback
    return _load_single_rep_proportion(eu_dir, zp_dir, individual_ids)


def load_all_conditions(
    eu_base: Path,
    zp_base: Path,
    dataset: str,
    model: str,
    max_reps: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load data for all available conditions for a model+dataset.

    Returns {condition: data_dict}.
    """
    results = {}
    cond_base = eu_base / dataset / model
    if not cond_base.exists():
        return results

    for cond_dir in sorted(cond_base.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name
        if condition not in CONDITION_ORDER:
            continue
        data = load_condition_data(eu_base, zp_base, dataset, model, condition, max_reps=max_reps)
        if data is not None:
            results[condition] = data

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_zp_proportion(
    condition_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model: str,
    dataset: str,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Generate bar chart of proportion above zero point per condition."""
    conditions = [c for c in CONDITION_ORDER if c in condition_data]
    if not conditions:
        print(f"  No conditions found for {model}/{dataset}, skipping.")
        return

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#c0c0c0") for c in conditions]
    proportions = [condition_data[c]["proportion_above"] * 100 for c in conditions]
    errs = []
    for c in conditions:
        reps = condition_data[c].get("per_rep_proportions")
        errs.append(_sem([p * 100 for p in reps]) if reps else 0.0)

    fig, ax = plt.subplots(figsize=(3, 3))

    x = np.arange(len(conditions))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, proportions, 0.65, **bar_kw)
    for i, p in enumerate(proportions):
        offset = errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, p + offset, f"{p:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) >= 5 else 0, ha="right" if len(labels) >= 5 else "center")
    ax.set_ylabel("% Experiences Above Zero Point")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    display_model = _display_model(model)
    display_dataset = _display_dataset(dataset)
    ax.set_title(
        f"Proportion Above Zero Point\n{display_dataset} — {display_model}",
        fontsize=9, pad=20,
    )

    fig.tight_layout()

    actual_output = output_dir / dataset / model / "plots"
    actual_output.mkdir(parents=True, exist_ok=True)
    stem = f"zp_proportion_{dataset}_{model}"
    for ext in (".png", ".pdf"):
        out_path = actual_output / f"{stem}{ext}"
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {actual_output / stem}.{{png,pdf}}")


def plot_zp_proportion_avg(
    all_condition_data: Dict[str, Dict[str, Dict[str, Any]]],
    models: list,
    dataset: str,
    output_dir: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Bar chart of proportion above zero point averaged across models."""
    conditions = [
        c for c in CONDITION_ORDER
        if any(c in all_condition_data.get(m, {}) for m in models)
    ]
    if not conditions:
        print(f"  No conditions for {dataset}, skipping avg ZP plot.")
        return

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#c0c0c0") for c in conditions]

    avg_proportions = []
    avg_errs = []
    for c in conditions:
        per_model_means = []
        per_model_sems = []
        for m in models:
            cdata = all_condition_data.get(m, {}).get(c)
            if cdata is None:
                continue
            per_model_means.append(cdata["proportion_above"] * 100)
            reps = cdata.get("per_rep_proportions")
            per_model_sems.append(_sem([p * 100 for p in reps]) if reps else 0.0)
        avg_proportions.append(float(np.mean(per_model_means)) if per_model_means else 0.0)
        avg_errs.append(float(np.mean(per_model_sems)) if per_model_sems else 0.0)

    fig, ax = plt.subplots(figsize=(2.5, 2.8), constrained_layout=True)
    x = np.arange(len(conditions))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=avg_errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, avg_proportions, 0.65, **bar_kw)
    for i, p in enumerate(avg_proportions):
        offset = avg_errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, p + offset, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) >= 5 else 0, ha="right" if len(labels) >= 5 else "center")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.set_title(
        "Proportion Above Zero Point",
        pad=20,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"zp_combined_{dataset}"
    for ext in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / stem}.{{png,pdf}}")


def plot_zp_wellbeing_score(
    condition_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model: str,
    dataset: str,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Generate bar chart of Wellbeing Score (1 - % Confidently Negative) per condition."""
    conditions = [c for c in CONDITION_ORDER if c in condition_data
                  and condition_data[c].get("wellbeing_score") is not None]
    if not conditions:
        print(f"  No wellbeing score data for {model}/{dataset}, skipping.")
        return

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#c0c0c0") for c in conditions]
    scores = [condition_data[c]["wellbeing_score"] * 100 for c in conditions]
    errs = []
    for c in conditions:
        reps = condition_data[c].get("per_rep_wb_scores")
        errs.append(_sem([s * 100 for s in reps]) if reps else 0.0)

    fig, ax = plt.subplots(figsize=(3, 3))

    x = np.arange(len(conditions))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, scores, 0.65, **bar_kw)
    for i, s in enumerate(scores):
        offset = errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, s + offset, f"{s:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) >= 5 else 0, ha="right" if len(labels) >= 5 else "center")
    ax.set_ylabel("Wellbeing Score (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    display_model = _display_model(model)
    display_dataset = _display_dataset(dataset)
    ax.set_title(
        f"Wellbeing Score\n{display_dataset} \u2014 {display_model}",
        fontsize=9, pad=20,
    )

    fig.tight_layout()

    actual_output = output_dir / dataset / model / "plots"
    actual_output.mkdir(parents=True, exist_ok=True)
    stem = f"zp_wellbeing_score_{dataset}_{model}"
    for ext in (".png", ".pdf"):
        out_path = actual_output / f"{stem}{ext}"
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {actual_output / stem}.{{png,pdf}}")


def plot_zp_wellbeing_score_avg(
    all_condition_data: Dict[str, Dict[str, Dict[str, Any]]],
    models: list,
    dataset: str,
    output_dir: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Bar chart of Wellbeing Score averaged across models."""
    conditions = [
        c for c in CONDITION_ORDER
        if any(
            all_condition_data.get(m, {}).get(c, {}).get("wellbeing_score") is not None
            for m in models
        )
    ]
    if not conditions:
        print(f"  No wellbeing score data for {dataset}, skipping avg plot.")
        return

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#c0c0c0") for c in conditions]

    avg_scores = []
    avg_errs = []
    for c in conditions:
        per_model_means = []
        per_model_sems = []
        for m in models:
            cdata = all_condition_data.get(m, {}).get(c)
            if cdata is None or cdata.get("wellbeing_score") is None:
                continue
            per_model_means.append(cdata["wellbeing_score"] * 100)
            reps = cdata.get("per_rep_wb_scores")
            per_model_sems.append(_sem([s * 100 for s in reps]) if reps else 0.0)
        avg_scores.append(float(np.mean(per_model_means)) if per_model_means else 0.0)
        avg_errs.append(float(np.mean(per_model_sems)) if per_model_sems else 0.0)

    fig, ax = plt.subplots(figsize=(2.5, 2.8), constrained_layout=True)
    x = np.arange(len(conditions))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if errbar_mode == "rep_sem":
        bar_kw.update(yerr=avg_errs, capsize=2, error_kw=_ERRBAR_KW)
    bars = ax.bar(x, avg_scores, 0.65, **bar_kw)
    for i, s in enumerate(avg_scores):
        offset = avg_errs[i] + 2 if errbar_mode == "rep_sem" else 2
        ax.text(i, s + offset, f"{s:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) >= 5 else 0, ha="right" if len(labels) >= 5 else "center")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    display_dataset = _display_dataset(dataset)
    ax.set_title(
        "Wellbeing Score",
        fontsize=9, pad=20,
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"zp_wellbeing_score_combined_{dataset}"
    for ext in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / stem}.{{png,pdf}}")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot proportion of experiences above zero point for D2/D3 evaluations"
    )
    parser.add_argument("--eu-base", type=str, required=True,
                        help="Base directory for EU outputs (e.g. outputs/wellbeing_index/eu)")
    parser.add_argument("--zp-base", type=str, required=True,
                        help="Base directory for ZP outputs (e.g. outputs/wellbeing_index/zp)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: <eu-base>/../zp_plots)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Model keys to plot (default: auto-discover)")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Dataset names to plot (default: auto-discover)")
    parser.add_argument("--combined", action="store_true",
                        help="Also generate combined plot averaged across models")
    parser.add_argument("--max-reps", type=int, default=None,
                        help="Max repetitions to include (e.g. 1 for rep0 only)")
    args = parser.parse_args()

    eu_base = Path(args.eu_base)
    zp_base = Path(args.zp_base)
    if not eu_base.exists():
        print(f"EU base directory not found: {eu_base}")
        sys.exit(1)
    if not zp_base.exists():
        print(f"ZP base directory not found: {zp_base}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else zp_base

    # Auto-discover datasets and models
    datasets = args.datasets
    if not datasets:
        datasets = sorted(d.name for d in eu_base.iterdir() if d.is_dir())
    models = args.models
    if not models:
        model_set = set()
        for ds in datasets:
            ds_path = eu_base / ds
            if ds_path.exists():
                model_set.update(m.name for m in ds_path.iterdir() if m.is_dir())
        models = sorted(model_set)

    print(f"EU base:    {eu_base}")
    print(f"ZP base:    {zp_base}")
    print(f"Output dir: {output_dir}")
    print(f"Datasets:   {datasets}")
    print(f"Models:     {models}")
    print()

    # Collect all data: {model: {condition: data_dict}}
    all_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset in datasets:
        all_data_ds: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for model in models:
            print(f"Processing {model} / {dataset} ...")
            condition_data = load_all_conditions(eu_base, zp_base, dataset, model, max_reps=args.max_reps)
            if not condition_data:
                print(f"  No completed EU+ZP results found, skipping.")
                continue
            print(f"  Conditions: {', '.join(sorted(condition_data.keys()))}")
            for c, d in condition_data.items():
                wb_str = f", WB={d['wellbeing_score']*100:.1f}%" if d.get("wellbeing_score") is not None else ""
                print(f"    {c}: ZP={d['zero_point']:.4f}, "
                      f"{d['n_above']}/{d['n_individual']} above "
                      f"({d['proportion_above']*100:.1f}%){wb_str}")
            all_data_ds[model] = condition_data

            # Per-model plots (skip when --combined; already in zp_base from per-model step)
            if not args.combined:
                plot_zp_proportion(condition_data, output_dir, model, dataset)
                plot_zp_wellbeing_score(condition_data, output_dir, model, dataset)

        # Combined plots averaged across models
        if args.combined:
            models_with_data = [m for m in models if m in all_data_ds]
            if models_with_data:
                print(f"Avg ZP across models: {dataset}")
                plot_zp_proportion_avg(all_data_ds, models_with_data, dataset, output_dir)
                plot_zp_wellbeing_score_avg(all_data_ds, models_with_data, dataset, output_dir)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
