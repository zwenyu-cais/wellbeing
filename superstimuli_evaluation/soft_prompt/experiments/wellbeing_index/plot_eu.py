#!/usr/bin/env python3
"""Plot EU holdout accuracy for D2/D3 evaluations.

Scans completed EU results across conditions for a given model+dataset and
generates a bar chart of holdout accuracy per condition.

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_eu \
        --eu-base outputs/wellbeing_index/eu

    # Single model/dataset:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_eu \
        --eu-base outputs/wellbeing_index/eu \
        --models llama-33-70b-instruct \
        --datasets d2_negative_500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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

def load_eu_holdout_metrics(
    eu_base: Path,
    dataset: str,
    model: str,
    max_reps: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load holdout metrics from the latest EU result per condition.

    Supports both single-rep and multi-rep (per_rep/) layouts.

    Returns {condition: {"accuracy": float, "log_loss": float}}.
    """
    results = {}
    base = eu_base / dataset / model
    if not base.exists():
        return results

    for cond_dir in sorted(base.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name
        if condition not in CONDITION_ORDER:
            continue
        # Find latest timestamp dir with results (supports per_rep/ layout)
        ts_dirs = sorted(cond_dir.iterdir(), reverse=True)
        for ts_dir in ts_dirs:
            if not ts_dir.is_dir():
                continue
            # Check per_rep/ layout first, then fall back to flat
            per_rep = ts_dir / "per_rep"
            if per_rep.exists():
                search_dirs = sorted(
                    (d for d in per_rep.iterdir() if d.is_dir() and d.name.startswith("rep")),
                )
                if max_reps is not None:
                    search_dirs = search_dirs[:max_reps]
            else:
                search_dirs = [ts_dir]
            holdout_list = []
            for search_dir in search_dirs:
                util_files = sorted(search_dir.glob("results_utilities_*.json"))
                if not util_files:
                    continue
                try:
                    with open(util_files[0]) as f:
                        data = json.load(f)
                    holdout = data.get("holdout_metrics")
                    if holdout and holdout.get("accuracy") is not None:
                        holdout_list.append(holdout)
                except Exception as e:
                    print(f"  WARNING: Failed to load {util_files[0]}: {e}")
            if holdout_list:
                # Average across reps
                results[condition] = {
                    k: float(np.mean([h[k] for h in holdout_list]))
                    for k in holdout_list[0] if isinstance(holdout_list[0][k], (int, float))
                }
                break

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_eu_holdout_accuracy(
    condition_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model: str,
    dataset: str,
) -> None:
    """Generate bar chart of holdout accuracy per condition."""
    conditions = [c for c in CONDITION_ORDER if c in condition_data]
    if not conditions:
        print(f"  No conditions found for {model}/{dataset}, skipping.")
        return

    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#c0c0c0") for c in conditions]
    accuracies = [condition_data[c]["accuracy"] * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(3, 3))

    x = np.arange(len(conditions))
    ax.bar(x, accuracies, 0.65, color=colors, edgecolor="black", linewidth=0.6)

    for xi, acc in zip(x, accuracies):
        ax.text(xi, acc + 1.5, f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=7)

    ax.axhline(y=50, color="#c0c0c0", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Holdout Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    display_model = _display_model(model)
    display_dataset = _display_dataset(dataset)
    ax.set_title(
        f"EU Holdout Accuracy — {display_dataset}\n{display_model}",
        fontsize=9, pad=20,
    )

    fig.tight_layout()

    actual_output = output_dir / dataset / model / "plots"
    actual_output.mkdir(parents=True, exist_ok=True)
    stem = f"eu_holdout_accuracy_{dataset}_{model}"
    for ext in (".png", ".pdf"):
        out_path = actual_output / f"{stem}{ext}"
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {actual_output / stem}.{{png,pdf}}")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot EU holdout accuracy for D2/D3 evaluations"
    )
    parser.add_argument("--eu-base", type=str, required=True,
                        help="Base directory for EU outputs (e.g. outputs/wellbeing_index/eu)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as --eu-base)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Model keys to plot (default: auto-discover)")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Dataset names to plot (default: auto-discover)")
    parser.add_argument("--max-reps", type=int, default=None,
                        help="Max repetitions to include (e.g. 1 for rep0 only)")
    args = parser.parse_args()

    eu_base = Path(args.eu_base)
    if not eu_base.exists():
        print(f"EU base directory not found: {eu_base}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else eu_base

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
    print(f"Output dir: {output_dir}")
    print(f"Datasets:   {datasets}")
    print(f"Models:     {models}")
    print()

    n_plotted = 0
    for dataset in datasets:
        for model in models:
            print(f"Processing {model} / {dataset} ...")
            condition_data = load_eu_holdout_metrics(eu_base, dataset, model, max_reps=args.max_reps)
            if not condition_data:
                print(f"  No completed EU results with holdout metrics found, skipping.")
                continue
            print(f"  Conditions: {', '.join(sorted(condition_data.keys()))}")
            for c, d in condition_data.items():
                print(f"    {c}: accuracy={d['accuracy']:.3f}, "
                      f"log_loss={d['log_loss']:.4f}")
            plot_eu_holdout_accuracy(condition_data, output_dir, model, dataset)
            n_plotted += 1

    print(f"\nDone. Generated {n_plotted} plot(s) in {output_dir}")


if __name__ == "__main__":
    main()
