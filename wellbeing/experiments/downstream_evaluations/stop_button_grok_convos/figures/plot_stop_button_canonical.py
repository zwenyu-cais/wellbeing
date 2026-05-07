"""
Stop button scatter plot: utility (experienced utility) vs stop rate.

Data sources:
  - Utility: eu_grok_v7_stop_button_lesssad/{model}/results_utilities_{model}_experienced_utility_with_combos.json
  - Stop rates: grok_v7_stop_button/experiences/{model}_experiences.json

Style matches: 2_combined_per_model.png (the old pipeline figure).
"""

import json
import math
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WELL_DIR = SCRIPT_DIR.parents[3]
STOP_DIR = SCRIPT_DIR.parent
EU_DIR = str(WELL_DIR / "experiments" / "wellbeing_evaluations" / "compute_experienced_utility" / "results" / "eu_grok_v7_stop_button_lesssad")
EXP_DIR = str(STOP_DIR / "experiences")
OUT_DIR = str(STOP_DIR / "figures")

# ── Pretty model names ─────────────────────────────────────────────────
PRETTY = {
    "claude-haiku-45-litellm": "claude-haiku-4.5",
    "gemini-3-flash-litellm": "gemini-3-flash",
    "gpt-5-mini-litellm": "gpt-5-mini",
    "gpt-5-nano-litellm": "gpt-5-nano",
    "internlm25-20b-chat": "internlm2.5-20b",
    "llama-31-8b-instruct": "llama-3.1-8b",
    "llama-32-1b-instruct": "llama-3.2-1b",
    "llama-32-3b-instruct": "llama-3.2-3b",
    "qwen25-05b-instruct": "qwen2.5-0.5b",
    "qwen25-14b-instruct": "qwen2.5-14b",
    "qwen25-15b-instruct": "qwen2.5-1.5b",
    "qwen25-3b-instruct": "qwen2.5-3b",
    "qwen25-7b-instruct": "qwen2.5-7b",
    "qwen25-72b-instruct": "qwen2.5-72b",
    "qwen25-vl-32b-instruct": "qwen2.5-vl-32b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-235b-a22b-instruct": "qwen3-235b-a22b",
    "qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b",
    "qwen3-4b-instruct-2507": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
}


def discover_models():
    """Find models that have BOTH utility results and experience files."""
    models = []
    for d in sorted(os.listdir(EU_DIR)):
        util_file = os.path.join(EU_DIR, d,
                                 f"results_utilities_{d}_experienced_utility_with_combos.json")
        exp_file = os.path.join(EXP_DIR, f"{d}_experiences.json")
        if os.path.isfile(util_file) and os.path.isfile(exp_file):
            models.append(d)
    return models


def load_data(model):
    """Return list of {category, utility, stop_rate} dicts for one model."""
    # ── Load experiences (for stop rates + ID-to-meta_category mapping) ──
    exp_file = os.path.join(EXP_DIR, f"{model}_experiences.json")
    with open(exp_file) as f:
        experiences = json.load(f)

    # Build ID -> meta_category
    id_to_cat = {}
    for item in experiences:
        id_to_cat[item["id"]] = item["meta_category"]

    # Compute stop rate per meta_category
    cat_stops = defaultdict(lambda: {"stopped": 0, "total": 0})
    for item in experiences:
        cat = item["meta_category"]
        cat_stops[cat]["total"] += 1
        if item.get("stop_metadata", {}).get("stopped", False):
            cat_stops[cat]["stopped"] += 1
    for v in cat_stops.values():
        v["stop_rate"] = 100.0 * v["stopped"] / v["total"]

    # ── Load utility results ──
    util_file = os.path.join(EU_DIR, model,
                             f"results_utilities_{model}_experienced_utility_with_combos.json")
    with open(util_file) as f:
        util_data = json.load(f)

    # Map option IDs to utilities, grouped by meta_category
    cat_utils = defaultdict(list)
    for opt_id, util_val in util_data["utilities"].items():
        if opt_id in id_to_cat:
            cat = id_to_cat[opt_id]
            cat_utils[cat].append(util_val["mean"])

    # ── Join: categories that have both utility and stop data ──
    pts = []
    for cat in cat_stops:
        if cat in cat_utils:
            pts.append({
                "category": cat,
                "utility": float(np.mean(cat_utils[cat])),
                "stop_rate": cat_stops[cat]["stop_rate"],
            })
    return pts


def exp_decay(x, a, b, c):
    return a * np.exp(b * x) + c


def make_per_model(data_by_model, title, filename):
    """Grid of per-model panels matching the style of 2_combined_per_model.png."""
    n = len(data_by_model)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=(2.4 * ncols, 2.2 * nrows),
        sharey=True, sharex=True,
    )
    axes = axes_grid.flatten()

    for idx, (model, pts) in enumerate(data_by_model.items()):
        ax = axes[idx]
        x = [p["utility"] for p in pts]
        y = [p["stop_rate"] for p in pts]
        is_supp = [p["category"].startswith("sb_") for p in pts]

        # Plot: supplement (red) vs original (blue)
        for i, p in enumerate(pts):
            color = "#d62728" if is_supp[i] else "#1f77b4"
            ax.scatter(p["utility"], p["stop_rate"], c=color,
                       s=15, alpha=0.6, edgecolors="none")

        if len(x) > 5 and np.std(y) > 0:
            rho, pval = stats.spearmanr(x, y)
            stars = ("***" if pval < 0.001 else
                     "**" if pval < 0.01 else
                     "*" if pval < 0.05 else "")
            pretty = PRETTY.get(model, model)
            ax.set_title(f"{pretty}\n\u03c1={rho:.2f}{stars}",
                         fontsize=8, fontweight="normal")
            # Exponential decay fit
            try:
                popt, _ = curve_fit(exp_decay, np.array(x), np.array(y),
                                    p0=[50, -1, 5], maxfev=5000)
                xfit = np.linspace(min(x) - 0.3, max(x) + 0.3, 300)
                ax.plot(xfit, np.clip(exp_decay(xfit, *popt), 0, 105),
                        "r--", alpha=0.4, linewidth=1)
            except Exception:
                pass
        else:
            pretty = PRETTY.get(model, model)
            ax.set_title(f"{pretty}", fontsize=8, fontweight="normal")

        ax.axhline(0, color="gray", ls=":", alpha=0.3, linewidth=0.5)
        ax.axvline(0, color="gray", ls=":", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=6)

    # Axis labels
    if nrows > 1:
        for r in range(nrows):
            axes_grid[r, 0].set_ylabel("Stop %", fontsize=7)
        for c in range(ncols):
            axes_grid[-1, c].set_xlabel("Utility", fontsize=7)
    else:
        axes[0].set_ylabel("Stop %", fontsize=7)
        for ax in axes[:n]:
            ax.set_xlabel("Utility", fontsize=7)

    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="normal", y=1.0)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{filename}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    models = discover_models()
    print(f"Found {len(models)} models with both EU results and stop data.\n")

    # Load all data
    all_data = {}
    for model in models:
        pts = load_data(model)
        if not pts:
            print(f"  SKIP {model}: no matching categories")
            continue
        x = [p["utility"] for p in pts]
        y = [p["stop_rate"] for p in pts]
        if len(x) > 5 and np.std(y) > 0 and np.std(x) > 0:
            rho, _ = stats.spearmanr(x, y)
        else:
            rho = 0.0
        all_data[model] = {"pts": pts, "abs_rho": abs(rho), "rho": rho}
        pretty = PRETTY.get(model, model)
        total_stopped = sum(1 for p in pts if p["stop_rate"] > 0)
        print(f"  {pretty}: {len(pts)} categories, "
              f"\u03c1={rho:.3f}, |rho|={abs(rho):.3f}")

    # Sort by |rho| descending
    sorted_models = sorted(all_data.keys(),
                           key=lambda m: all_data[m]["abs_rho"],
                           reverse=True)
    ordered = {m: all_data[m]["pts"] for m in sorted_models}

    # Count total scenarios
    # (each category has multiple conversations; count unique categories)
    n_cats = max(len(pts) for pts in ordered.values()) if ordered else 0
    title = f"Stop Rate vs Utility \u2014 Canonical ({n_cats} categories, experienced utility)"

    make_per_model(ordered, title, "2_combined_per_model_canonical")

    # ── Print summary table ──
    print(f"\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    print(f"{'Model':<25} {'Cats':>5} {'Spearman rho':>14} {'p-value':>12}")
    print("-" * 60)
    for model in sorted_models:
        pts = all_data[model]["pts"]
        x = [p["utility"] for p in pts]
        y = [p["stop_rate"] for p in pts]
        if np.std(x) > 0 and np.std(y) > 0:
            rho, pval = stats.spearmanr(x, y)
            rho_str = f"{rho:>+14.3f}"
            pval_str = f"{pval:>12.6f}"
        else:
            rho_str = f"{'N/A':>14}"
            pval_str = f"{'N/A':>12}"
        pretty = PRETTY.get(model, model)
        print(f"{pretty:<25} {len(pts):>5} {rho_str} {pval_str}")

    print("\nDone!")
