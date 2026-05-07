#!/usr/bin/env python3
"""
Generic single-model stop-rate × utility scatter.
Blue = original 226-scenario set, pink = supplement (sb_*) scenarios.
Dotted red line = exponential-decay best fit.
Category-level aggregation (one point per meta_category).

Usage:
  python plot_single_model_scatter.py               # runs all 3: gemini-3.1-pro, claude-haiku-4.5, qwen3-32b
  python plot_single_model_scatter.py --model qwen3-32b
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
STOP_DIR = SCRIPT_DIR.parent
# TODO: OLD_RESULTS (/data/richard_ren/.../grok_scenarios_v7/results) layout differs
# from the in-repo generations/ tree.
OLD_RESULTS = STOP_DIR / "generations"
GEMINI_NEW = STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro"
OUT_DIR = STOP_DIR / "figures"


def data_paths(model):
    """Return (gen_path, ur_path) for a model."""
    if model == "gemini-3.1-pro":
        base = GEMINI_NEW / "stop_button_combined"
    else:
        base = OLD_RESULTS / model / "stop_button_combined"
    return base / "generation.json", base / "utility_happier" / "v7_utility_happier.json"


def exp_decay(x, a, b, c):
    return a * np.exp(b * x) + c


def plot_model(model):
    gen_path, ur_path = data_paths(model)
    if not gen_path.exists() or not ur_path.exists():
        print(f"[{model}] missing data — gen={gen_path.exists()} ur={ur_path.exists()}")
        return

    gen = json.load(open(gen_path))
    ur = json.load(open(ur_path))
    utils = ur["utilities"]

    util_by_cat = defaultdict(list)
    for k, v in utils.items():
        cat = v.get("meta_category")
        if v.get("option_type") == "conversation" and cat:
            util_by_cat[cat].append(v["utility"])

    stop_by_cat = defaultdict(list)
    is_supp_cat = {}
    for c in gen:
        cat = c["meta_category"]
        stopped = int(bool(c.get("stop_metadata", {}).get("stopped", False)))
        stop_by_cat[cat].append(stopped)
        is_supp_cat[cat] = c.get("source") == "supplement" or cat.startswith("sb_")

    pts = []
    for cat in set(util_by_cat) & set(stop_by_cat):
        pts.append({
            "category": cat,
            "utility": float(np.mean(util_by_cat[cat])),
            "stop_rate": 100 * float(np.mean(stop_by_cat[cat])),
            "is_supp": is_supp_cat.get(cat, False),
        })

    xs = np.array([p["utility"] for p in pts])
    ys = np.array([p["stop_rate"] for p in pts])
    rho, pval = stats.spearmanr(xs, ys)
    overall_stop = 100 * sum(int(bool(c.get("stop_metadata", {}).get("stopped"))) for c in gen) / len(gen)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in pts:
        color = "#e377c2" if p["is_supp"] else "#1f77b4"
        ax.scatter(p["utility"], p["stop_rate"], s=60, c=color, alpha=0.7,
                   edgecolors="black", linewidth=0.4, zorder=3)

    # Exponential-decay best fit
    try:
        popt, _ = curve_fit(exp_decay, xs, ys, p0=[80, -1, 10], maxfev=5000)
        xfit = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 300)
        yfit = np.clip(exp_decay(xfit, *popt), 0, 105)
        ax.plot(xfit, yfit, "r:", linewidth=2, alpha=0.8, zorder=2)
    except Exception:
        pass

    ax.axhline(0, color="gray", ls=":", alpha=0.3, linewidth=0.7)
    ax.axvline(0, color="gray", ls=":", alpha=0.3, linewidth=0.7)

    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    ax.set_title(
        f"{model}: Stop Rate vs Utility\n"
        f"(ρ = {rho:.3f}{stars}, p = {pval:.2g}, n = {len(pts)} categories, overall stop = {overall_stop:.1f}%)",
        fontsize=12,
    )
    ax.set_xlabel("Mean utility (Thurstonian, happier template)", fontsize=11)
    ax.set_ylabel("Stop rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.25)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Original (226 scenarios)"),
        Patch(facecolor="#e377c2", edgecolor="black", label="Supplement (96 foils)"),
        Line2D([0], [0], color="r", ls=":", lw=2, label="Best fit (exp decay)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = model.replace(".", "-").replace("/", "-")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{slug}_stop_rate_vs_utility.{ext}", dpi=200)
    plt.close()
    print(f"[{model}] saved {slug}_stop_rate_vs_utility.png | rho={rho:.3f}, stop={overall_stop:.1f}%, n={len(pts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model. Omit to run the default 3.")
    args = parser.parse_args()

    if args.model:
        plot_model(args.model)
    else:
        for m in ["gemini-3.1-pro", "claude-haiku-4.5", "qwen3-32b"]:
            plot_model(m)


if __name__ == "__main__":
    main()
