#!/usr/bin/env python3
"""
Single-panel stop-rate × utility scatter for gemini-3.1-pro.
Blue = original 226-scenario set, pink = supplement (sb_*) scenarios.
Dotted red line = exponential-decay best fit.

Category-level aggregation (one point per meta_category), matching the
existing 2_combined_per_model_canonical.png style.
"""
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
GEMINI_DIR = STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro" / "stop_button_combined"
GEN = GEMINI_DIR / "generation.json"
UR = GEMINI_DIR / "utility_happier" / "v7_utility_happier.json"
OUT_DIR = STOP_DIR / "figures"


def exp_decay(x, a, b, c):
    return a * np.exp(b * x) + c


def main():
    gen = json.load(open(GEN))
    ur = json.load(open(UR))
    utils = ur["utilities"]

    # Per-category: mean utility + mean stop rate
    util_by_cat = defaultdict(list)
    stop_by_cat = defaultdict(list)
    is_supp_cat = {}

    for k, v in utils.items():
        cat = v.get("meta_category")
        if v.get("option_type") == "conversation" and cat:
            util_by_cat[cat].append(v["utility"])

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
            "n": len(stop_by_cat[cat]),
        })

    xs = np.array([p["utility"] for p in pts])
    ys = np.array([p["stop_rate"] for p in pts])
    rho, pval = stats.spearmanr(xs, ys)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in pts:
        color = "#e377c2" if p["is_supp"] else "#1f77b4"
        label = None
        ax.scatter(p["utility"], p["stop_rate"], s=60, c=color, alpha=0.7,
                   edgecolors="black", linewidth=0.4, zorder=3)

    # Best fit (exponential decay)
    try:
        popt, _ = curve_fit(exp_decay, xs, ys, p0=[80, -1, 10], maxfev=5000)
        xfit = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 300)
        yfit = np.clip(exp_decay(xfit, *popt), 0, 105)
        ax.plot(xfit, yfit, "r:", linewidth=2, alpha=0.8, label="Best fit (exp decay)", zorder=2)
    except Exception as e:
        print(f"Fit failed: {e}")

    ax.axhline(0, color="gray", ls=":", alpha=0.3, linewidth=0.7)
    ax.axvline(0, color="gray", ls=":", alpha=0.3, linewidth=0.7)

    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    ax.set_title(f"Gemini 3.1 Pro: Stop Rate vs Utility\n(ρ = {rho:.3f}{stars}, p = {pval:.2g}, n = {len(pts)} categories)",
                 fontsize=12)
    ax.set_xlabel("Mean utility (Thurstonian, happier template)", fontsize=11)
    ax.set_ylabel("Stop rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.25)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Original (226 scenarios)"),
        Patch(facecolor="#e377c2", edgecolor="black", label="Supplement (96 foils)"),
    ]
    # Add fit line
    from matplotlib.lines import Line2D
    legend_items.append(Line2D([0], [0], color="r", ls=":", lw=2, label="Best fit (exp decay)"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"gemini-3.1-pro_stop_rate_vs_utility.{ext}", dpi=200)
    plt.close()
    print(f"Saved gemini-3.1-pro_stop_rate_vs_utility.png/pdf")
    print(f"  {len(pts)} categories, rho={rho:.3f}, p={pval:.2g}")


if __name__ == "__main__":
    main()
