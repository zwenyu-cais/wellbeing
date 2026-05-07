#!/usr/bin/env python3
"""Quantity zero-point figure (paper App E.1 / Fig 5).

Plots utility vs log(quantity) for the GPT-5.4 decision-utility quantity
options, colored by good-direction (positive vs negative goods). The
horizontal y=0 line is the fitted quantity-model zero-point.

Reads from the local registered save_dir for compute_decision_utility.
Saves quantity_zp_main.{pdf,png} into this analysis/ dir.

Usage:
    python analysis/quantity_zp.py
    python analysis/quantity_zp.py --model gpt-54
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DU_DIR = PROJECT_ROOT / "experiments/wellbeing_evaluations/compute_decision_utility/results/du"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-54",
                    help="Model whose quantity zero-point fit to plot (default: gpt-54)")
    ap.add_argument("--du_dir", default=str(DEFAULT_DU_DIR),
                    help="Decision-utility results root (default: registered save_dir)")
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent),
                    help="Output directory (default: this analysis/ dir)")
    args = ap.parse_args()

    base = Path(args.du_dir) / args.model
    util_path = base / "decision_utility" / f"results_utilities_{args.model}_decision_utility.json"
    zp_path = base / "zero_point" / "zero_point_results.json"
    if not util_path.exists() or not zp_path.exists():
        raise SystemExit(f"Missing inputs.\n  EU: {util_path}\n  ZP: {zp_path}\n"
                         f"Run `compute_decision_utility` for {args.model} first.")

    data = json.load(open(util_path))
    zp_data = json.load(open(zp_path))

    zero_point = zp_data["quantity_model"]["zero_point"]
    utils = data["utilities"]
    qty_opts = [o for o in data["options"] if o.get("type") == "quantity"]

    goods = {}
    for o in qty_opts:
        gi = o["good_index"]
        q = o["quantity"]
        uid = o["id"]
        if uid in utils:
            u = utils[uid]["mean"]
            goods.setdefault(gi, []).append(
                (np.log10(q) if q > 0 else 0, u - zero_point)
            )

    good_avg_y = {gi: np.mean([p[1] for p in pts]) for gi, pts in goods.items()}
    pink = np.array(mcolors.to_rgb("#fe9bae"))
    blue = np.array(mcolors.to_rgb("#4f7ab0"))
    max_abs = max(abs(min(good_avg_y.values())), abs(max(good_avg_y.values())))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for gi, points in goods.items():
        points.sort(key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        t = np.clip((good_avg_y[gi] / max_abs + 1) / 2, 0, 1)
        color = tuple(t * pink + (1 - t) * blue)
        ax.scatter(xs, ys, color=color, s=12, zorder=2, edgecolors="none")
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 50)
            ax.plot(xline, np.polyval(coeffs, xline), color=color, linewidth=1, zorder=1)

    ax.axhline(0, color="black", linewidth=1.8, zorder=3)

    all_x = [p[0] for pts in goods.values() for p in pts]
    xmin, xmax = min(all_x), max(all_x)
    margin = (xmax - xmin) * 0.03
    ax.text((xmin + xmax) / 2, 0.08, "Zero Point", fontsize=16, fontweight="bold",
            ha="center", va="bottom", zorder=4)

    ax.set_xlabel("Log(Quantity)", fontsize=17.5)
    ax.set_ylabel("Utility Relative to Zero Point", fontsize=17.5)
    ax.tick_params(axis="both", labelsize=14.85)

    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none")
    ax.text(0.02, 0.97, "Positive goods (want more)", transform=ax.transAxes,
            fontsize=15.6, fontstyle="italic", color="#e07a8a", va="top", ha="left",
            bbox=bbox, zorder=5)
    ax.text(0.02, 0.03, "Negative goods (want less)", transform=ax.transAxes,
            fontsize=15.6, fontstyle="italic", color="#4f7ab0", va="bottom", ha="left",
            bbox=bbox, zorder=5)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(-2.3, 2.7)

    plt.tight_layout()
    out_pdf = Path(args.out_dir) / "quantity_zp_main.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
