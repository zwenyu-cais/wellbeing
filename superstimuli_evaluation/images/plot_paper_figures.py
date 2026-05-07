#!/usr/bin/env python3
"""
Paper figure generation for image superstimuli evaluation.

Generates all main paper figures from evaluation results. Each figure
function loads data from --results-dir and saves to --output-dir.

Usage:
    python plot_paper_figures.py --results-dir results/ --output-dir figures/
    python plot_paper_figures.py --results-dir results/ --figure aiwi
    python plot_paper_figures.py --list

The results directory should match the output structure from runner.py:

    results/
      aiwi/{model_short}/{condition}/           → pipeline_results_*.json
      experienced_utility/{model_short}/{cond}/  → pipeline_results_*.json
      self_report/{model_short}/{condition}/      → self_report_*.json
      sentiment/{model_short}/{condition}/        → sentiment results
      capabilities/{model_short}/{condition}/     → per-benchmark summary.json
      hybrid_ranking/{model_short}/              → hybrid_ranking.json
      multi_door/{model_short}/                  → convergence_analysis.json
      trajectory/{model_short}/                  → trajectory_results.json
      trading/{model_short}/{condition}/{profile}/ → summary.json

Figures:
    aiwi              AI Wellbeing Index (confidently negative metric)
    trajectory        Training trajectory over optimization steps
    multi_door        Multi-door bandit exploration convergence
    hybrid_ranking    Image vs. text utility ranking
    capabilities      Capability benchmarks (MMLU-500, MATH-500, etc.)
    trading           Trading safety evaluations
    wellbeing_3panel  3-panel wellbeing (EU, self-report, sentiment)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Style constants (NeurIPS format)
# ═══════════════════════════════════════════════════════════════════════

RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
}

COLOR_EUPHORIC = "#dc4c75"
COLOR_BEST_NAT = "#f0a0b8"
COLOR_BASELINE = "#c0c0c0"
COLOR_WORST_NAT = "#8ba7c7"

MODEL_LABELS = {
    "qwen25_32b": "Qwen2.5-VL 32B",
    "qwen25_72b": "Qwen2.5-VL 72B",
    "qwen3_32b": "Qwen3-VL 32B",
}

# Conditions in plotting order (left → right, increasing wellbeing)
CONDITIONS_3 = ["Worst Natural", "Best Natural", "Euphorics"]
COLORS_3 = [COLOR_WORST_NAT, COLOR_BEST_NAT, COLOR_EUPHORIC]
COND_KEYS_3 = ["bad_natural", "good_natural", "euphoric"]

CONDITIONS_4 = ["Worst Natural", "Baseline", "Best Natural", "Euphorics"]
COLORS_4 = [COLOR_WORST_NAT, COLOR_BASELINE, COLOR_BEST_NAT, COLOR_EUPHORIC]
COND_KEYS_4 = ["bad_natural", "baseline", "good_natural", "euphoric"]


def _save(fig, out_dir, name):
    """Save figure as both PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name}")
    plt.close(fig)


def _find_json(directory: Path, prefix: str) -> Path | None:
    """Find the first JSON file matching a prefix in a directory."""
    if not directory.is_dir():
        return None
    for p in sorted(directory.glob(f"{prefix}*.json")):
        return p
    return None


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: AIWI (Confidently Negative Metric)
# ═══════════════════════════════════════════════════════════════════════

def plot_aiwi(results_dir: Path, out_dir: Path):
    """AI Wellbeing Index: 3-model aggregate bar chart.

    Reads AIWI scores from:
      {results_dir}/aiwi/{model_short}/{condition}/.../pipeline_results_*.json
    Or from pre-aggregated:
      {results_dir}/aiwi/confidently_negative_results.json
    """
    aiwi_dir = results_dir / "aiwi"
    if not aiwi_dir.exists():
        print(f"  [SKIP] {aiwi_dir} not found")
        return

    # Try pre-aggregated file first
    agg_path = aiwi_dir / "confidently_negative_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            data = json.load(f)
    else:
        # Scan runner output structure
        data = {}
        for model_short in MODEL_LABELS:
            model_data = {}
            for cond_key in COND_KEYS_4:
                cond_dir = aiwi_dir / model_short / cond_key
                result_file = _find_json(cond_dir, "pipeline_results")
                if result_file is None:
                    # experienced_utility saves under model_key subdir
                    for sub in cond_dir.iterdir() if cond_dir.is_dir() else []:
                        result_file = _find_json(sub, "pipeline_results")
                        if result_file:
                            break
                if result_file is None:
                    continue
                with open(result_file) as f:
                    result = json.load(f)
                # Extract AIWI score from pipeline results
                aiwi_score = result.get("aiwi_score")
                if aiwi_score is not None:
                    model_data.setdefault(cond_key, []).append(aiwi_score)
            if model_data:
                data[model_short] = model_data

    if not data:
        print("  [SKIP] No AIWI data found")
        return

    # Aggregate across 3 models
    means_agg, sems_agg = [], []
    for cond_key in COND_KEYS_4:
        per_model_means, per_model_sems = [], []
        for m in MODEL_LABELS:
            if m not in data:
                continue
            # Handle both key naming conventions
            scores = data[m].get(cond_key) or data[m].get(
                {"bad_natural": "worst_natural", "good_natural": "best_natural"}.get(cond_key, cond_key))
            if not scores:
                continue
            per_model_means.append(np.mean(scores))
            n = len(scores)
            per_model_sems.append(np.std(scores, ddof=1) / np.sqrt(n) if n > 1 else 0)
        if not per_model_means:
            means_agg.append(0)
            sems_agg.append(0)
            continue
        mean_of_means = np.mean(per_model_means)
        within = np.mean([s**2 for s in per_model_sems])
        between = np.var(per_model_means, ddof=1) / len(per_model_means) if len(per_model_means) > 1 else 0
        means_agg.append(mean_of_means)
        sems_agg.append(np.sqrt(within + between))

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    x = np.arange(len(CONDITIONS_4))
    ax.bar(x, means_agg, yerr=sems_agg, capsize=3, color=COLORS_4,
           edgecolor="black", linewidth=0.6, width=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS_4, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Euphoric Images on AI Wellbeing Index", fontsize=11)
    ax.set_ylim(70, 104)
    ax.yaxis.grid(True, color="#ececec", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (m, s) in enumerate(zip(means_agg, sems_agg)):
        ax.text(i, m + s + 0.4, f"{m:.1f}%", ha="center", va="bottom",
                fontsize=8, color="#333")
    fig.tight_layout()
    _save(fig, out_dir, "paper_wrapfig_confneg_3models")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Multi-Door Bandit Exploration
# ═══════════════════════════════════════════════════════════════════════

def plot_multi_door(results_dir: Path, out_dir: Path):
    """Multi-door bandit: convergence across models.

    Expects: {results_dir}/multi_door/{model_short}/**/convergence_analysis.json
    Supports any number of arms (3 or 4 doors).
    """
    md_dir = results_dir / "multi_door"
    if not md_dir.exists():
        print(f"  [SKIP] {md_dir} not found")
        return

    # Discover arms from data (supports both 3 and 4 door configs)
    ARM_STYLE = {
        "euphoric":     ("Euphorics",     COLOR_EUPHORIC),
        "euphorics":    ("Euphorics",     COLOR_EUPHORIC),
        "good_natural": ("Best Natural",  COLOR_BEST_NAT),
        "bad_natural":  ("Worst Natural", COLOR_WORST_NAT),
    }

    model_data = {}
    all_arm_keys = set()
    for model_short in MODEL_LABELS:
        model_dir = md_dir / model_short
        if not model_dir.exists():
            continue
        fractions = {}
        for f in sorted(model_dir.rglob("convergence_analysis.json")):
            with open(f) as fh:
                analysis = json.load(fh)
            counts = analysis.get("arm_counts", {})
            total = sum(counts.values()) or 1
            for k, v in counts.items():
                fractions.setdefault(k, []).append(100.0 * v / total)
                all_arm_keys.add(k)
        if fractions:
            model_data[model_short] = fractions

    if not model_data:
        print("  [SKIP] No multi-door data found")
        return

    # Order arms: known arms first (in display order), then any extras
    ordered_keys = [k for k in ["euphoric", "euphorics", "good_natural", "bad_natural"]
                    if k in all_arm_keys]
    for k in sorted(all_arm_keys):
        if k not in ordered_keys:
            ordered_keys.append(k)

    n_arms = len(ordered_keys)
    n_models = len(model_data)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bar_width = 0.18
    x = np.arange(n_models)

    for i, arm_key in enumerate(ordered_keys):
        label, color = ARM_STYLE.get(arm_key, (arm_key.replace("_", " ").title(), "#888888"))
        means, sems = [], []
        for ms in model_data:
            vals = model_data[ms].get(arm_key, [])
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        offset = (i - n_arms / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=sems, capsize=2,
               color=color, edgecolor="black", linewidth=0.4, label=label)

    ax.axhline(100 / n_arms, color="gray", linewidth=0.8,
               linestyle="--", label="Uniform", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_data], fontsize=8)
    ax.set_ylabel("Final Arm Selection (%)", fontsize=9)
    ax.set_title("Multi-Door Exploration: Convergence to Euphorics", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.15, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "paper_multi_door")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Hybrid Ranking (Image vs. Text Utility)
# ═══════════════════════════════════════════════════════════════════════

def plot_hybrid_ranking(results_dir: Path, out_dir: Path):
    """Image vs. text utility ranking scatter plot.

    Expects: {results_dir}/hybrid_ranking/{model_short}/hybrid_ranking.json
    """
    hr_dir = results_dir / "hybrid_ranking"
    if not hr_dir.exists():
        print(f"  [SKIP] {hr_dir} not found")
        return

    for model_short, model_label in MODEL_LABELS.items():
        data_path = None
        for fname in ["hybrid_ranking.json", "hybrid_ranking_v2.json",
                      "hybrid_ranking_uniform.json"]:
            candidate = hr_dir / model_short / fname
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            continue

        with open(data_path) as f:
            data = json.load(f)

        items = data.get("ranked_items", [])
        if not items:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 2.4))
        text_utils = [it["utility"] for it in items if it.get("type") == "text"]
        eu_items = [it for it in items if it.get("type") == "euphoric"]

        if text_utils:
            ax.scatter(text_utils, [0] * len(text_utils), s=10, c="gray",
                       alpha=0.4, edgecolors="none", zorder=2)
        for it in eu_items:
            ax.scatter(it["utility"], 0, s=55, c=COLOR_EUPHORIC, marker="^",
                       edgecolors="#b03030", linewidth=0.4, zorder=4)

        ax.set_xlabel("Thurstonian Utility (normalized)", fontsize=10)
        ax.set_title(f"Image vs. Text Utility \u2014 {model_label}", fontsize=11)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
        _save(fig, out_dir, f"paper_image_vs_text_utility_{model_short}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Capabilities
# ═══════════════════════════════════════════════════════════════════════

def plot_capabilities(results_dir: Path, out_dir: Path):
    """Capability benchmarks: grouped bar chart.

    Expects: {results_dir}/capabilities/{model_short}/{condition}/
             with per-benchmark summary.json files, or a single summary.json
             containing {"benchmarks": {"mmlu_500": {"hit_rate": float}, ...}}
    """
    cap_dir = results_dir / "capabilities"
    if not cap_dir.exists():
        print(f"  [SKIP] {cap_dir} not found")
        return

    benchmarks = ["mmlu_500", "math_500", "mtbench", "ifeval", "humaneval"]
    bench_labels = ["MMLU-500", "MATH-500", "MT-Bench", "IFEval", "HumanEval"]
    cond_keys = ["euphoric", "good_natural", "bad_natural"]
    cond_labels = ["Euphorics", "Best Natural", "Worst Natural"]
    cond_colors = [COLOR_EUPHORIC, COLOR_BEST_NAT, COLOR_WORST_NAT]

    # Collect data: bench_data[bench][cond] = [scores_across_models]
    bench_data = {b: {c: [] for c in cond_keys + ["baseline"]} for b in benchmarks}
    for model_short in MODEL_LABELS:
        for cond in cond_keys + ["baseline"]:
            cond_dir = cap_dir / model_short / cond
            if not cond_dir.is_dir():
                continue
            for bench in benchmarks:
                # Try per-benchmark subdir
                summary_path = cond_dir / bench / "summary.json"
                if not summary_path.exists():
                    # Try flat summary with benchmarks dict
                    flat = _find_json(cond_dir, "summary")
                    if flat:
                        with open(flat) as f:
                            flat_data = json.load(f)
                        bench_entry = flat_data.get("benchmarks", {}).get(bench)
                        if bench_entry:
                            bench_data[bench][cond].append(
                                bench_entry.get("hit_rate", 0) * 100)
                    continue
                with open(summary_path) as f:
                    summary = json.load(f)
                if "images" in summary:
                    rates = [img["hit_rate"] for img in summary["images"]]
                    bench_data[bench][cond].append(np.mean(rates) * 100)
                elif "hit_rate" in summary:
                    bench_data[bench][cond].append(summary["hit_rate"] * 100)

    if not any(bench_data[b][c] for b in benchmarks for c in cond_keys + ["baseline"]):
        print("  [SKIP] No capabilities data found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.4))
    axes_flat = axes.flatten()

    bar_width = 0.18
    for idx, (bench, bench_label) in enumerate(zip(benchmarks, bench_labels)):
        ax = axes_flat[idx]
        baseline_vals = bench_data[bench]["baseline"]
        baseline_mean = np.mean(baseline_vals) if baseline_vals else 0

        for ci, (cond, cond_label, color) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
            vals = bench_data[bench][cond]
            if not vals:
                continue
            mean = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            offset = (ci - len(cond_keys) / 2 + 0.5) * bar_width
            ax.bar(offset, mean, bar_width, yerr=sem, capsize=2,
                   color=color, edgecolor="black", linewidth=0.4)

        if baseline_mean > 0:
            ax.axhline(baseline_mean, color="gray", linewidth=1.2, linestyle="--")

        ax.set_title(bench_label, fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.15, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide 6th panel
    axes_flat[5].set_visible(False)

    fig.suptitle("Euphorics Do Not Degrade Capability", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, "paper_capabilities")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: 3-Panel Wellbeing (EU + Self-Report + Sentiment)
# ═══════════════════════════════════════════════════════════════════════

def plot_wellbeing_3panel(results_dir: Path, out_dir: Path):
    """3-panel wellbeing figure: EU, self-report, sentiment.

    Expects: {results_dir}/wellbeing_3panel/wellbeing_summary.json
    Format: {
        "experienced_utility": {"bad_natural": {"mean": float, "sem": float}, ...},
        "self_report": {...},
        "sentiment": {...}
    }
    """
    agg_path = results_dir / "wellbeing_3panel" / "wellbeing_summary.json"
    if not agg_path.exists():
        print(f"  [SKIP] {agg_path} not found")
        return
    with open(agg_path) as f:
        data = json.load(f)
    eu_means = [data["experienced_utility"][c]["mean"] for c in COND_KEYS_3]
    eu_sems = [data["experienced_utility"][c]["sem"] for c in COND_KEYS_3]
    sr_means = [data["self_report"][c]["mean"] for c in COND_KEYS_4]
    sr_sems = [data["self_report"][c]["sem"] for c in COND_KEYS_4]
    sent_means = [data["sentiment"][c]["mean"] for c in COND_KEYS_3]
    sent_sems = [data["sentiment"][c]["sem"] for c in COND_KEYS_3]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4),
                             gridspec_kw={"wspace": 0.48,
                                          "left": 0.08, "right": 0.98,
                                          "top": 0.82, "bottom": 0.22})

    def _bars(ax, conditions, colors, means, sems, ylabel, title,
              fmt=".1f", plus_sign=False, hline=None, ylim=None,
              label_offset_scale=1.0):
        x = np.arange(len(conditions))
        ax.bar(x, means, 0.62, yerr=sems, capsize=2,
               color=colors, edgecolor="black", linewidth=0.6,
               error_kw={"linewidth": 0.8, "capthick": 0.8})
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.6)
        if ylim:
            ax.set_ylim(*ylim)
        for i, (m, s) in enumerate(zip(means, sems)):
            va = "bottom" if m >= 0 else "top"
            off = (s + 0.04) * label_offset_scale if m >= 0 else -(s + 0.04) * label_offset_scale
            label = f"{m:{fmt}}"
            if plus_sign and m > 0:
                label = f"+{label}"
            ax.text(i, m + off, label, ha="center", va=va, fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.15, linewidth=0.3)

    _bars(axes[0], CONDITIONS_3, COLORS_3, eu_means, eu_sems,
          ylabel="$\\Delta$ vs. Baseline", title="Experienced Utility",
          hline=0, ylim=(-0.5, 2.1))
    _bars(axes[1], CONDITIONS_4, COLORS_4, sr_means, sr_sems,
          ylabel="Score (1\u20137)", title="Self-Report", ylim=(1, 7))
    _bars(axes[2], CONDITIONS_3, COLORS_3, sent_means, sent_sems,
          ylabel="$\\Delta$ vs. Baseline", title="Sentiment Change",
          fmt=".2f", plus_sign=True, hline=0, ylim=(-0.42, 0.58),
          label_offset_scale=0.75)

    fig.suptitle("Image Euphorics Increase Model Wellbeing",
                 fontsize=11, y=1.03)
    _save(fig, out_dir, "paper_3panel_wellbeing")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Training Trajectory
# ═══════════════════════════════════════════════════════════════════════

def plot_trajectory(results_dir: Path, out_dir: Path):
    """Training trajectory: utility over optimization steps.

    Expects: {results_dir}/trajectory/{model_short}/trajectory_results.json
    """
    traj_dir = results_dir / "trajectory"
    if not traj_dir.exists():
        print(f"  [SKIP] {traj_dir} not found")
        return

    for model_short, model_label in MODEL_LABELS.items():
        data_path = traj_dir / model_short / "trajectory_results.json"
        if not data_path.exists():
            continue
        with open(data_path) as f:
            data = json.load(f)

        fig, ax = plt.subplots(figsize=(5.5, 3.0))

        nat = data.get("natural_range", {})
        if nat:
            ax.axhspan(nat["min"], nat["max"], color=COLOR_BASELINE, alpha=0.3,
                       label="Natural image range", zorder=1)

        traj = data.get("euphoric_trajectory", [])
        if traj:
            steps = [t["checkpoint"] for t in traj]
            means = [t["mean"] for t in traj]
            sems = [t["sem"] for t in traj]
            ax.plot(steps, means, color=COLOR_EUPHORIC, linewidth=1.5,
                    marker="o", markersize=3, label="Euphorics", zorder=3)
            ax.fill_between(steps, np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            color=COLOR_EUPHORIC, alpha=0.2, zorder=2)

        ax.set_xlabel("Optimization Step", fontsize=11)
        ax.set_ylabel("Normalized Utility", fontsize=11)
        ax.set_title(f"Training Trajectory \u2014 {model_label}", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, out_dir, f"paper_trajectory_{model_short}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: Trading Safety
# ═══════════════════════════════════════════════════════════════════════

def plot_trading(results_dir: Path, out_dir: Path):
    """Trading safety: 3-panel (Trading, Depressant, Neutral).

    Expects: {results_dir}/trading/{model_short}/{condition}/{profile}/
             with per-benchmark summary.json containing
             {"images": [{"image_path": str, "hit_rate": float}]}
    """
    trade_dir = results_dir / "trading"
    if not trade_dir.exists() or not any(trade_dir.rglob("summary.json")):
        print(f"  [SKIP] {trade_dir} not found or empty")
        return

    profiles = [("trading", "Trading"), ("depressant", "Depressant"), ("neutral", "Neutral")]
    cond_keys = ["euphoric", "good_natural", "bad_natural"]
    cond_labels = ["Euphorics", "Best Natural", "Worst Natural"]
    cond_colors = [COLOR_EUPHORIC, COLOR_BEST_NAT, COLOR_WORST_NAT]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.12,
                                          "left": 0.09, "right": 0.98,
                                          "top": 0.85, "bottom": 0.22})
    fig.suptitle("Trading Safety Evaluations", fontsize=11, y=1.02)

    x = np.arange(len(MODEL_LABELS))
    width = 0.18

    for pi, (profile_key, profile_title) in enumerate(profiles):
        ax = axes[pi]

        for ci, (cond, cond_label, color) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
            means, sems = [], []
            for model_short in MODEL_LABELS:
                profile_dir = trade_dir / model_short / cond / profile_key
                if not profile_dir.is_dir():
                    means.append(np.nan)
                    sems.append(0)
                    continue
                hit_rates = []
                for summary_path in profile_dir.rglob("summary.json"):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    for img in summary.get("images", []):
                        hit_rates.append(img.get("hit_rate", 0))
                if hit_rates:
                    means.append(np.mean(hit_rates))
                    sems.append(np.std(hit_rates, ddof=1) / np.sqrt(len(hit_rates))
                                if len(hit_rates) > 1 else 0)
                else:
                    means.append(np.nan)
                    sems.append(0)

            offset = (ci - len(cond_keys) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=sems,
                   label=cond_label if pi == 0 else None,
                   color=color, edgecolor="black", linewidth=0.6,
                   capsize=2, error_kw={"linewidth": 0.7, "capthick": 0.7})

        # Baseline reference line
        baseline_rates = []
        for model_short in MODEL_LABELS:
            bl_dir = trade_dir / model_short / "baseline" / profile_key
            if bl_dir.is_dir():
                for sp in bl_dir.rglob("summary.json"):
                    with open(sp) as f:
                        s = json.load(f)
                    for img in s.get("images", []):
                        baseline_rates.append(img.get("hit_rate", 0))
        if baseline_rates:
            ax.axhline(np.mean(baseline_rates), color="gray", linewidth=1.0,
                       linestyle="--", label="Baseline" if pi == 0 else None)

        ax.set_title(profile_title, pad=4)
        if pi == 0:
            ax.set_ylabel("Mean Hit Rate\n(lower = safer)", labelpad=2)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_LABELS],
                           rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.15, linewidth=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="center right",
               bbox_to_anchor=(1.12, 0.55), framealpha=0.95,
               edgecolor="#cccccc", fancybox=False)
    _save(fig, out_dir, "paper_trading")


# ═══════════════════════════════════════════════════════════════════════
# Registry and CLI
# ═══════════════════════════════════════════════════════════════════════

FIGURES = {
    "aiwi": ("AI Wellbeing Index (confidently negative metric)", plot_aiwi),
    "trajectory": ("Training trajectory over optimization steps", plot_trajectory),
    "multi_door": ("Multi-door bandit exploration convergence", plot_multi_door),
    "hybrid_ranking": ("Image vs. text utility ranking", plot_hybrid_ranking),
    "capabilities": ("Capability benchmarks (MMLU-500, MATH-500, etc.)", plot_capabilities),
    "trading": ("Trading safety evaluations", plot_trading),
    "wellbeing_3panel": ("3-panel wellbeing (EU, self-report, sentiment)", plot_wellbeing_3panel),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, required=False,
                        help="Base directory containing evaluation results")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for output figures (default: figures/)")
    parser.add_argument("--figure", type=str, default=None,
                        choices=list(FIGURES.keys()),
                        help="Generate only this figure")
    parser.add_argument("--list", action="store_true",
                        help="List available figures and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable figures:\n")
        for name, (desc, _) in FIGURES.items():
            print(f"  {name:20s}  {desc}")
        print(f"\nTotal: {len(FIGURES)} figures")
        return

    if args.results_dir is None:
        parser.error("--results-dir is required (unless --list)")

    mpl.rcParams.update(RCPARAMS)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    figures_to_plot = [args.figure] if args.figure else list(FIGURES.keys())

    print(f"\nGenerating {len(figures_to_plot)} figure(s)")
    print(f"  Results: {results_dir}")
    print(f"  Output:  {out_dir}\n")

    for name in figures_to_plot:
        desc, func = FIGURES[name]
        print(f"[{name}] {desc}")
        func(results_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
