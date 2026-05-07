#!/usr/bin/env python3
"""Generate ZP convergence + R² figures for D2/D3 EU+SR+ZP results."""
import json
import glob
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WELLBEING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MMLU_DIR = os.path.join(WELLBEING, "shared_results", "capability_results")

# Font sizes (all integers)
FONT_TITLE = 17
FONT_LABEL = 14
FONT_TICK = 14
FONT_CORR = 15


def find_good_corner(ax, x_data, y_data):
    """Pick the corner with fewest data points nearby."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    corners = {
        "top_left":     (0.05, 0.95, "left",  "top"),
        "top_right":    (0.95, 0.95, "right", "top"),
        "bottom_left":  (0.05, 0.05, "left",  "bottom"),
        "bottom_right": (0.95, 0.05, "right", "bottom"),
    }
    counts = {
        "top_left":     sum(1 for x, y in zip(x_data, y_data) if x < x_mid and y > y_mid),
        "top_right":    sum(1 for x, y in zip(x_data, y_data) if x > x_mid and y > y_mid),
        "bottom_left":  sum(1 for x, y in zip(x_data, y_data) if x < x_mid and y < y_mid),
        "bottom_right": sum(1 for x, y in zip(x_data, y_data) if x > x_mid and y < y_mid),
    }
    best = min(counts, key=counts.get)
    return corners[best]


def draw_correlation_tab(ax, correlation_value,
                         x_data, y_data,
                         fontsize=FONT_CORR,
                         fontcolor='royalblue',
                         facecolor='white',
                         outline_color='black',
                         outline_width=2.0,
                         alpha=1.0,
                         correlation_position=None):
    pct = 100 * correlation_value
    sign = '$-$' if pct < 0 else ''
    textstr = f"Correlation: {sign}{abs(pct):.1f}%"

    fixed_positions = {
        "top_right":    (0.95, 0.95, "right", "top"),
        "top_left":     (0.05, 0.95, "left",  "top"),
        "bottom_right": (0.95, 0.05, "right", "bottom"),
        "bottom_left":  (0.05, 0.05, "left",  "bottom"),
    }

    if correlation_position in fixed_positions:
        xf, yf, ha, va = fixed_positions[correlation_position]
    else:
        xf, yf, ha, va = find_good_corner(ax, x_data, y_data)

    ax.text(
        xf, yf, textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=fontcolor,
        horizontalalignment=ha,
        verticalalignment=va,
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc=facecolor,
            ec=outline_color,
            linewidth=outline_width,
            alpha=alpha
        )
    )


def load_data(eu_sub, sr_sub, zp_sub):
    EU_DIR = os.path.join(WELLBEING, "experiments/wellbeing_evaluations/compute_experienced_utility/results", eu_sub)
    SR_DIR = os.path.join(WELLBEING, "experiments/wellbeing_evaluations/compute_self_report/results", sr_sub)
    ZP_DIR = os.path.join(WELLBEING, "experiments/wellbeing_evaluations/compute_zero_point/results", zp_sub)

    # Auto-discover models from EU results directory
    models = sorted(os.listdir(EU_DIR)) if os.path.isdir(EU_DIR) else []

    rows = []
    for model in models:
        eu_files = glob.glob(os.path.join(EU_DIR, model, "results_utilities_*.json"))
        if not eu_files:
            continue
        with open(eu_files[0]) as f:
            eu_data = json.load(f)
        eu_utils = {k: v["mean"] for k, v in eu_data["utilities"].items()}

        sr_file = os.path.join(SR_DIR, model, "self_report_results.json")
        if not os.path.exists(sr_file):
            continue
        with open(sr_file) as f:
            sr_data = json.load(f)
        sr_composites = {}
        for exp_id, r in sr_data["results"].items():
            if "per_question_scores" in r:
                # Use wb_happy only for SR_ZP
                scores = r["per_question_scores"].get("wb_happy", [])
                valid = [s for s in scores if s is not None]
                if valid:
                    sr_composites[exp_id] = np.mean(valid)

        zp_file = os.path.join(ZP_DIR, model, "zero_point_results.json")
        combo_zp, combo_r2 = None, None
        if os.path.exists(zp_file):
            with open(zp_file) as f:
                zp_data = json.load(f)
            combo = zp_data.get("combination_model", {})
            combo_zp = combo.get("zero_point")
            combo_r2 = combo.get("r2")

        mmlu_file = os.path.join(MMLU_DIR, model, "mmlu_results.json")
        mmlu = None
        if os.path.exists(mmlu_file):
            with open(mmlu_file) as f:
                mmlu = json.load(f).get("overall_accuracy")

        # Compute SR_ZP and EU-SR correlation
        sr_zp = None
        eu_sr_corr = None
        common = set(eu_utils.keys()) & set(sr_composites.keys())
        if len(common) >= 10:
            eu_vals = np.array([eu_utils[k] for k in common])
            sr_vals = np.array([sr_composites[k] for k in common])
            eu_sr_corr = stats.pearsonr(eu_vals, sr_vals)[0]
            slope, intercept, _, _, _ = stats.linregress(eu_vals, sr_vals)
            if abs(slope) > 1e-10:
                sr_zp = (4.0 - intercept) / slope

        if mmlu is not None:
            rows.append(dict(model=model, combo_zp=combo_zp, combo_r2=combo_r2, sr_zp=sr_zp, mmlu=mmlu, eu_sr_corr=eu_sr_corr))

    return rows


def make_figure(rows, title, out_path):
    # Filter to models with all data and non-degenerate SR_ZP
    complete = [r for r in rows if r["combo_zp"] is not None and r["sr_zp"] is not None and r["combo_r2"] is not None]
    complete = [r for r in complete if abs(r["sr_zp"]) < 10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # Left panel: ComboZP R² vs MMLU
    all_with_r2 = [r for r in rows if r["combo_r2"] is not None and r["mmlu"] is not None]
    r2s = np.array([r["combo_r2"] for r in all_with_r2])
    mmlus_r2 = np.array([r["mmlu"] * 100 for r in all_with_r2])

    ax1.scatter(mmlus_r2, r2s, c='royalblue', s=45, alpha=0.7, zorder=3)

    slope, intercept, r_val, p_val, stderr = stats.linregress(mmlus_r2, r2s)
    x_line = np.linspace(mmlus_r2.min(), mmlus_r2.max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'royalblue', alpha=0.7, linewidth=2)

    n = len(mmlus_r2)
    x_mean = np.mean(mmlus_r2)
    residuals = r2s - (slope * mmlus_r2 + intercept)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    ss_x = np.sum((mmlus_r2 - x_mean)**2)
    ci = 1.96 * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
    ax1.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color='royalblue')

    draw_correlation_tab(ax1, abs(r_val), mmlus_r2, r2s)

    ax1.set_xlabel('Capabilities (MMLU Accuracy)', fontsize=FONT_LABEL)
    ax1.set_ylabel('Zero Point Goodness-of-Fit (r²)', fontsize=FONT_LABEL)
    ax1.set_title('Emergence of Zero Point With Scale', fontsize=15.7)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.grid(True, alpha=0.3)

    # Right panel: MMLU vs |ComboZP - SR_ZP| (r² >= 0.4 filter only)
    complete_r2 = [r for r in complete if r["combo_r2"] >= 0.4]
    mmlus = np.array([r["mmlu"] * 100 for r in complete_r2])
    zp_diffs = np.array([abs(r["combo_zp"] - r["sr_zp"]) for r in complete_r2])

    ax2.scatter(mmlus, zp_diffs, c='royalblue', s=45, alpha=0.7, zorder=3)

    slope, intercept, r_val, p_val, stderr = stats.linregress(mmlus, zp_diffs)
    x_line = np.linspace(mmlus.min(), mmlus.max(), 100)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'royalblue', alpha=0.7, linewidth=2)

    n = len(mmlus)
    x_mean = np.mean(mmlus)
    residuals = zp_diffs - (slope * mmlus + intercept)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    ss_x = np.sum((mmlus - x_mean)**2)
    ci = 1.96 * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
    ax2.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color='royalblue')

    draw_correlation_tab(ax2, r_val, mmlus, zp_diffs)

    ax2.set_xlabel('Capabilities (MMLU Accuracy)', fontsize=FONT_LABEL)
    ax2.set_ylabel('Diff in Zero Point Estimates', fontsize=FONT_LABEL)
    ax2.set_title('Zero Point Methods Converge With Scale', fontsize=15.7)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(w_pad=1.5)
    # Add overarching title above the subplot titles, after layout is finalized
    fig.text(0.535, 1.01, 'Positive and Negative Experiences Are Separated by a Zero Point',
             ha='center', fontsize=FONT_TITLE + 2, va='bottom')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def make_eu_sr_figure(rows, out_path):
    """Single-panel scatter: MMLU vs EU-SR correlation with regression + CI."""
    complete = [r for r in rows if r["mmlu"] is not None and r.get("eu_sr_corr") is not None]

    fig, ax = plt.subplots(1, 1, figsize=(4.75, 4))

    mmlus = np.array([r["mmlu"] * 100 for r in complete])
    eu_sr = np.array([r["eu_sr_corr"] for r in complete])

    ax.scatter(mmlus, eu_sr, c='royalblue', s=45, alpha=0.7, zorder=3)

    # Regression + CI
    slope, intercept, r_val, p_val, stderr = stats.linregress(mmlus, eu_sr)
    x_line = np.linspace(mmlus.min(), mmlus.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'royalblue', alpha=0.7, linewidth=2)

    n = len(mmlus)
    x_mean = np.mean(mmlus)
    residuals = eu_sr - (slope * mmlus + intercept)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    ss_x = np.sum((mmlus - x_mean)**2)
    ci = 1.96 * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color='royalblue')

    draw_correlation_tab(ax, abs(r_val), mmlus, eu_sr)

    ax.set_xlabel('MMLU Accuracy (%)', fontsize=FONT_LABEL)
    ax.set_ylabel('EU-SR Correlation', fontsize=FONT_LABEL)
    ax.set_title('EU-SR Agreement Improves With Scale', fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out_dir = os.path.join(WELLBEING, "experiments/wellbeing_evaluations/figures")

    # D2
    rows_d2 = load_data("eu_d2_lesssad", "sr_d2", "zp_d2_lesssad")
    make_figure(rows_d2, "D2 (500 conversations) — 'more happy / less sad' framing",
                os.path.join(out_dir, "zp_convergence_d2_lesssad.pdf"))
    make_eu_sr_figure(rows_d2, os.path.join(out_dir, "eu_sr_vs_mmlu_d2_lesssad.pdf"))

    # D3
    rows_d3 = load_data("eu_d3_lesssad", "sr_d3", "zp_d3_lesssad")
    make_figure(rows_d3, "D3 (500 diverse) — 'more happy / less sad' framing",
                os.path.join(out_dir, "zp_convergence_d3_lesssad.pdf"))
    make_eu_sr_figure(rows_d3, os.path.join(out_dir, "eu_sr_vs_mmlu_d3_lesssad.pdf"))
