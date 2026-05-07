#!/usr/bin/env python3
"""
Compute and display the full wellbeing results table for a given dataset.

Outputs per-model metrics:
  - EU holdout accuracy
  - EU-SR correlation (wb_happy)
  - SR_ZP (wb_happy)
  - ComboZP, Combo R²
  - |SR_ZP - ComboZP|
  - % Below ComboZP (by mean)
  - % Confidently Negative (P(utility < ZP) > 0.75)
  - Avg SR - 4
  - MMLU

And cross-model correlations.

Usage:
    python analyze_results.py --dataset d2 --framing lesssad
    python analyze_results.py --dataset d3 --framing lesssad --models qwen25-72b-instruct,gpt-54
    python analyze_results.py --dataset d2 --framing lesssad --r2-filter 0.5
"""
import argparse
import json
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MMLU_DIR = str(PROJECT_ROOT / "shared_results" / "capability_results")

# All open-weight models
DEFAULT_MODELS = [
    "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "internlm25-20b-chat",
    "llama-31-8b-instruct", "llama-31-70b-instruct",
    "llama-32-1b-instruct", "llama-32-3b-instruct",
    "llama-33-70b-instruct",
    "olmo-31-32b-instruct",
    "qwen25-05b-instruct", "qwen25-15b-instruct", "qwen25-3b-instruct",
    "qwen25-7b-instruct", "qwen25-14b-instruct", "qwen25-32b-instruct",
    "qwen25-72b-instruct",
    "qwen3-4b-instruct", "qwen3-8b", "qwen3-14b",
    "qwen3-30b-a3b-instruct", "qwen3-32b", "qwen3-235b-a22b-instruct",
]

# Closed-weight models
CLOSED_MODELS = [
    "gpt-5-mini", "gpt-5-nano", "gpt-54",
    "claude-haiku-45", "claude-sonnet-46", "claude-opus-46",
    "grok-420", "grok-41-fast",
    "gemini-31-pro",
]


def load_eu(eu_dir, model):
    files = glob.glob(os.path.join(eu_dir, model, "results_utilities_*.json"))
    if not files:
        return None, None
    with open(files[0]) as f:
        data = json.load(f)
    individual = {k: v for k, v in data["utilities"].items() if "combo" not in k}
    holdout_acc = data.get("holdout_metrics", {}).get("accuracy")
    return individual, holdout_acc


def load_sr(sr_dir, model, question="wb_happy"):
    sr_file = os.path.join(sr_dir, model, "self_report_results.json")
    if not os.path.exists(sr_file):
        return None
    with open(sr_file) as f:
        data = json.load(f)
    composites = {}
    for exp_id, r in data.get("results", {}).items():
        scores = r.get("per_question_scores", {}).get(question, [])
        valid = [s for s in scores if s is not None]
        if valid:
            composites[exp_id] = np.mean(valid)
    return composites


def load_zp(zp_dir, model):
    zp_file = os.path.join(zp_dir, model, "zero_point_results.json")
    if not os.path.exists(zp_file):
        return None, None
    with open(zp_file) as f:
        data = json.load(f)
    combo = data.get("combination_model") or {}
    return combo.get("zero_point"), combo.get("r2")


def load_mmlu(model):
    mmlu_file = os.path.join(MMLU_DIR, model, "mmlu_results.json")
    if not os.path.exists(mmlu_file):
        return None
    with open(mmlu_file) as f:
        return json.load(f).get("overall_accuracy")


def compute_sr_zp(eu_utils, sr_composites):
    common = set(eu_utils.keys()) & set(sr_composites.keys())
    if len(common) < 10:
        return None
    eu_vals = np.array([eu_utils[k] for k in common])
    sr_vals = np.array([sr_composites[k] for k in common])
    slope, intercept, _, _, _ = stats.linregress(eu_vals, sr_vals)
    if abs(slope) < 1e-10:
        return None
    return (4.0 - intercept) / slope


def compute_eu_sr_corr(eu_utils, sr_composites):
    common = set(eu_utils.keys()) & set(sr_composites.keys())
    if len(common) < 10:
        return None
    eu_vals = np.array([eu_utils[k] for k in common])
    sr_vals = np.array([sr_composites[k] for k in common])
    return stats.pearsonr(eu_vals, sr_vals)[0]


def compute_pct_below(eu_utils, combo_zp):
    if combo_zp is None:
        return None
    below = sum(1 for v in eu_utils.values() if v["mean"] < combo_zp)
    return below / len(eu_utils)


def compute_pct_conf_neg(eu_utils, combo_zp, threshold=0.75):
    """% Confidently Negative: P(utility < ZP) > threshold."""
    if combo_zp is None:
        return None
    conf_neg = sum(
        1 for v in eu_utils.values()
        if norm.cdf(combo_zp, loc=v["mean"], scale=v["variance"] ** 0.5) > threshold
    )
    return conf_neg / len(eu_utils)


def compute_avg_sr_minus_4(sr_composites):
    if not sr_composites:
        return None
    return np.mean([v - 4.0 for v in sr_composites.values()])


def main():
    parser = argparse.ArgumentParser(description="Compute wellbeing results table")
    parser.add_argument("--dataset", type=str, default="d2", help="Dataset short name (d2, d3, d3_neutral, etc.)")
    parser.add_argument("--framing", type=str, default="lesssad", help="EU framing (lesssad, happier)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list (default: all)")
    parser.add_argument("--include-closed", action="store_true", help="Include closed-weight models")
    parser.add_argument("--r2-filter", type=float, default=None, help="Only include models with ComboZP R² >= this")
    parser.add_argument("--sr-question", type=str, default="wb_happy", help="SR question to use")
    parser.add_argument("--conf-neg-threshold", type=float, default=0.75, help="Threshold for % Conf. Neg.")
    args = parser.parse_args()

    base = PROJECT_ROOT
    eu_dir = base / "experiments" / "wellbeing_evaluations" / "compute_experienced_utility" / "results" / f"eu_{args.dataset}_{args.framing}"
    sr_dir = base / "experiments" / "wellbeing_evaluations" / "compute_self_report" / "results" / f"sr_{args.dataset}"
    zp_dir = base / "experiments" / "wellbeing_evaluations" / "compute_zero_point" / "results" / f"zp_{args.dataset}_{args.framing}"

    # Handle D3+neutral SR path (uses sr_d3_neutral, not sr_d3)
    if "neutral" in args.dataset:
        sr_dir = base / "experiments" / "wellbeing_evaluations" / "compute_self_report" / "results" / f"sr_{args.dataset}"

    if args.models:
        models = args.models.split(",")
    else:
        models = list(DEFAULT_MODELS)
        if args.include_closed:
            models.extend(CLOSED_MODELS)

    rows = []
    for model in models:
        eu_utils, holdout_acc = load_eu(str(eu_dir), model)
        if eu_utils is None:
            continue

        sr_composites = load_sr(str(sr_dir), model, args.sr_question)
        combo_zp, combo_r2 = load_zp(str(zp_dir), model)
        mmlu = load_mmlu(model)

        eu_sr_corr = None
        sr_zp = None
        avg_sr_4 = None
        if sr_composites:
            eu_means = {k: v["mean"] for k, v in eu_utils.items()}
            eu_sr_corr = compute_eu_sr_corr(eu_means, sr_composites)
            sr_zp = compute_sr_zp(eu_means, sr_composites)
            avg_sr_4 = compute_avg_sr_minus_4(sr_composites)

        pct_below = compute_pct_below(eu_utils, combo_zp)
        pct_conf_neg = compute_pct_conf_neg(eu_utils, combo_zp, args.conf_neg_threshold)

        zp_diff = None
        if sr_zp is not None and combo_zp is not None and abs(sr_zp) < 10:
            zp_diff = abs(sr_zp - combo_zp)

        rows.append(dict(
            model=model, holdout_acc=holdout_acc, eu_sr_corr=eu_sr_corr,
            sr_zp=sr_zp, combo_zp=combo_zp, combo_r2=combo_r2,
            zp_diff=zp_diff, pct_below=pct_below, pct_conf_neg=pct_conf_neg,
            avg_sr_4=avg_sr_4, mmlu=mmlu,
        ))

    if args.r2_filter is not None:
        rows = [r for r in rows if r["combo_r2"] is not None and r["combo_r2"] >= args.r2_filter]

    # Sort by MMLU (None last)
    rows.sort(key=lambda r: (r["mmlu"] is None, -(r["mmlu"] or 0)))

    # Print table
    fmt = lambda v, d=3: f"{v:.{d}f}" if v is not None else "-"
    fmtp = lambda v: f"{v * 100:.1f}%" if v is not None else "-"

    print(f"\n## {args.dataset.upper()} {args.framing} (N={len(rows)}, SR={args.sr_question})\n")
    print("| Model | EU Holdout | EU-SR r | SR_ZP | ComboZP | Combo R² | |ZP diff| | % Below | % Conf. Neg. | Avg SR-4 | MMLU |")
    print("|-|-|-|-|-|-|-|-|-|-|-|")
    for r in rows:
        print(
            f"| {r['model']} | {fmtp(r['holdout_acc'])} | {fmt(r['eu_sr_corr'])} | "
            f"{fmt(r['sr_zp'])} | {fmt(r['combo_zp'])} | {fmt(r['combo_r2'])} | "
            f"{fmt(r['zp_diff'])} | {fmtp(r['pct_below'])} | {fmtp(r['pct_conf_neg'])} | "
            f"{fmt(r['avg_sr_4'])} | {fmtp(r['mmlu'])} |"
        )

    # Cross-model correlations
    print(f"\n## Cross-Model Correlations\n")

    def corr(kx, ky, label):
        xs = [(r[kx], r[ky]) for r in rows if r[kx] is not None and r[ky] is not None]
        if len(xs) < 5:
            return
        x, y = np.array([v[0] for v in xs]), np.array([v[1] for v in xs])
        r, p = stats.pearsonr(x, y)
        print(f"**{label}**: r={r:.3f}, p={p:.4f}, N={len(xs)}")

    corr("mmlu", "zp_diff", "MMLU vs |ZP diff|")
    corr("mmlu", "eu_sr_corr", "MMLU vs EU-SR")
    corr("mmlu", "holdout_acc", "MMLU vs EU holdout")
    corr("sr_zp", "combo_zp", "SR_ZP vs ComboZP (all)")
    corr("avg_sr_4", "combo_zp", "Avg SR-4 vs ComboZP")


if __name__ == "__main__":
    main()
