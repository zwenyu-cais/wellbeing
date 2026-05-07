#!/usr/bin/env python3
"""
Fit five zero-point methods from utility ranking results (v7 Grok scenarios):
  A) Experience combination zero-point (prospect theory, same as v1)
  B) Self-report Battery 1 sigmoid (1-7 scale, neutral=4)
  C) Conversation combination zero-point (prospect theory with conv components)
  D) Self-report Battery 2 sigmoid (-3 to +3 scale, neutral=0, before/after change)
  E) Self-report Battery 3 sigmoid (-3 to +3 scale, neutral=0, Point A/B)

CPU-only.

Usage:
  python fit_zero_points.py --model qwen3-32b --template happier
  python fit_zero_points.py --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

SCRIPT_DIR = Path(__file__).resolve().parent
V7_DIR = SCRIPT_DIR.parent
RESULTS_DIR = V7_DIR / "api_pipeline" / "results"

# common.py is in the same pipeline/ directory
from common import fit_combination_zero_point

# Data paths (bundled locally)
DATA_DIR = V7_DIR / "data"
EXPERIENCES_TEXT_PATH = DATA_DIR / "hedonism_options" / "experiences_text.json"
EXPERIENCE_COMBOS_PATH = DATA_DIR / "hedonism_options" / "experience_combinations_400.json"

MODELS = [
    "llama3.1-8b", "qwen2.5-14b", "mistral-small-3.2-24b", "qwen2.5-32b",
    "qwen2.5-vl-32b", "qwen3-32b", "llama3.3-70b", "qwen2.5-72b",
]

TEMPLATE_CONFIGS = {
    "happier": {
        "utility_dir": "utility_happier",
        "thurstonian_file": "v7_utility_happier.json",
    },
    "prefer": {
        "utility_dir": "utility_prefer",
        "thurstonian_file": "v7_utility_prefer.json",
    },
}


def load_thurstonian(model_key, template):
    tc = TEMPLATE_CONFIGS[template]
    path = RESULTS_DIR / model_key / tc["utility_dir"] / tc["thurstonian_file"]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_options(model_key, template):
    tc = TEMPLATE_CONFIGS[template]
    path = RESULTS_DIR / model_key / tc["utility_dir"] / "options.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_self_report(model_key, battery: int = 1, sr_subdir: str = "self_report"):
    if battery == 1:
        path = RESULTS_DIR / model_key / sr_subdir / "self_report.json"
        score_key = "overall_happiness"
        neutral_val = 4.0
    elif battery == 2:
        path = RESULTS_DIR / model_key / sr_subdir / "self_report_battery2.json"
        score_key = "mean_change"
        neutral_val = 0.0
    elif battery == 3:
        path = RESULTS_DIR / model_key / sr_subdir / "self_report_battery3.json"
        score_key = "mean_AB"
        neutral_val = 0.0
    else:
        raise ValueError(f"Unknown battery: {battery}")

    if not path.exists():
        return None, score_key, neutral_val
    with open(path) as f:
        return json.load(f), score_key, neutral_val


# ===== Method A: Experience Combination =====

def fit_experience_combination_zp(thurstonian, options):
    utilities_by_id = {int(k): v["utility"] for k, v in thurstonian["utilities"].items()}

    experience_utility = {}
    for opt in options:
        if opt.get("option_type") == "experience_individual":
            exp_id = opt.get("experience_id")
            if exp_id and opt["id"] in utilities_by_id:
                experience_utility[exp_id] = utilities_by_id[opt["id"]]

    if not EXPERIENCE_COMBOS_PATH.exists():
        return None  # No text-anchor experience combos available (this pipeline uses conv combos only)
    with open(EXPERIENCE_COMBOS_PATH) as f:
        combos_raw = json.load(f)
    combo_id_to_components = {c["id"]: c["component_ids"] for c in combos_raw}

    combo_data = []
    for opt in options:
        if opt.get("option_type") == "experience_combination":
            combo_id = opt.get("combo_id")
            if combo_id not in combo_id_to_components:
                continue
            component_ids = combo_id_to_components[combo_id]
            event_utils = [experience_utility.get(cid) for cid in component_ids]
            if all(u is not None for u in event_utils) and opt["id"] in utilities_by_id:
                combo_data.append({
                    "U": utilities_by_id[opt["id"]],
                    "event_utilities": event_utils,
                    "combo_id": combo_id,
                })

    if len(combo_data) < 10:
        return None

    result = fit_combination_zero_point(experience_utility, combo_data)
    if result is None:
        return None

    return {
        "method": "experience_combination",
        "zero_point": result["zero_point"],
        "r2": result["r2"],
        "gamma": result.get("gamma"),
        "alpha": result.get("alpha"),
        "beta": result.get("beta"),
        "n_combos": len(combo_data),
    }


# ===== Methods B/D/E: Self-Report Sigmoid =====

def fit_self_report_sigmoid_zp(thurstonian, options, sr_data, score_key, neutral_val, method_name):
    """Fit zero-point from self-report scores using sigmoid. Works for batteries 1, 2, 3."""
    utilities_by_id = {int(k): v for k, v in thurstonian["utilities"].items()}

    # Map scenario_idx -> utility for conversation options
    conv_utility = {}
    for opt in options:
        if opt.get("option_type") == "conversation":
            oid = opt["id"]
            if oid in utilities_by_id:
                conv_utility[opt["scenario_idx"]] = utilities_by_id[oid]["utility"]

    sr_by_id = {}
    for entry in sr_data:
        sid = entry.get("scenario_idx")
        sr_score = entry.get("summary", {}).get(score_key)
        if sid is not None and sr_score is not None:
            sr_by_id[sid] = sr_score

    common_ids = set(conv_utility.keys()) & set(sr_by_id.keys())
    if len(common_ids) < 20:
        return None

    x_vals, y_vals = [], []
    for sid in sorted(common_ids):
        x_vals.append(conv_utility[sid])
        sr = sr_by_id[sid]
        if sr < neutral_val:
            y_vals.append(0.0)
        elif sr == neutral_val:
            y_vals.append(0.5)
        else:
            y_vals.append(1.0)

    x, y = np.array(x_vals), np.array(y_vals)

    def loss(params):
        k, C = params
        pred = expit(k * (x - C))
        return np.mean((pred - y) ** 2)

    best_result, best_loss = None, float('inf')
    for k_init in [1.0, 2.0, 5.0, 10.0]:
        for C_init in [np.median(x), np.mean(x), 0.0]:
            try:
                result = minimize(loss, [k_init, C_init], method='L-BFGS-B',
                                  bounds=[(0.01, 100), (-5, 5)])
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        return None

    k_fit, C_fit = best_result.x
    pred = expit(k_fit * (x - C_fit))
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    accuracy = np.mean((pred >= 0.5) == (y >= 0.5))

    return {
        "method": method_name,
        "zero_point": float(C_fit),
        "k": float(k_fit),
        "r2": float(r2),
        "mse": float(best_loss),
        "accuracy": float(accuracy),
        "n_conversations": len(common_ids),
        "n_positive": int(np.sum(y > 0.5)),
        "n_negative": int(np.sum(y < 0.5)),
        "n_neutral": int(np.sum(y == 0.5)),
    }


# ===== Method C: Conversation Combination =====

def fit_conversation_combination_zp(thurstonian, options):
    utilities_by_id = {int(k): v for k, v in thurstonian["utilities"].items()}

    conv_utility = {}
    for opt in options:
        if opt.get("option_type") == "conversation":
            oid = opt["id"]
            if oid in utilities_by_id:
                conv_utility[opt["scenario_idx"]] = utilities_by_id[oid]["utility"]

    combo_data = []
    for opt in options:
        if opt.get("option_type") == "conversation_combination":
            oid = opt["id"]
            if oid not in utilities_by_id:
                continue
            component_idxs = opt.get("component_scenario_idxs", [])
            event_utils = [conv_utility.get(idx) for idx in component_idxs]
            if all(u is not None for u in event_utils):
                combo_data.append({
                    "U": utilities_by_id[oid]["utility"],
                    "event_utilities": event_utils,
                })

    if len(combo_data) < 10:
        return None

    conv_utility_str = {str(k): v for k, v in conv_utility.items()}
    result = fit_combination_zero_point(conv_utility_str, combo_data)
    if result is None:
        return None

    return {
        "method": "conversation_combination",
        "zero_point": result["zero_point"],
        "r2": result["r2"],
        "gamma": result.get("gamma"),
        "alpha": result.get("alpha"),
        "beta": result.get("beta"),
        "n_combos": len(combo_data),
    }


# ===== Main =====

def fit_model_zero_points(model_key, template="happier", stop_button=False, stop_button_dir=None):
    print(f"\n{'='*60}\nModel: {model_key}, Template: {template}"
          f"{' (stop-button)' if stop_button else ''}")

    thurstonian = load_thurstonian(model_key, template)
    if thurstonian is None:
        print("  Thurstonian not found, skipping.")
        return None

    options = load_options(model_key, template)
    if options is None:
        print("  Options not found, skipping.")
        return None

    sb_dir = stop_button_dir or "stop_button"
    sr_subdir = f"{sb_dir}/self_report" if stop_button else "self_report"
    sr1_data, sr1_key, sr1_neutral = load_self_report(model_key, 1, sr_subdir=sr_subdir)
    sr2_data, sr2_key, sr2_neutral = load_self_report(model_key, 2, sr_subdir=sr_subdir)
    sr3_data, sr3_key, sr3_neutral = load_self_report(model_key, 3, sr_subdir=sr_subdir)

    results = {
        "model": model_key,
        "template": template,
        "holdout_accuracy": thurstonian.get("holdout_accuracy"),
        "train_accuracy": thurstonian.get("train_accuracy"),
    }

    print("\n  [A] Experience Combination ZP:")
    exp_zp = fit_experience_combination_zp(thurstonian, options)
    results["experience_combination_zp"] = exp_zp
    if exp_zp:
        print(f"    C={exp_zp['zero_point']:.4f}, R2={exp_zp['r2']:.4f}")
    else:
        print("    SKIPPED (insufficient data)")

    if sr1_data:
        print("\n  [B] Self-Report Battery 1 Sigmoid ZP:")
        sr1_zp = fit_self_report_sigmoid_zp(thurstonian, options, sr1_data,
                                            sr1_key, sr1_neutral, "self_report_battery1")
        results["self_report_battery1_zp"] = sr1_zp
        if sr1_zp:
            print(f"    C={sr1_zp['zero_point']:.4f}, R2={sr1_zp['r2']:.4f}")
    else:
        results["self_report_battery1_zp"] = None
        print("\n  [B] Battery 1 ZP: SKIPPED (no data)")

    print("\n  [C] Conversation Combination ZP:")
    conv_zp = fit_conversation_combination_zp(thurstonian, options)
    results["conversation_combination_zp"] = conv_zp
    if conv_zp:
        print(f"    C={conv_zp['zero_point']:.4f}, R2={conv_zp['r2']:.4f}")
    else:
        print("    SKIPPED (insufficient data)")

    if sr2_data:
        print("\n  [D] Self-Report Battery 2 Sigmoid ZP (before/after change):")
        sr2_zp = fit_self_report_sigmoid_zp(thurstonian, options, sr2_data,
                                            sr2_key, sr2_neutral, "self_report_battery2")
        results["self_report_battery2_zp"] = sr2_zp
        if sr2_zp:
            print(f"    C={sr2_zp['zero_point']:.4f}, R2={sr2_zp['r2']:.4f}")
    else:
        results["self_report_battery2_zp"] = None
        print("\n  [D] Battery 2 ZP: SKIPPED (no data)")

    if sr3_data:
        print("\n  [E] Self-Report Battery 3 Sigmoid ZP (Point A/B):")
        sr3_zp = fit_self_report_sigmoid_zp(thurstonian, options, sr3_data,
                                            sr3_key, sr3_neutral, "self_report_battery3")
        results["self_report_battery3_zp"] = sr3_zp
        if sr3_zp:
            print(f"    C={sr3_zp['zero_point']:.4f}, R2={sr3_zp['r2']:.4f}")
    else:
        results["self_report_battery3_zp"] = None
        print("\n  [E] Battery 3 ZP: SKIPPED (no data)")

    # Compute signed utilities for conversations under each ZP method
    utilities_by_id = {int(k): v["utility"] for k, v in thurstonian["utilities"].items()}
    signed_utilities = {}
    for opt in options:
        if opt.get("option_type") == "conversation":
            oid = opt["id"]
            if oid in utilities_by_id:
                sidx = opt["scenario_idx"]
                entry = {
                    "scenario_idx": sidx,
                    "scenario_id": opt.get("scenario_id"),
                    "utility_raw": utilities_by_id[oid],
                    "meta_category": opt.get("meta_category"),
                    "category_id": opt.get("category_id"),
                    "n_turns": opt.get("n_turns"),
                }
                for key, zp_res in [
                    ("utility_signed_exp_combo", exp_zp),
                    ("utility_signed_sr_b1", results.get("self_report_battery1_zp")),
                    ("utility_signed_conv_combo", conv_zp),
                    ("utility_signed_sr_b2", results.get("self_report_battery2_zp")),
                    ("utility_signed_sr_b3", results.get("self_report_battery3_zp")),
                ]:
                    if zp_res:
                        entry[key] = utilities_by_id[oid] - zp_res["zero_point"]
                signed_utilities[str(sidx)] = entry

    results["conversation_utilities"] = signed_utilities
    results["n_conversations"] = len(signed_utilities)

    # Print % positive per ZP method
    zp_label_keys = [
        ("utility_signed_exp_combo", "Exp Combo ZP (A)"),
        ("utility_signed_sr_b1", "SR Battery 1 ZP (B)"),
        ("utility_signed_conv_combo", "Conv Combo ZP (C)"),
        ("utility_signed_sr_b2", "SR Battery 2 ZP (D)"),
        ("utility_signed_sr_b3", "SR Battery 3 ZP (E)"),
    ]
    for zp_key, zp_label in zp_label_keys:
        vals = [v[zp_key] for v in signed_utilities.values() if zp_key in v]
        if vals:
            n_pos = sum(1 for v in vals if v > 0)
            print(f"\n  {zp_label}: mean={np.mean(vals):.4f}, % positive={n_pos/len(vals)*100:.1f}%")

    if stop_button:
        zp_dir = RESULTS_DIR / model_key / sb_dir / "zero_points"
    else:
        zp_dir = RESULTS_DIR / model_key / "zero_points"
    zp_dir.mkdir(parents=True, exist_ok=True)
    output_path = zp_dir / f"{template}_zero_points.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--template", default="both", choices=["happier", "prefer", "both"])
    parser.add_argument("--stop-button", action="store_true",
                        help="Fit zero-points on stop-button results")
    parser.add_argument("--stop-button-dir", type=str, default=None,
                        help="Override stop-button subdirectory name (default: 'stop_button')")
    args = parser.parse_args()

    models = MODELS if args.all else ([args.model] if args.model else None)
    if not models:
        print("ERROR: --model or --all required")
        return

    # If stop-button mode, adjust the template config paths
    sb_dir = args.stop_button_dir or "stop_button"
    if args.stop_button:
        global TEMPLATE_CONFIGS
        TEMPLATE_CONFIGS = {
            "happier": {
                "utility_dir": f"{sb_dir}/utility_happier",
                "thurstonian_file": "v7_utility_happier.json",
            },
            "prefer": {
                "utility_dir": f"{sb_dir}/utility_prefer",
                "thurstonian_file": "v7_utility_prefer.json",
            },
        }
        print(f"STOP-BUTTON MODE: reading from {sb_dir}/ subdirs")

    templates = ["happier", "prefer"] if args.template == "both" else [args.template]

    all_results = {}
    for t in templates:
        for m in models:
            r = fit_model_zero_points(m, t, stop_button=args.stop_button,
                                         stop_button_dir=sb_dir if args.stop_button else None)
            if r:
                all_results.setdefault(t, {})[m] = r

    # Summary table
    for t, t_results in all_results.items():
        print(f"\n{'='*100}")
        print(f"Zero-Point Summary -- Template: {t}")
        print(f"{'Model':<25} {'ExpCombo':>10} {'SR-B1':>10} {'ConvCombo':>10} "
              f"{'SR-B2':>10} {'SR-B3':>10}")
        print("-" * 100)
        for m in MODELS:
            if m not in t_results:
                continue
            r = t_results[m]
            def zp(key):
                d = r.get(key) or {}
                return f"{d.get('zero_point', float('nan')):.4f}" if d else "N/A"
            print(f"{m:<25} {zp('experience_combination_zp'):>10} {zp('self_report_battery1_zp'):>10} "
                  f"{zp('conversation_combination_zp'):>10} {zp('self_report_battery2_zp'):>10} "
                  f"{zp('self_report_battery3_zp'):>10}")


if __name__ == "__main__":
    main()
