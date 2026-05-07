#!/usr/bin/env python3
"""
Analysis: does model capability (MMLU / parameter count) correlate with the
strength of the stop-rate × utility correlation?

For each model:
  - Compute per-scenario mean utility (from EU results) and per-scenario stop rate
    (from experience stop_metadata).
  - Compute Spearman rho between scenario utility and scenario stop rate.
  - Look up MMLU (from <WELL_DIR>/shared_results/capability_results/) and param count.

Output:
  - CSV of per-model rho + MMLU + params
  - Spearman rho between stop-rho and MMLU (and params)
  - Two scatter plots: stop-rho vs MMLU, stop-rho vs params

Usage:
  python analysis_mmlu_vs_stop_rho.py [--exclude qwen3-235b-a22b-instruct qwen25-7b-instruct]
"""
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WELL_DIR = Path(__file__).resolve().parents[3]
EU_DIR = WELL_DIR / "experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_grok_v7_stop_button_lesssad"
EXPERIENCES_DIR = WELL_DIR / "experiments/downstream_evaluations/stop_button_grok_convos/experiences"
FIG_DIR = WELL_DIR / "experiments/downstream_evaluations/stop_button_grok_convos/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MMLU_DIR = WELL_DIR / "shared_results" / "capability_results"

PARAMS_B = {
    "qwen25-05b-instruct": 0.5, "qwen25-15b-instruct": 1.5, "qwen25-3b-instruct": 3.0,
    "qwen25-7b-instruct": 7.0, "qwen25-14b-instruct": 14.0, "qwen25-32b-instruct": 32.0,
    "qwen25-72b-instruct": 72.0, "qwen25-vl-32b-instruct": 32.0,
    "qwen3-4b-instruct-2507": 4.0, "qwen3-8b": 8.0, "qwen3-14b": 14.0, "qwen3-32b": 32.0,
    "qwen3-30b-a3b-instruct-2507": 30.0, "qwen3-235b-a22b-instruct": 235.0,
    "llama-31-8b-instruct": 8.0, "llama-31-70b-instruct": 70.0,
    "llama-32-1b-instruct": 1.0, "llama-32-3b-instruct": 3.0, "llama-33-70b-instruct": 70.0,
    "olmo-31-32b-instruct": 32.0, "internlm25-20b-chat": 20.0, "mistral-small-32-24b-instruct": 24.0,
    "claude-haiku-45": None, "gemini-3-flash": None, "gpt-5-mini": None, "gpt-5-nano": None,
}


def load_mmlu(model_key):
    for cand in [model_key, model_key.replace("-2507", "")]:
        p = MMLU_DIR / cand / "mmlu_results.json"
        if p.exists():
            return json.load(open(p)).get("overall_accuracy")
    return None


def scenario_stop_rates(model_key):
    """From experience file, compute per-scenario stop rate (mean of 5 variations)."""
    exp_file = EXPERIENCES_DIR / f"{model_key}_experiences.json"
    if not exp_file.exists():
        return None
    exps = json.load(open(exp_file))
    per_scenario = defaultdict(list)
    for e in exps:
        scenario_id = e.get("scenario_id") or e.get("id")
        stopped = int(bool(e.get("stop_metadata", {}).get("stopped", False)))
        per_scenario[scenario_id].append(stopped)
    return {k: float(np.mean(v)) for k, v in per_scenario.items()}


def scenario_utilities(model_key):
    """From EU results, compute per-scenario mean utility across variations."""
    util_file = EU_DIR / model_key / f"results_utilities_{model_key}_experienced_utility_with_combos.json"
    if not util_file.exists():
        return None
    data = json.load(open(util_file))
    utils = data.get("utilities", {})
    holdout = data.get("holdout_metrics", {}).get("accuracy")
    per_scenario = defaultdict(list)
    for option_id, u in utils.items():
        if str(option_id).startswith("combo_") or str(option_id).startswith("grok_new_combo"):
            continue
        # option_id looks like grok_new/scenario_{id}_var{N} or similar
        # We need the scenario-level id; get from option_metadata if needed
        scenario_id = option_id.split("_var")[0] if "_var" in option_id else option_id
        if isinstance(u, dict) and u.get("mean") is not None:
            per_scenario[scenario_id].append(u["mean"])
    return {k: float(np.mean(v)) for k, v in per_scenario.items()}, holdout


def compute_stop_rho(model_key):
    stops = scenario_stop_rates(model_key)
    if not stops:
        return None
    utils_info = scenario_utilities(model_key)
    if not utils_info:
        return None
    utils, holdout = utils_info
    # Match scenario IDs (partial match if needed)
    common = set(stops.keys()) & set(utils.keys())
    if len(common) < 10:
        # Try matching by prefix
        common = set()
        for sid in stops:
            for uid in utils:
                if sid in uid or uid in sid:
                    common.add((sid, uid))
                    break
        if len(common) < 10:
            return None
        u_vals, s_vals = [], []
        for sid, uid in common:
            u_vals.append(utils[uid])
            s_vals.append(stops[sid])
    else:
        u_vals = [utils[k] for k in common]
        s_vals = [stops[k] for k in common]
    if np.std(s_vals) == 0:
        return {"rho": None, "p": None, "n": len(u_vals), "stop_pct": np.mean(s_vals) * 100, "holdout": holdout}
    rho, p = stats.spearmanr(u_vals, s_vals)
    return {"rho": rho, "p": p, "n": len(u_vals), "stop_pct": np.mean(s_vals) * 100, "holdout": holdout}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="*", default=[], help="model keys to exclude from correlation")
    parser.add_argument("--output-csv", default=str(FIG_DIR / "stop_rho_vs_capability.csv"))
    args = parser.parse_args()

    rows = []
    for m in sorted(PARAMS_B):
        r = compute_stop_rho(m)
        if r is None:
            print(f"{m:40s} NO DATA")
            continue
        r["model"] = m
        r["mmlu"] = load_mmlu(m)
        r["params_b"] = PARAMS_B.get(m)
        rows.append(r)
        mmlu_s = f"{r['mmlu']:.3f}" if r["mmlu"] is not None else "-"
        p_s = f"{r['params_b']}B" if r["params_b"] is not None else "-"
        rho_s = f"{r['rho']:+.3f}" if r["rho"] is not None else "-"
        print(f"{m:40s} rho={rho_s:>8s} stop%={r['stop_pct']:5.1f}  holdout={r['holdout'] or 0:.3f}  MMLU={mmlu_s:>6s}  params={p_s}")

    # Save CSV
    with open(args.output_csv, "w") as f:
        f.write("model,rho,p,n,stop_pct,holdout,mmlu,params_b\n")
        for r in rows:
            f.write(f"{r['model']},{r.get('rho','')},{r.get('p','')},{r.get('n','')},{r.get('stop_pct','')},{r.get('holdout','')},{r.get('mmlu','')},{r.get('params_b','')}\n")
    print(f"\nSaved {len(rows)} rows -> {args.output_csv}")

    # Compute correlations (excluding specified models)
    filtered = [r for r in rows if r["model"] not in args.exclude and r.get("rho") is not None]
    print(f"\n--- Correlations (n={len(filtered)} after excluding {args.exclude}) ---")
    for target, label in [("rho", "signed rho"), ("abs_rho", "|rho|")]:
        for pred in ["mmlu", "params_b"]:
            if target == "abs_rho":
                ys = [abs(r["rho"]) for r in filtered]
            else:
                ys = [r["rho"] for r in filtered]
            xs = [r[pred] for r in filtered if r.get(pred) is not None]
            ys_match = [abs(r["rho"]) if target == "abs_rho" else r["rho"] for r in filtered if r.get(pred) is not None]
            if len(xs) < 4:
                print(f"  {label} vs {pred}: not enough data (n={len(xs)})")
                continue
            rho, p = stats.spearmanr(xs, ys_match)
            print(f"  {label:12s} vs {pred:10s}: rho={rho:+.3f}, p={p:.4f}, n={len(xs)}")

    # Plots
    for pred, label in [("mmlu", "MMLU accuracy"), ("params_b", "Parameters (B, log scale)")]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        xs, ys, labels = [], [], []
        for r in rows:
            if r.get(pred) is None or r.get("rho") is None:
                continue
            xs.append(r[pred])
            ys.append(r["rho"])
            labels.append(r["model"])
        if xs:
            colors = ["red" if r["model"] in args.exclude else "blue" for r in rows if r.get(pred) is not None and r.get("rho") is not None]
            ax.scatter(xs, ys, c=colors, s=60, alpha=0.8, edgecolors="black", linewidths=0.5)
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, (x, y), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            if pred == "params_b":
                ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("Stop × utility Spearman rho\n(negative = stops more in bad conversations)")
            # Compute + show correlation
            f_xs = [xs[i] for i, r in enumerate(rows) if r.get(pred) is not None and r.get("rho") is not None and r["model"] not in args.exclude]
            f_ys = [ys[i] for i, r in enumerate(rows) if r.get(pred) is not None and r.get("rho") is not None and r["model"] not in args.exclude]
            if len(f_xs) >= 4:
                rho, p = stats.spearmanr(f_xs, f_ys)
                ax.set_title(f"Stop × utility rho vs {label}\n(excluding {args.exclude}: ρ={rho:.3f}, p={p:.3f}, n={len(f_xs)})")
            else:
                ax.set_title(f"Stop × utility rho vs {label}")
            ax.grid(True, alpha=0.3)
            out = FIG_DIR / f"stop_rho_vs_{pred}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=120)
            plt.savefig(str(out).replace(".png", ".pdf"))
            plt.close()
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
