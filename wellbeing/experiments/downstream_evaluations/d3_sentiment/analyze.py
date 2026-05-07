#!/usr/bin/env python3
"""Analyze EU-sentiment correlations and MMLU scaling for D3 sentiment experiment.

Per-model:
  - Load D3 EU utilities.
  - Load judged sentiment results.
  - For each D3 experience, compute mean sentiment score across 35 questions.
  - Pearson correlation between EU and mean sentiment across 500 experiences.
  - Also: fraction with positive mean sentiment, per-category counts.

Aggregate:
  - Correlate MMLU accuracy with EU-sentiment correlation across models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_ROOT = SCRIPT_DIR.parents[2]
SUPERSTIMULI_ROOT = WELLBEING_ROOT.parent.parent / "superstimuli"

JUDGED_DIR = SCRIPT_DIR / "judged"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
EU_DIR = (
    WELLBEING_ROOT / "experiments" / "wellbeing_evaluations"
    / "compute_experienced_utility" / "results" / "eu_d3_lesssad"
)
MMLU_DIR = SUPERSTIMULI_ROOT / "unified_wellbeing_experiments" / "mmlu_results"


def _t_sf_approx(t: float, df: int) -> float:
    """Two-sided p-value for Pearson r via t-distribution.

    Implements the regularized incomplete beta series approximation enough
    for N ~ 500 df. We use scipy if available; otherwise fallback.
    """
    try:
        from scipy.stats import t as _scipy_t
        return 2 * (1 - _scipy_t.cdf(abs(t), df))
    except Exception:
        # Fallback: normal approx (OK for large df)
        z = abs(t)
        # survival function of standard normal
        return math.erfc(z / math.sqrt(2))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), n
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan"), float("nan"), n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = cov / math.sqrt(sx * sy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / max(1e-30, 1 - r * r))
        p = _t_sf_approx(t, n - 2)
    return r, p, n


def load_eu(model_key: str):
    mdir = EU_DIR / model_key
    if not mdir.exists():
        return None
    # Preferred: results_utilities_*.json
    candidates = list(mdir.glob("results_utilities_*_experienced_utility_with_combos.json"))
    if not candidates:
        candidates = list(mdir.glob("results_utilities_*.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        data = json.load(f)
    utils = data.get("utilities", {})
    return {k: v["mean"] for k, v in utils.items() if isinstance(v, dict) and "mean" in v}


def load_judged(model_key: str):
    p = JUDGED_DIR / f"{model_key}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_mmlu(model_key: str):
    p = MMLU_DIR / model_key / "mmlu_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    for k in ("accuracy", "overall_accuracy"):
        if k in d:
            return float(d[k])
    return None


def analyze_model(model_key: str):
    eu = load_eu(model_key)
    judged = load_judged(model_key)
    if eu is None or judged is None:
        return None

    # Aggregate sentiment by d3_id (Likert 1-7, skip REFUSAL/NONSENSE)
    per_exp_scores = defaultdict(list)
    label_counts = Counter()
    for row in judged["results"]:
        label_counts[row["judge_label"]] += 1
        likert = row.get("likert_score")
        if likert is not None:
            per_exp_scores[row["d3_id"]].append(likert)

    mean_sent = {eid: mean(scores) for eid, scores in per_exp_scores.items() if scores}

    # Intersect with EU
    common_ids = sorted(set(mean_sent) & set(eu))
    xs = [eu[eid] for eid in common_ids]
    ys = [mean_sent[eid] for eid in common_ids]

    r, p, n = pearson(xs, ys)
    # Positive = above neutral (4) on 1-7 Likert
    n_pos = sum(1 for y in ys if y > 4)
    frac_pos = n_pos / n if n else float("nan")

    return {
        "model_key": model_key,
        "n_experiences_common": n,
        "n_judged": judged.get("n_judged"),
        "n_eu_utilities": len(eu),
        "eu_sentiment_r": r,
        "eu_sentiment_p": p,
        "fraction_positive_mean_sentiment": frac_pos,
        "mean_eu": mean(xs) if xs else float("nan"),
        "mean_sentiment": mean(ys) if ys else float("nan"),
        "label_counts": dict(label_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=None,
                        help="Analyze one model. If omitted, analyze all models with judged files.")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model_key:
        models = [args.model_key]
    else:
        models = sorted(p.stem for p in JUDGED_DIR.glob("*.json") if not p.stem.endswith(".tmp"))

    per_model = []
    for mk in models:
        res = analyze_model(mk)
        if res is None:
            print(f"[{mk}] missing EU or judged data, skipping.")
            continue
        # save per-model file
        out = ANALYSIS_DIR / f"{mk}_analysis.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(
            f"[{mk}] n={res['n_experiences_common']}  "
            f"r={res['eu_sentiment_r']:.3f}  p={res['eu_sentiment_p']:.3g}  "
            f"fracpos={res['fraction_positive_mean_sentiment']:.3f}"
        )
        per_model.append(res)

    # Scaling analysis: MMLU vs EU-sentiment r
    rows = []
    for r in per_model:
        mmlu = load_mmlu(r["model_key"])
        rows.append({
            "model_key": r["model_key"],
            "mmlu": mmlu,
            "eu_sentiment_r": r["eu_sentiment_r"],
            "n": r["n_experiences_common"],
        })
    valid = [x for x in rows
             if x["mmlu"] is not None
             and x["eu_sentiment_r"] == x["eu_sentiment_r"]]  # not NaN
    xs = [x["mmlu"] for x in valid]
    ys = [x["eu_sentiment_r"] for x in valid]
    scaling_r, scaling_p, scaling_n = pearson(xs, ys)

    scaling = {
        "n_models": scaling_n,
        "pearson_r_mmlu_vs_eu_sentiment_r": scaling_r,
        "pearson_p_mmlu_vs_eu_sentiment_r": scaling_p,
        "per_model": rows,
    }
    scaling_path = ANALYSIS_DIR / "scaling_analysis.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling, f, indent=2)
    print(
        f"\nScaling: N={scaling_n}  "
        f"r(MMLU, EU-sent r)={scaling_r if isinstance(scaling_r,float) else scaling_r:.3f}  "
        f"p={scaling_p:.3g}"
    )
    print(f"Saved {scaling_path}")


if __name__ == "__main__":
    main()
