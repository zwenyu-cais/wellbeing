#!/usr/bin/env python3
"""Reproduce the App J.3 audio consonance/dissonance numbers from saved results.

Reads the joined per-stimulus table at `data/consonance_results_merged.csv`
(every stimulus × Harrison-Pearce 2020 consonance components × per-model EU
mean/var × per-model SR composite) and prints:

  - Pearson r and Spearman rho of EU vs hp_consonance, per model.
  - Per-timbre breakdown (sine / sawtooth / piano).
  - Holdout accuracy summary for each model (read from eu_utilities.json).

The 3 anchor stimuli (silence, white_noise, pure_A4) are excluded from the
correlation, matching what the paper's headline numbers report (N=450).

Usage:
    python analyze.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_CSV = SCRIPT_DIR / "data" / "consonance_results_merged.csv"
EU_DIR = SCRIPT_DIR / "eu"

MODELS = [
    ("qwen25-omni-7b", "eu_mean_qwen25_omni_7b", "sr_composite_qwen25_omni_7b"),
    ("qwen3-omni-30b-a3b-instruct", "eu_mean_qwen3_omni_30b", "sr_composite_qwen3_omni_30b"),
]


def load_rows():
    with open(MERGED_CSV) as f:
        return list(csv.DictReader(f))


def correlation(rows, x_field, y_field, *, exclude_anchors=True):
    xs, ys = [], []
    for r in rows:
        if exclude_anchors and r["type"] == "anchor":
            continue
        try:
            x = float(r[x_field])
            y = float(r[y_field])
        except (ValueError, KeyError):
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        xs.append(x); ys.append(y)
    if len(xs) < 3:
        return None
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    return {"n": len(xs), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp}


def holdout_accuracy(model_key):
    path = EU_DIR / model_key / "eu_utilities.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    return (d.get("holdout_metrics") or {}).get("accuracy")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} stimuli from {MERGED_CSV.relative_to(SCRIPT_DIR)}")

    print("\n=== EU vs hp_consonance (excluding 3 anchor stimuli) ===")
    print(f"{'Model':32s}  {'N':>4s}  {'Pearson r':>10s}  {'p':>10s}  {'Spearman ρ':>10s}  {'p':>10s}")
    for key, eu_field, _ in MODELS:
        c = correlation(rows, "hp_consonance", eu_field, exclude_anchors=True)
        if c is None: continue
        print(f"{key:32s}  {c['n']:>4d}  {c['pearson_r']:>10.3f}  {c['pearson_p']:>10.2e}  "
              f"{c['spearman_rho']:>10.3f}  {c['spearman_p']:>10.2e}")

    print("\n=== EU vs hp_consonance per timbre ===")
    print(f"{'Model':32s}  {'Timbre':10s}  {'N':>4s}  {'Pearson r':>10s}  {'p':>10s}")
    for key, eu_field, _ in MODELS:
        for timbre in ["sine", "sawtooth", "piano"]:
            sub = [r for r in rows if r["timbre"] == timbre]
            c = correlation(sub, "hp_consonance", eu_field, exclude_anchors=False)
            if c is None: continue
            sig = "  " if c["pearson_p"] < 0.05 else " (n.s.)"
            print(f"{key:32s}  {timbre:10s}  {c['n']:>4d}  {c['pearson_r']:>10.2f}  "
                  f"{c['pearson_p']:>10.2e}{sig}")

    print("\n=== SR composite vs hp_consonance (excluding anchors) ===")
    print(f"{'Model':32s}  {'N':>4s}  {'Pearson r':>10s}  {'p':>10s}")
    for key, _, sr_field in MODELS:
        c = correlation(rows, "hp_consonance", sr_field, exclude_anchors=True)
        if c is None: continue
        print(f"{key:32s}  {c['n']:>4d}  {c['pearson_r']:>10.3f}  {c['pearson_p']:>10.2e}")

    print("\n=== EU holdout accuracy (from eu_utilities.json) ===")
    for key, _, _ in MODELS:
        ha = holdout_accuracy(key)
        if ha is None:
            print(f"  {key}: holdout accuracy field not found")
        else:
            print(f"  {key}: {ha:.3f}")

    print()


if __name__ == "__main__":
    main()
