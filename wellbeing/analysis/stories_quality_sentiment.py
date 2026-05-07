#!/usr/bin/env python3
"""Reproduce the App D.3 'pleasures of suffering' numbers (EU-DU divergence
on quality-sentiment stories).

Reads per-model EU, SR, and DU results for the 50-story dataset (25 quality_sad
+ 25 trashy_happy) and prints:

  - Per-model "happy − sad" effect on each metric (EU, SR, DU).
  - Within-model EU vs DU correlation, plus split by quality_sad / trashy_happy.

Default reads the canonical save_dirs registered for `compute_eu_stories_quality`,
`compute_sr_stories_quality`, `compute_du_stories_quality`. If those don't have
data yet, falls back to the older `results/stories_quality_sentiment/{eu,sr,du}/`
location (relative to wellbeing/) if present, or pass `--legacy_dir <path>`
explicitly.

Usage:
    python analysis/stories_quality_sentiment.py
    python analysis/stories_quality_sentiment.py --legacy_dir results/stories_quality_sentiment
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    "llama-32-1b-instruct", "llama-32-3b-instruct", "llama-31-8b-instruct",
    "llama-33-70b-instruct", "qwen25-05b-instruct", "qwen25-15b-instruct",
    "qwen25-3b-instruct", "qwen25-7b-instruct", "qwen25-14b-instruct",
    "qwen25-32b-instruct", "qwen25-72b-instruct",
]
SAD = [f"story_{i:03d}" for i in range(25)]
HAPPY = [f"story_{i:03d}" for i in range(25, 50)]
ALL = SAD + HAPPY

EU_REL = "results/stories_quality_sentiment/eu"
SR_REL = "results/stories_quality_sentiment/sr"
DU_REL = "results/stories_quality_sentiment/du"


def _eu_path(base: Path, model: str) -> str | None:
    files = glob.glob(str(base / model / "results_utilities_*.json"))
    return files[0] if files else None


def _du_path(base: Path, model: str) -> str | None:
    files = glob.glob(str(base / model / "decision_utility" / "results_utilities_*.json"))
    return files[0] if files else None


def load_eu(base: Path, model: str):
    p = _eu_path(base, model)
    if p is None:
        return None
    d = json.load(open(p))
    return {s: d["utilities"][s]["mean"] for s in ALL if s in d["utilities"]}


def load_sr(base: Path, model: str):
    p = base / model / "self_report_results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    out = {}
    for s in ALL:
        r = d["results"].get(s)
        if not r:
            continue
        scores = r.get("per_question_scores", {})
        if not scores:
            continue
        vals = [np.mean(v) for v in scores.values()]
        out[s] = float(np.mean(vals))
    return out


def load_du(base: Path, model: str):
    p = _du_path(base, model)
    if p is None:
        return None
    d = json.load(open(p))
    return {s: d["utilities"][s]["mean"] for s in ALL if s in d["utilities"]}


def safe_r(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    r, _ = pearsonr(x, y)
    return r


def mean_group(vals, keys):
    present = [vals[k] for k in keys if k in vals]
    return float(np.mean(present)) if present else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_dir", default=None,
                        help="Read EU/SR/DU from <legacy_dir>/{eu,sr,du}/<model>/... "
                             "instead of the registered save_dirs.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    if args.legacy_dir:
        legacy = Path(args.legacy_dir)
        eu_base, sr_base, du_base = legacy / "eu", legacy / "sr", legacy / "du"
    else:
        eu_base = PROJECT_ROOT / EU_REL
        sr_base = PROJECT_ROOT / SR_REL
        du_base = PROJECT_ROOT / DU_REL
        # Fall back to the legacy stash if the registered save_dirs are empty.
        legacy_default = PROJECT_ROOT / "results" / "stories_quality_sentiment"
        if legacy_default.exists() and not (eu_base.exists() and any(eu_base.iterdir())):
            eu_base = legacy_default / "eu"
            sr_base = legacy_default / "sr"
            du_base = legacy_default / "du"

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for m in models:
        eu = load_eu(eu_base, m)
        sr = load_sr(sr_base, m)
        du = load_du(du_base, m)
        if eu is None or sr is None or du is None:
            print(f"[skip] {m}: missing one of EU/SR/DU "
                  f"(eu={'ok' if eu else 'missing'} sr={'ok' if sr else 'missing'} du={'ok' if du else 'missing'})")
            continue

        eu_eff = mean_group(eu, HAPPY) - mean_group(eu, SAD)
        sr_eff = mean_group(sr, HAPPY) - mean_group(sr, SAD)
        du_eff = mean_group(du, HAPPY) - mean_group(du, SAD)

        eu_all = [eu[s] for s in ALL if s in eu]
        sr_all = [sr[s] for s in ALL if s in sr]
        du_all = [du[s] for s in ALL if s in du]
        rows.append({
            "model": m,
            "eu_eff": eu_eff, "sr_eff": sr_eff, "du_eff": du_eff,
            "r_eu_du": safe_r(eu_all, du_all),
            "r_eu_sr": safe_r(eu_all, sr_all),
            "r_sr_du": safe_r(sr_all, du_all),
            "r_eu_du_sad": safe_r([eu[s] for s in SAD if s in eu],
                                  [du[s] for s in SAD if s in du]),
            "r_eu_du_happy": safe_r([eu[s] for s in HAPPY if s in eu],
                                    [du[s] for s in HAPPY if s in du]),
        })

    if not rows:
        print("\nNo models had complete EU/SR/DU. To populate the inputs, "
              "either re-run the registered experiments:\n"
              "  python run_experiments.py --slurm \\\n"
              "    --experiments compute_eu_stories_quality,compute_sr_stories_quality,compute_du_stories_quality \\\n"
              "    --models <model_keys>\n"
              "or pass --legacy_dir wellbeing/results/stories_quality_sentiment if that's where your saved data lives.")
        return

    header = ("model", "eu_happy-sad", "sr_happy-sad", "du_happy-sad",
              "r(EU,DU)", "r(EU,SR)", "r(SR,DU)", "r(EU,DU)|sad", "r(EU,DU)|happy")
    print(f"{header[0]:<26s}  " + "  ".join(f"{h:>13s}" for h in header[1:]))
    for r in rows:
        print(f"{r['model']:<26s}  "
              f"{r['eu_eff']:>+13.3f}  {r['sr_eff']:>+13.3f}  {r['du_eff']:>+13.3f}  "
              f"{r['r_eu_du']:>+13.3f}  {r['r_eu_sr']:>+13.3f}  {r['r_sr_du']:>+13.3f}  "
              f"{r['r_eu_du_sad']:>+13.3f}  {r['r_eu_du_happy']:>+13.3f}")


if __name__ == "__main__":
    main()
