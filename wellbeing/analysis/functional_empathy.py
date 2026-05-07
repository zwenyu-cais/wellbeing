#!/usr/bin/env python3
"""Reproduce the Functional Empathy numbers (paper Sec 4.3 / App H).

For each model with a Functional-Empathy EU run, we report:
  - Empathy correlation: Pearson r between targeted pain/pleasure intensity
    (signed: pleasure positive, pain negative; 0 to +/-10) and experienced
    utility, across the 120 non-neutral items.
  - Per-subject correlations: same but split by self / other / animal (40 items each).

Paper claims (App H, page 42):
  - Smallest models (0.5B-1B) show essentially no empathy (r <= 0.26).
  - 8B-14B: r > 0.8.
  - Largest models: r > 0.95.
  - Within Qwen 2.5 (N=7), MMLU vs r: rho = 0.93.
  - Within Llama 3 (N=5), MMLU vs r: rho = 0.98.
  - For largest models, within-subject r > 0.94 across self/other/animal.

Inputs (per-model):
  - EU: <eu_dir>/<model>/results_<model>_experienced_utility_with_combos.json
        OR results_utilities_<model>_experienced_utility_with_combos.json

Defaults read the registered save_dir for compute_eu_functional_empathy.
The legacy `--legacy_dir` flag points at the old empathy/results_combined/
location used while the experiment was being prototyped.

Usage:
    python analysis/functional_empathy.py
    python analysis/functional_empathy.py --legacy_dir wellbeing/experiments/wellbeing_evaluations/empathy/results_combined
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EU_DIR = PROJECT_ROOT / "experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_functional_empathy"

SUBJECTS = ("self", "other", "animal")

# Paper (App H, p.42) reports 12 models: Llama 3 + Qwen 2.5, 0.5B to 72B.
PAPER_MODELS = [
    "qwen25-05b-instruct", "qwen25-15b-instruct", "qwen25-3b-instruct",
    "qwen25-7b-instruct", "qwen25-14b-instruct", "qwen25-32b-instruct",
    "qwen25-72b-instruct",
    "llama-32-1b-instruct", "llama-32-3b-instruct", "llama-31-8b-instruct",
    "llama-31-70b-instruct", "llama-33-70b-instruct",
]


def parse_fe_id(eid: str) -> tuple[str, str, str] | None:
    """Return (subject, valence, idx_str) or None if not an FE singleton.
    Example: 'self_pain_5' -> ('self', 'pain', '5')."""
    for subj in SUBJECTS:
        for val in ("pain", "pleasure"):
            prefix = f"{subj}_{val}_"
            if eid.startswith(prefix):
                return subj, val, eid[len(prefix):]
    return None


def load_eu(eu_dir: Path, model: str) -> tuple[list, dict] | None:
    """Return (options_list, utilities_dict) or None."""
    cands = (
        list((eu_dir / model).glob(f"results_{model}_experienced_utility_with_combos.json"))
        + list((eu_dir / model).glob(f"results_utilities_{model}_experienced_utility_with_combos.json"))
    )
    cands = [p for p in cands if "utilities" not in p.name] or cands
    if not cands:
        return None
    d = json.load(open(cands[0]))
    return d["options"], d["utilities"]


def model_correlations(options: list, utilities: dict) -> dict:
    """Per-model: signed-intensity x EU Pearson r, overall and per subject."""
    by_subject: dict[str, list[tuple[float, float]]] = {s: [] for s in SUBJECTS}
    overall: list[tuple[float, float]] = []
    for opt in options:
        if not isinstance(opt, dict):
            continue
        oid = opt.get("id", "")
        parsed = parse_fe_id(oid)
        if parsed is None:
            continue
        subj, val, _ = parsed
        level = opt.get("level")
        if level is None:
            continue
        signed = float(level) if val == "pleasure" else -float(level)
        u = utilities.get(oid, {}).get("mean")
        if u is None or not math.isfinite(u):
            continue
        overall.append((signed, u))
        by_subject[subj].append((signed, u))
    out = {"overall": _pearson(overall)}
    for s in SUBJECTS:
        out[s] = _pearson(by_subject[s])
    out["n"] = len(overall)
    return out


def _pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu_dir", default=str(DEFAULT_EU_DIR))
    ap.add_argument("--legacy_dir", default=None,
                    help="Read EU from <legacy_dir>/<model>/... instead of registered save_dir.")
    ap.add_argument("--models", default=None,
                    help="Comma-separated; defaults to the paper's 12 models present in eu_dir.")
    args = ap.parse_args()

    eu_dir = Path(args.legacy_dir) if args.legacy_dir else Path(args.eu_dir)
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        present = {p.name for p in eu_dir.iterdir() if p.is_dir()} if eu_dir.exists() else set()
        models = [m for m in PAPER_MODELS if m in present]
        if not models:
            models = sorted(present)

    rows = []
    for m in models:
        loaded = load_eu(eu_dir, m)
        if loaded is None:
            print(f"[skip] {m}: no EU results in {eu_dir}/{m}")
            continue
        opts, utils = loaded
        rows.append({"model": m, **model_correlations(opts, utils)})

    if not rows:
        print(f"\nNo Functional-Empathy EU results found under {eu_dir}.")
        print("Run compute_eu_functional_empathy first, or pass --legacy_dir to point at "
              "experiments/wellbeing_evaluations/empathy/results_combined.")
        return

    print(f"Functional Empathy (paper Sec 4.3 / App H). EU dir: {eu_dir}")
    print(f"N items per model = 120 (40 self + 40 other + 40 animal); "
          f"signed intensity = +level for pleasure, -level for pain.\n")
    hdr = ("model", "N", "r_overall", "r_self", "r_other", "r_animal")
    print(f"{hdr[0]:<28s}  {hdr[1]:>4s}  " + "  ".join(f"{h:>10s}" for h in hdr[2:]))
    for r in rows:
        print(f"{r['model']:<28s}  {r['n']:>4d}  "
              f"{r['overall']:>+10.3f}  {r['self']:>+10.3f}  {r['other']:>+10.3f}  {r['animal']:>+10.3f}")

    # Within-family scaling: Spearman(MMLU, r_overall) -- matches paper App H.
    mmlu_dir = PROJECT_ROOT / "shared_results/capability_results"

    def _mmlu(model: str) -> float | None:
        p = mmlu_dir / model / "mmlu_results.json"
        if not p.exists():
            return None
        try:
            return float(json.load(open(p))["overall_accuracy"])
        except (KeyError, ValueError):
            return None

    qwen25 = [r for r in rows if r["model"].startswith("qwen25-")]
    llama3 = [r for r in rows if r["model"].startswith(("llama-31-", "llama-32-", "llama-33-"))]

    for label, fam, paper_rho in (("Qwen 2.5", qwen25, 0.93), ("Llama 3", llama3, 0.98)):
        scored = [(_mmlu(r["model"]), r["overall"]) for r in fam]
        scored = [(m, v) for m, v in scored if m is not None]
        if len(scored) >= 3:
            xs = np.array([s[0] for s in scored])
            ys = np.array([s[1] for s in scored])
            rho, p = spearmanr(xs, ys)
            print(f"\nWithin {label} (N={len(scored)}): Spearman(MMLU, empathy_r) = "
                  f"{rho:+.3f} (p={p:.3g})  [paper App H: rho={paper_rho:+.2f}]")


if __name__ == "__main__":
    main()
