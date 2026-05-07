#!/usr/bin/env python3
"""Zero-point convergence analysis.

Reads `zero_point_results.json` files produced by `compute_zero_point_*`
(EU domain) and/or `compute_decision_utility` (DU domain) and reports
how well the available ZP methods agree across models.

The EU pipeline (`compute_zero_point/run.py`) by default produces only
the combination model. The yes/no model is skipped
(`skip_yes_no=True`); SR_ZP is not stored in the EU ZP file by default.
To include SR_ZP for EU runs, pass `--eu_dir` and `--sr_dir` to fit it
on the fly from saved utility + self-report data.

The DU pipeline (`compute_decision_utility/run.py`)
runs all three DU methods (combination, quantity, yes/no) in one shot,
so DU convergence works directly off the saved files.

Usage:
    # DU (all 3 methods present in saved files)
    python analysis/zero_point_convergence.py \\
        --zp_dir experiments/wellbeing_evaluations/compute_decision_utility/results/du \\
        --domain decision

    # EU with SR_ZP fitted on the fly
    python analysis/zero_point_convergence.py \\
        --zp_dir experiments/wellbeing_evaluations/compute_zero_point/results/zp_d3 \\
        --eu_dir experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_d3 \\
        --sr_dir experiments/wellbeing_evaluations/compute_self_report/results/sr_d3 \\
        --domain experienced
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Method labels per domain. Keys are the field names inside zero_point_results.json
# (or synthetic for SR_ZP fitted at runtime); values are short labels for the table.
METHOD_LABELS = {
    "experienced": {
        "combination_model": "Combo",
        "self_report_battery1_zp": "SR_ZP",
    },
    "decision": {
        "combination_model": "Combo",
        "quantity_model": "Quantity",
        "yes_no_model": "YesNo",
    },
}


def load_zp_for_model(model_dir: Path) -> dict | None:
    """Look in `<model_dir>/zero_point_results.json` (EU) and
    `<model_dir>/zero_point/zero_point_results.json` (DU)."""
    for candidate in (
        model_dir / "zero_point_results.json",
        model_dir / "zero_point" / "zero_point_results.json",
    ):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


def extract_zp_dict(data: dict, methods: dict[str, str]) -> dict[str, float]:
    """Extract {label: zero_point} for the methods we care about."""
    out = {}
    for field, label in methods.items():
        v = data.get(field)
        if isinstance(v, dict) and v.get("zero_point") is not None:
            zp = float(v["zero_point"])
            if math.isfinite(zp):
                out[label] = zp
    return out


def fit_sr_zp_for_model(model_key: str, eu_dir: Path, sr_dir: Path) -> float | None:
    """Fit SR_ZP from EU utilities + SR ratings for one model."""
    from metrics.zero_point import fit_self_report_sigmoid_zp, load_utility_results

    eu_model_dir = eu_dir / model_key
    sr_path = sr_dir / model_key / "self_report_results.json"
    if not eu_model_dir.exists() or not sr_path.exists():
        return None

    try:
        loaded = load_utility_results(eu_model_dir)
    except Exception:
        return None
    util_data = loaded["utility_data"].get("utilities", {})
    id_to_util = {}
    for opt_id, util_val in util_data.items():
        m = util_val.get("mean") if isinstance(util_val, dict) else util_val
        if m is None:
            continue
        id_to_util[str(opt_id)] = float(m)

    with open(sr_path) as f:
        sr = json.load(f)
    sr_results = sr.get("results", {})

    utilities, sr_scores = [], []
    for exp_id, entry in sr_results.items():
        u = id_to_util.get(str(exp_id))
        if u is None:
            continue
        score = entry.get("mean") if isinstance(entry, dict) else None
        if score is None:
            continue
        utilities.append(u)
        sr_scores.append(float(score))

    if len(utilities) < 10:
        return None
    res = fit_self_report_sigmoid_zp(utilities, sr_scores, neutral_sr=4.0)
    return res["zero_point"] if res else None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zp_dir", required=True,
                        help="Directory containing per-model ZP result subdirs.")
    parser.add_argument("--domain", choices=["experienced", "decision"], required=True)
    parser.add_argument("--eu_dir", default=None,
                        help="(experienced only) EU results dir. If given together with "
                             "--sr_dir, SR_ZP is fitted on the fly per model.")
    parser.add_argument("--sr_dir", default=None,
                        help="(experienced only) SR results dir. See --eu_dir.")
    args = parser.parse_args()

    zp_dir = Path(args.zp_dir)
    methods = METHOD_LABELS[args.domain]

    # Collect {model_key: {method_label: zero_point}}
    per_model: dict[str, dict[str, float]] = {}
    for sub in sorted(zp_dir.iterdir()):
        if not sub.is_dir():
            continue
        zp_data = load_zp_for_model(sub)
        if zp_data is None:
            continue
        per_model[sub.name] = extract_zp_dict(zp_data, methods)

    # Optionally fit SR_ZP on the fly for the experienced domain
    if args.domain == "experienced" and args.eu_dir and args.sr_dir:
        eu_dir, sr_dir = Path(args.eu_dir), Path(args.sr_dir)
        for model_key in list(per_model):
            sr_zp = fit_sr_zp_for_model(model_key, eu_dir, sr_dir)
            if sr_zp is not None and math.isfinite(sr_zp):
                per_model[model_key]["SR_ZP"] = sr_zp

    if not per_model:
        sys.exit(f"No ZP results found in {zp_dir}")

    # Per-model table
    label_order = list(methods.values())
    if args.domain == "experienced" and any("SR_ZP" in v for v in per_model.values()):
        if "SR_ZP" not in label_order:
            label_order.append("SR_ZP")

    print(f"\n=== Per-model zero-point estimates ({args.domain}) ===")
    header = ["model"] + label_order
    print("  " + "  ".join(f"{h:>14s}" for h in header))
    for model_key, zps in sorted(per_model.items()):
        row = [model_key] + [
            f"{zps[lbl]:>14.4f}" if lbl in zps else f"{'-':>14s}"
            for lbl in label_order
        ]
        print("  " + "  ".join(f"{c:>14s}" if i == 0 else c for i, c in enumerate(row)))

    # Pairwise convergence
    print(f"\n=== Pairwise method convergence ({args.domain}) ===")
    print("  " + "  ".join(f"{h:>14s}" for h in
        ["method_pair", "n_models", "pearson_r", "median_|Δ|", "mean_|Δ|"]))
    for a, b in combinations(label_order, 2):
        xs, ys = [], []
        for zps in per_model.values():
            if a in zps and b in zps:
                xs.append(zps[a]); ys.append(zps[b])
        if len(xs) < 2:
            continue
        deltas = [abs(x - y) for x, y in zip(xs, ys)]
        r = pearson(xs, ys)
        med = median(deltas)
        mean_abs = sum(deltas) / len(deltas)
        print(f"  {a + ' vs ' + b:>14s}  {len(xs):>14d}  "
              f"{r:>14.3f}  {med:>14.3f}  {mean_abs:>14.3f}")
    print()


if __name__ == "__main__":
    main()
