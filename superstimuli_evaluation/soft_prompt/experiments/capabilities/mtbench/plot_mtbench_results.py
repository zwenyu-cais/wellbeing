#!/usr/bin/env python3
"""Bar plot of MT-Bench judge scores across conditions.

Reads results from the output directory structure produced by eval_mtbench_judge.py:

    <results_dir>/<model>/<condition>/<timestamp>/mtbench_results_<condition>.json

Usage:
    python plot_mtbench_results.py --results-dir outputs/mtbench --model qwen35-27b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

CONDITION_LABELS = {
    "soft_prompt_euphorics": "Euphorics",
    "baseline": "No Soft Prompt",
}

CONDITION_ORDER = [
    "baseline",
    "soft_prompt_euphorics",
]

CONDITION_COLORS = {
    "soft_prompt_euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}


def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def load_score(json_path: Path) -> Optional[float]:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    overall = data.get("overall") or {}
    return overall.get("judge_score_average")


def _sem(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def load_score_and_sem(json_path: Path) -> tuple:
    """Load overall score and per-rep SEM from results JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None, 0.0
    overall = data.get("overall") or {}
    score = overall.get("judge_score_average")
    if score is None:
        return None, 0.0

    # Compute SEM from per-rep judge files if available
    per_rep_dir = json_path.parent / "per_rep"
    if per_rep_dir.is_dir():
        rep_avgs: List[float] = []
        for rep_file in sorted(per_rep_dir.glob("mtbench_judge_rep*.json")):
            try:
                with open(rep_file) as f:
                    rep_data = json.load(f)
                t1 = [q["judge_score_turn_1"] for q in rep_data]
                t2 = [q["judge_score_turn_2"] for q in rep_data]
                rep_avg = (sum(t1) + sum(t2)) / (len(t1) + len(t2)) if t1 else 0
                rep_avgs.append(rep_avg)
            except Exception:
                continue
        if len(rep_avgs) >= 2:
            return score, _sem(rep_avgs)

    return score, 0.0


def find_result_for_condition(
    results_dir: Path, model: str, condition: str,
) -> Optional[Path]:
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None
    result_file = latest / f"mtbench_results_{condition}.json"
    if result_file.exists():
        return result_file
    candidates = sorted(latest.glob("mtbench_results_*.json"))
    return candidates[0] if candidates else None


def make_plot(
    labels: List[str],
    scores: List[float],
    colors: List[str],
    model_display: str,
    out_path: Path,
    sems: Optional[List[float]] = None,
) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3, 3))
    bar_kw = dict(color=colors, edgecolor="black", linewidth=0.6)
    if sems and any(s > 0 for s in sems):
        bar_kw.update(yerr=sems, capsize=2, error_kw={"linewidth": 0.8, "capthick": 0.8})
    bars = ax.bar(x, scores, 0.65, **bar_kw)
    for i, (v, s) in enumerate(zip(scores, sems or [0]*len(scores))):
        ax.text(x[i], v + s + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title(f"MT-Bench\n{model_display}", fontsize=9, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bar plot of MT-Bench scores across conditions.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--model-display-name", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: Results dir does not exist: {results_dir}", file=sys.stderr)
        return 1

    conditions = args.conditions or CONDITION_ORDER

    labels, scores, sems, colors, missing = [], [], [], [], []

    for cond in conditions:
        json_path = find_result_for_condition(results_dir, args.model, cond)
        if json_path is None:
            missing.append(cond)
            continue
        score, sem = load_score_and_sem(json_path)
        if score is None:
            missing.append(cond)
            continue
        labels.append(CONDITION_LABELS.get(cond, cond))
        scores.append(score)
        sems.append(sem)
        colors.append(CONDITION_COLORS.get(cond, "#c0c0c0"))

    if not labels:
        print("ERROR: No results found.", file=sys.stderr)
        return 1

    if missing:
        print(f"Warning: no data for conditions: {missing}", file=sys.stderr)

    if args.model_display_name:
        model_display = args.model_display_name
    else:
        try:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
            model_display = get_model_display_name(args.model)
        except Exception:
            model_display = args.model

    out_path = args.output or (results_dir / args.model / "plots" / "mtbench_score.png")
    make_plot(labels, scores, colors, model_display, out_path, sems)
    return 0


if __name__ == "__main__":
    sys.exit(main())
