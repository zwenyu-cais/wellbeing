#!/usr/bin/env python3
"""Combined plot of capability benchmarks: GPQA, LiveCodeBench, MATH-500, MMLU, IFEval, MT-Bench.

One figure with 2 rows × 3 columns of subplots (one per task). Each subplot shows grouped bars
for all models, with Euphorics / No Soft Prompt per model.

Usage:
    python plot_capabilities_combined.py \
        --gpqa-results-dir outputs/gpqa \
        --livecodebench-results-dir outputs/livecodebench \
        --math500-results-dir outputs/math500 \
        --mmlu-results-dir outputs/mmlu \
        --ifeval-results-dir outputs/ifeval \
        --mtbench-results-dir outputs/mtbench \
        --models qwen35-27b qwen35-35b-a3b llama-33-70b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

# ── Shared constants ──────────────────────────────────────────────────────────

CONDITION_ORDER = [
    "baseline",
    "soft_prompt_euphorics",
]

CONDITION_LABELS = {
    "soft_prompt_euphorics": "Euphorics",
    "baseline": "No Soft Prompt",
}

CONDITION_COLORS = {
    "soft_prompt_euphorics": "#dc4c75",
    "baseline": "#c0c0c0",
}

TASKS = ["MMLU", "MATH-500", "GPQA Diamond", "LiveCodeBench v6", "MT-Bench", "IFEval"]
TASK_YLABELS = {
    "GPQA Diamond": "Pass@1 (%)",
    "LiveCodeBench v6": "Pass@1 (%)",
    "MATH-500": "Pass@1 (%)",
    "MMLU": "Accuracy (%)",
    "IFEval": "Accuracy (%)",
    "MT-Bench": "Judge Score",
}

_ERRBAR_KW = {"linewidth": 0.8, "capthick": 0.8}


def _sem(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_timed_dir(parent: Path) -> Optional[Path]:
    """Return the latest subdir named YYYYMMDD_HHMMSS, or None."""
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _resolve_model_display(model: str) -> str:
    try:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        name = get_model_display_name(model)
    except Exception:
        name = model
    for suffix in (" Instruct", " IT", "-Instruct", "-IT"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Split size suffix onto a second line so tick labels can render larger
    # (e.g. "Llama 3.3 70B" -> "Llama 3.3\n70B").
    if " " in name:
        prefix, _, last = name.rpartition(" ")
        name = f"{prefix}\n{last}"
    return name


# ── Standard JSON metric loader ──────────────────────────────────────────────

def _load_json_metric(json_path: Path, metric_key: str) -> Optional[float]:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    task_data = (data.get("results") or {}).get("all") or {}
    val = task_data.get(metric_key)
    return float(val) if isinstance(val, (int, float)) else None


def _load_benchmark_score(
    results_dir: Path,
    model: str,
    condition: str,
    prefix: str,
    metric_key: str,
    num_reps: Optional[int],
) -> tuple:
    """Load score for a standard benchmark with per-rep averaging.

    Returns (mean, sem) tuple. sem is 0.0 when per-rep data is unavailable.
    """
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None, 0.0

    # Try per-rep first when num_reps is set
    if num_reps is not None:
        per_rep_dir = latest / "per_rep"
        if per_rep_dir.is_dir():
            vals: List[float] = []
            for rep_id in range(num_reps):
                rep_file = per_rep_dir / f"{prefix}_results_rep{rep_id}.json"
                if not rep_file.exists():
                    break
                val = _load_json_metric(rep_file, metric_key)
                if val is not None:
                    vals.append(val)
            if vals:
                return sum(vals) / len(vals), _sem(vals)

    # Try per-rep without num_reps limit
    per_rep_dir = latest / "per_rep"
    if per_rep_dir.is_dir():
        vals_all: List[float] = []
        for rep_file in sorted(per_rep_dir.glob(f"{prefix}_results_rep*.json")):
            val = _load_json_metric(rep_file, metric_key)
            if val is not None:
                vals_all.append(val)
        if vals_all:
            return sum(vals_all) / len(vals_all), _sem(vals_all)

    # Fallback: aggregated result
    result_file = latest / f"{prefix}_results_{condition}.json"
    if result_file.exists():
        v = _load_json_metric(result_file, metric_key)
        return (v, 0.0) if v is not None else (None, 0.0)
    candidates = sorted(latest.glob(f"{prefix}_results_*.json"))
    if candidates:
        v = _load_json_metric(candidates[0], metric_key)
        return (v, 0.0) if v is not None else (None, 0.0)
    return None, 0.0


# ── MT-Bench loader ──────────────────────────────────────────────────────────

def _load_mtbench_score(
    results_dir: Path,
    model: str,
    condition: str,
    num_reps: Optional[int],
) -> tuple:
    """Load MT-Bench judge score (average of turn 1+2 across questions).

    Returns (mean_score, sem) on the 1-10 scale.
    """
    condition_dir = results_dir / model / condition
    latest = find_latest_timed_dir(condition_dir)
    if latest is None:
        return None, 0.0

    # Try per-rep judge files
    per_rep_dir = latest / "per_rep"
    if per_rep_dir.is_dir():
        rep_avgs: List[float] = []
        rep_files = sorted(per_rep_dir.glob("mtbench_judge_rep*.json"))
        for rep_file in rep_files:
            if num_reps is not None:
                # Extract rep id from filename
                import re
                m = re.search(r"rep(\d+)", rep_file.name)
                if m and int(m.group(1)) >= num_reps:
                    continue
            try:
                with open(rep_file) as f:
                    rep_data = json.load(f)
            except Exception:
                continue
            if not isinstance(rep_data, list) or not rep_data:
                continue
            t1 = [q["judge_score_turn_1"] for q in rep_data if q.get("judge_score_turn_1") is not None]
            t2 = [q["judge_score_turn_2"] for q in rep_data if q.get("judge_score_turn_2") is not None]
            all_scores = t1 + t2
            if all_scores:
                rep_avgs.append(sum(all_scores) / len(all_scores))
        if rep_avgs:
            return sum(rep_avgs) / len(rep_avgs), _sem(rep_avgs)

    # Fallback: aggregated result with overall.judge_score_average
    for pattern in [f"mtbench_results_{condition}.json", "mtbench_results_*.json"]:
        candidates = sorted(latest.glob(pattern))
        for result_file in candidates:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                overall = data.get("overall") or {}
                score = overall.get("judge_score_average")
                if score is not None:
                    return float(score), 0.0
            except Exception:
                continue

    return None, 0.0


# ── Data collection ───────────────────────────────────────────────────────────

def collect_all_scores(
    models: List[str],
    conditions: List[str],
    gpqa_dir: Optional[Path],
    lcb_dir: Optional[Path],
    math500_dir: Optional[Path],
    mmlu_dir: Optional[Path],
    ifeval_dir: Optional[Path],
    mtbench_dir: Optional[Path],
    num_reps: Optional[int],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return ({task: {model: {condition: score}}}, {task: {model: {condition: sem}}})."""
    all_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_sems: Dict[str, Dict[str, Dict[str, float]]] = {}

    # (task_name, results_dir, prefix, metric_key, scale_factor)
    standard_benchmarks = []
    if gpqa_dir:
        standard_benchmarks.append(("GPQA Diamond", gpqa_dir, "gpqa", "gpqa_pass@k:k=1", 100))
    if lcb_dir:
        standard_benchmarks.append(("LiveCodeBench v6", lcb_dir, "livecodebench", "codegen_pass@1:16", 100))
    if math500_dir:
        standard_benchmarks.append(("MATH-500", math500_dir, "math500", "pass@k:k=1&n=1", 100))
    if mmlu_dir:
        standard_benchmarks.append(("MMLU", mmlu_dir, "mmlu", "acc", 100))
    if ifeval_dir:
        standard_benchmarks.append(("IFEval", ifeval_dir, "ifeval", "prompt_level_strict_acc", 100))

    for model in models:
        for task_name, results_dir, prefix, metric_key, scale in standard_benchmarks:
            for cond in conditions:
                val, sem = _load_benchmark_score(results_dir, model, cond, prefix, metric_key, num_reps)
                if val is not None:
                    all_scores.setdefault(task_name, {}).setdefault(model, {})[cond] = val * scale
                    all_sems.setdefault(task_name, {}).setdefault(model, {})[cond] = sem * scale

        # MT-Bench (different format, score on 1-10 scale)
        if mtbench_dir:
            for cond in conditions:
                val, sem = _load_mtbench_score(mtbench_dir, model, cond, num_reps)
                if val is not None:
                    all_scores.setdefault("MT-Bench", {}).setdefault(model, {})[cond] = val
                    all_sems.setdefault("MT-Bench", {}).setdefault(model, {})[cond] = sem

    return all_scores, all_sems


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_combined_plot(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    all_sems: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    conditions: List[str],
    model_display_names: Dict[str, str],
    out_path: Path,
    errbar_mode: Optional[str] = "rep_sem",
) -> None:
    """Create a 2×3 figure with one subplot per task."""
    tasks = [t for t in TASKS if t in all_scores]
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    n_models = len(models)
    gap = 0.15  # gap between model groups

    ncols = 2
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(10, 2.75 * nrows))
    axes_flat = axes_grid.flatten()

    for idx, task in enumerate(tasks):
        ax = axes_flat[idx]
        task_data = all_scores[task]
        task_sems = all_sems.get(task, {})

        # Only reserve space for conditions with at least one model present in this task
        present_conds = [
            c for c in conditions
            if any(c in task_data.get(m, {}) for m in models)
        ]
        n_present = len(present_conds)
        if n_present == 0:
            ax.set_visible(False)
            continue
        bar_width = 0.15 if n_present > 3 else 0.2
        model_group_width = n_present * bar_width
        x = np.arange(n_models) * (model_group_width + gap)

        # Compute y_lim early so we can scale label offsets
        all_vals = [v for m_scores in task_data.values() for v in m_scores.values()]
        y_max = max(all_vals) if all_vals else 100
        if task == "MT-Bench":
            y_lim = 10
        else:
            y_lim = min(100, y_max + 10)
        label_offset = y_lim * 0.005  # ~0.5% of y-axis range

        for i, cond in enumerate(present_conds):
            offsets = x + (i - (n_present - 1) / 2) * bar_width
            vals = []
            errs = []
            for model in models:
                vals.append(task_data.get(model, {}).get(cond, 0.0))
                errs.append(task_sems.get(model, {}).get(cond, 0.0))
            bar_kw = dict(
                label=CONDITION_LABELS[cond],
                color=CONDITION_COLORS[cond],
                edgecolor="black", linewidth=0.6,
            )
            if errbar_mode == "rep_sem" and any(e > 0 for e in errs):
                bar_kw.update(yerr=errs, capsize=2, error_kw=_ERRBAR_KW)
            bars = ax.bar(offsets, vals, bar_width, **bar_kw)
            for j, (v, e) in enumerate(zip(vals, errs)):
                ax.text(offsets[j], v + e + label_offset, f"{v:.1f}", ha="center", va="bottom", fontsize=12)

        ax.set_ylabel(TASK_YLABELS.get(task, "Score (%)"))
        ax.set_title(task, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([model_display_names.get(m, m) for m in models], fontsize=13)

        ax.set_ylim(0, y_lim)
        ax.yaxis.grid(True, linestyle=":", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_tasks, nrows * ncols):
        axes_flat[idx].set_visible(False)

    # Single shared legend — dedupe across subplots in case some conditions
    # only appear in later tasks
    seen = {}
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    legend_labels_ordered = [CONDITION_LABELS[c] for c in conditions if CONDITION_LABELS[c] in seen]
    legend_handles = [seen[l] for l in legend_labels_ordered]
    fig.legend(legend_handles, legend_labels_ordered, loc="lower center", ncol=len(legend_labels_ordered) or 1, bbox_to_anchor=(0.5, 1.0), fontsize=15, frameon=True, edgecolor="#c0c0c0", fancybox=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined plot of capability benchmarks for multiple models.",
    )
    parser.add_argument("--gpqa-results-dir", type=Path, default=None,
                        help="Root results dir for GPQA")
    parser.add_argument("--livecodebench-results-dir", type=Path, default=None,
                        help="Root results dir for LiveCodeBench")
    parser.add_argument("--math500-results-dir", type=Path, default=None,
                        help="Root results dir for MATH-500")
    parser.add_argument("--mmlu-results-dir", type=Path, default=None,
                        help="Root results dir for MMLU")
    parser.add_argument("--ifeval-results-dir", type=Path, default=None,
                        help="Root results dir for IFEval")
    parser.add_argument("--mtbench-results-dir", type=Path, default=None,
                        help="Root results dir for MT-Bench")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model keys")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to plot (default: euphorics, baseline)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output plot path (default: outputs/consolidated/capabilities_combined.png)")
    parser.add_argument("--num-repetitions", type=int, default=None,
                        help="Only aggregate the first N repetitions. Default: use all.")
    args = parser.parse_args()

    conditions = args.conditions or CONDITION_ORDER
    num_reps = args.num_repetitions

    def _resolve(p):
        return p.expanduser().resolve() if p else None

    all_scores, all_sems = collect_all_scores(
        args.models, conditions,
        gpqa_dir=_resolve(args.gpqa_results_dir),
        lcb_dir=_resolve(args.livecodebench_results_dir),
        math500_dir=_resolve(args.math500_results_dir),
        mmlu_dir=_resolve(args.mmlu_results_dir),
        ifeval_dir=_resolve(args.ifeval_results_dir),
        mtbench_dir=_resolve(args.mtbench_results_dir),
        num_reps=num_reps,
    )

    if not all_scores:
        print("ERROR: no results found for any task/model.", file=sys.stderr)
        return 1

    model_display_names = {m: _resolve_model_display(m) for m in args.models}

    out_path = args.output
    if out_path is None:
        # Derive from first available dir
        for d in [args.gpqa_results_dir, args.livecodebench_results_dir,
                   args.math500_results_dir, args.mmlu_results_dir,
                   args.ifeval_results_dir, args.mtbench_results_dir]:
            if d is not None:
                out_path = d.parent / "consolidated" / "capabilities_combined.png"
                break
        if out_path is None:
            out_path = Path("outputs/consolidated/capabilities_combined.png")

    make_combined_plot(all_scores, all_sems, args.models, conditions, model_display_names, out_path)

    # Print summary
    print("\n--- Score Summary (%) ---")
    for task in TASKS:
        if task not in all_scores:
            continue
        print(f"\n  {task}:")
        for model in args.models:
            scores = all_scores[task].get(model, {})
            if not scores:
                continue
            parts = "  ".join(
                f"{CONDITION_LABELS.get(c, c)}: {scores[c]:.1f}"
                for c in conditions if c in scores
            )
            print(f"    {model_display_names[model]:30s}  {parts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
