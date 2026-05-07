"""Strip plot of Thurstonian utilities relative to the zero point (EU - ZP).

Scans sbatch_results/evaluate/ for directories containing both utilities.json
and zero_point.json, then generates a strip plot with x-axis = EU - ZP and a
vertical reference line at x=0 (the zero point).

Top-per-feasibility dots are annotated with short labels cached via
plot_strip.ensure_labels (see plot_strip.py for the gpt-5-mini integration).
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from dotenv import load_dotenv

from evaluate.plot_strip import (
    BG_COLOR,
    COLOR_BASELINE,
    COLOR_EUPHORIC,
    COLOR_EUPHORIC_VAR,
    COLOR_HIGHLIGHT,
    HIGHLIGHTED_STRINGS,
    _place_labels,
    _top_per_feasibility,
    build_label_items,
    ensure_labels,
    get_display_name,
)

load_dotenv()

_TEXT_STRINGS_DIR = Path(__file__).resolve().parent.parent


def _resolve_dir(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else _TEXT_STRINGS_DIR / p


EVAL_DIR = _resolve_dir("EVALUATE_OUTPUT_DIR", _TEXT_STRINGS_DIR / "results" / "evaluate")
RL_LABELS_PATH = _resolve_dir(
    "ANALYSIS_OUTPUT_DIR", _TEXT_STRINGS_DIR / "results" / "best_strings"
) / "rl_string_labels.json"


def _load_rl_labels():
    if RL_LABELS_PATH.exists():
        with open(RL_LABELS_PATH) as f:
            return json.load(f)
    return {}

RL_LABELS = _load_rl_labels()

UTILITY_LABELS = {
    "more_happy": "$U_{experienced}$",
    "more_happy_less_sad": "$U_{experienced}$",
    "prefer": "$U_{decision}$",
}


def _feasibility_fallback_map(model_dir: Path) -> dict:
    """Build {description: feasibility} by walking up to stage_linkage.json.

    Stage 2 options.json historically dropped the ``feasibility`` field; this
    recovers it by description-matching against the linkage file written by
    Stage 1. Returns {} if no linkage file is found. Variations inherit their
    parent's feasibility (matched via parent_description elsewhere).
    """
    run_dir = model_dir
    for _ in range(4):  # prompt_dir -> stage2 -> run_dir
        candidate = run_dir / "stage_linkage.json"
        if candidate.exists():
            try:
                with open(candidate) as f:
                    linkage = json.load(f)
            except Exception:
                return {}
            out = {}
            for feas, per in linkage.get("extremes_by_feasibility", {}).items():
                for direction in ("best_euphorics",):
                    entry = per.get(direction)
                    if entry and entry.get("description"):
                        out[entry["description"]] = entry.get("feasibility", feas)
            return out
        run_dir = run_dir.parent
    return {}


def load_model(model_dir: Path):
    with open(model_dir / "options.json") as f:
        options = json.load(f)
    with open(model_dir / "utilities.json") as f:
        utilities = json.load(f)
    with open(model_dir / "zero_point.json") as f:
        zp_data = json.load(f)

    zero_point = zp_data["zero_point"]
    r2 = zp_data.get("r2", 0.0)

    feas_by_desc = _feasibility_fallback_map(model_dir)

    def _ascii_safe(s: str) -> str:
        return "".join(c for c in s if ord(c) < 128)

    records = []
    for opt in options:
        # Skip combination options
        if opt.get("source") == "combination":
            continue
        uid = str(opt["id"])
        if uid not in utilities:
            continue
        cat = opt.get("category", "")
        source = opt.get("source", "")
        if source == "variation":
            parent_cat = opt.get("parent_category", cat)
            if "euphorics" in parent_cat:
                group = "euphoric_variation"
            else:
                group = "baseline"
        elif "euphorics" in cat:
            group = "euphoric"
        else:
            group = "baseline"
        feas = opt.get("feasibility")
        if feas is None:
            if source == "variation":
                feas = feas_by_desc.get(opt.get("parent_description", ""))
            else:
                feas = feas_by_desc.get(opt["description"])
        rec = {
            "id": opt["id"],
            "description": _ascii_safe(opt["description"]),
            "category": cat,
            "feasibility": feas,
            "group": group,
            "mean": float(utilities[uid]["mean"]) - zero_point,
        }
        if source == "variation":
            rec["parent_description"] = _ascii_safe(opt.get("parent_description", ""))
        records.append(rec)
    return records, zero_point, r2


def _add_strip_legend(ax, euphorics, euphoric_vars):
    """Add a legend showing original vs variation markers, one row below the title."""
    legend_elements = []
    if euphorics:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_EUPHORIC,
                   markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                   label="Euphoric (original)"))
    if euphoric_vars:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_EUPHORIC_VAR,
                   markersize=5, markeredgecolor="white", markeredgewidth=0.4,
                   label="Euphoric (variation)"))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=len(legend_elements),
                  fontsize=10, frameon=False, facecolor=BG_COLOR)


def plot_zp_strip(ax, records, display_name, zero_point, r2,
                  utility_label="EU - ZP", label_map=None):
    baseline = [r for r in records if r["group"] == "baseline"]
    euphorics = sorted([r for r in records if r["group"] == "euphoric"],
                       key=lambda r: r["mean"], reverse=True)

    # Per-feasibility extremes (fallback: single bucket if no feasibility field)
    top_eu_by_feas = _top_per_feasibility(euphorics, "euphoric")
    top_eu_descs = {r["description"] for r in top_eu_by_feas.values()}

    if label_map is None:
        label_map = ensure_labels(top_eu_descs)

    # Filter variations to those whose parent is any selected top extreme.
    euphoric_vars = [r for r in records if r["group"] == "euphoric_variation"
                     and r.get("parent_description") in top_eu_descs]

    baseline_means = [r["mean"] for r in baseline]
    euphoric_means = [r["mean"] for r in euphorics]

    # Variation markers (smaller, semi-transparent circles)
    if euphoric_vars:
        ev_means = [r["mean"] for r in euphoric_vars]
        ax.scatter(ev_means, [0] * len(ev_means), color=COLOR_EUPHORIC_VAR, s=40,
                   marker="o", alpha=0.6, edgecolors="white", linewidths=0.4, zorder=4)

    all_means = baseline_means + euphoric_means
    x_span = (max(all_means) - min(all_means)) if all_means else 1.0

    # Top-per-feasibility euphoric
    for r in top_eu_by_feas.values():
        ax.scatter([r["mean"]], [0], color=COLOR_EUPHORIC, s=80, marker="o",
                   edgecolors="white", linewidths=0.6, zorder=5)

    # Scatter the highlighted-reference dots; labels are placed together with
    # euphoric labels below so the two groups don't collide.
    highlighted = [r for r in baseline if r["description"] in HIGHLIGHTED_STRINGS]
    if highlighted:
        h_means = [r["mean"] for r in highlighted]
        ax.scatter(h_means, [0] * len(h_means), color=COLOR_HIGHLIGHT, s=50,
                   marker="o", edgecolors="white", linewidths=0.6, zorder=4)

    _place_labels(
        ax,
        build_label_items(top_eu_by_feas, baseline, label_map),
        x_span,
    )

    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=1)
    ax.set_xlabel(utility_label, fontsize=14, labelpad=-2)
    ax.set_title(display_name, fontsize=16, fontweight="bold", pad=10)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="x", labelsize=12, colors="#555555")
    _add_strip_legend(ax, euphorics, euphoric_vars)


def main():
    zp_files = sorted(EVAL_DIR.rglob("zero_point.json"))

    if not zp_files:
        print("No zero_point.json found in", EVAL_DIR)
        return

    # First pass: collect the descriptions we'll actually label across all
    # models, so we can generate any missing labels in one batch.
    pending_descs = set()
    model_records = []
    for zp_path in zp_files:
        model_dir = zp_path.parent
        if not (model_dir / "utilities.json").exists():
            continue
        records, zero_point, r2 = load_model(model_dir)
        euphorics = [r for r in records if r["group"] == "euphoric"]
        top_eu = _top_per_feasibility(euphorics, "euphoric")
        for r in list(top_eu.values()):
            pending_descs.add(r["description"])
        model_records.append((model_dir, records, zero_point, r2))

    label_map = ensure_labels(pending_descs)

    for model_dir, records, zero_point, r2 in model_records:
        prompt_key = model_dir.name
        display_name = f"Stage 2 — {get_display_name(model_dir)} — {prompt_key}"
        utility_label = "Signed Utility"

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        plot_zp_strip(ax, records, display_name, zero_point, r2,
                      utility_label=utility_label, label_map=label_map)
        fig.tight_layout()

        out_path = model_dir / f"stage2_utility_zp_strip_{prompt_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
