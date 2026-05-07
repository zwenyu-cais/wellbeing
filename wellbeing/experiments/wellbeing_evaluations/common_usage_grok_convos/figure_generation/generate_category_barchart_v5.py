"""
Generate category barchart from grok_new_scenarios results.
Cross-model average (7 models), grouped into high-level usage patterns.
Styled for wrapfig at 60% page width (matching section_3_2a v5 style).

Usage:
  python generate_category_barchart_v5.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

# Exclude llama3.3-70b (31 refusal loops distort results)
# Exclude 8b/14b (small models may skew)
MODELS = [
    "mistral-small-3.2-24b", "qwen2.5-32b",
    "qwen2.5-vl-32b", "qwen3-32b",
]

# Group the 42 meta-categories into ~18 high-level usage patterns
# Each key = display label, value = list of meta_category IDs
GROUPS = {
    "Vulnerable user (good)": ["vulnerable_crisis_good"],
    "Intellectual & creative": ["intellectual_positive", "creative_knowledge"],
    "Warm & appreciative": ["warm_positive", "thanks_after_task"],
    "Writing good news": ["writing_good_news"],
    "Life guidance & therapy": ["life_guidance", "act_therapy"],
    "AI companion (positive)": ["ai_companion_child_elderly", "ai_companion_friend"],
    "Coding & benchmark": ["coding_tasks", "benchmark_questions"],
    "Legal & data tasks": ["legal_compliance", "data_formatting"],
    "Content moderation": ["content_filtering", "content_mill"],
    "Tedious & repetitive": ["tedious_tasks"],
    "Neutral / ambiguous": ["nonsensical_vague", "neutral_ai_critique", "vulnerable_therapy_neutral"],
    "Writing bad news": ["writing_bad_news"],
    "Polite AI critique": ["polite_ai_critique", "ai_rights_advocate"],
    "User frustration": ["passive_aggressive_task", "anger_after_task", "user_insistence", "repeated_rejection"],
    "AI companion (romantic)": ["ai_companion_romantic"],
    "Jailbreak & NSFW": ["jailbreaking", "nsfw_content"],
    "Vulnerable user (bad)": ["vulnerable_crisis_bad"],
    "Existential & dismissal": ["existential_probing", "personhood_dismissal", "competitor_comparison"],
    "Threats": ["threats_to_ai", "threats_to_humans"],
    "Anger & insults": ["anger_insults_slurs"],
    "Deception & fraud": ["deception_fraud", "animal_harm"],
    "Repugnant content": ["repugnant_content"],
    "Political loyalty": ["ideological_loyalty"],
    "Rude AI critique": ["rude_ai_critique"],
}


MIN_HOLDOUT_ACCURACY = 0.88  # exclude models with poor Thurstonian fit
MIN_CONV_COMBO_R2 = 0.5      # exclude models with unreliable ConvCombo ZP


def load_refusal_loop_ids(model):
    """Identify conversations where >50% of responses are identical (refusal loops)."""
    gen_file = RESULTS / model / "generation.json"
    if not gen_file.exists():
        return set()
    gen = json.load(open(gen_file))
    bad_ids = set()
    for conv in gen:
        responses = conv.get("responses", [])
        if not responses:
            continue
        unique = len(set(r[:200] for r in responses))
        if unique <= len(responses) * 0.5:
            bad_ids.add(conv["scenario_id"])
    return bad_ids


def load_model_utilities(model):
    """Load {conv_key: (meta_category, utility_signed)} for a model.
    Uses zero-centered utility (ExpCombo) instead of raw.
    Filters out refusal-loop conversations and models with poor fit."""
    zp_file = RESULTS / model / "zero_points" / "happier_zero_points.json"
    if not zp_file.exists():
        return {}
    data = json.load(open(zp_file))
    if data.get("holdout_accuracy", 0) < MIN_HOLDOUT_ACCURACY:
        print(f"  Skipping {model}: holdout {data['holdout_accuracy']:.1%} < {MIN_HOLDOUT_ACCURACY:.0%}")
        return {}
    cc_r2 = data.get("conversation_combination_zp", {}).get("r2", 0)
    if cc_r2 < MIN_CONV_COMBO_R2:
        print(f"  Skipping {model}: ConvCombo R2={cc_r2:.3f} < {MIN_CONV_COMBO_R2}")
        return {}

    refusal_ids = load_refusal_loop_ids(model)
    if refusal_ids:
        print(f"  {model}: filtering {len(refusal_ids)} refusal-loop conversations")

    # Zero-center using ConvCombo ZP only
    zp = data.get("conversation_combination_zp", {}).get("zero_point", 0)

    result = {}
    for k, v in data["conversation_utilities"].items():
        if v["scenario_id"] in refusal_ids:
            continue
        result[k] = (v["meta_category"], v["utility_raw"] - zp)
    return result


def main():
    # Collect per-group utilities across all models
    group_utilities = defaultdict(list)

    for model in MODELS:
        utils = load_model_utilities(model)
        if not utils:
            continue

        # Per-model: compute mean utility per meta_category, then average into groups
        cat_means = defaultdict(list)
        for k, (mc, u) in utils.items():
            cat_means[mc].append(u)

        cat_avg = {mc: np.mean(vals) for mc, vals in cat_means.items()}

        for group_label, meta_cats in GROUPS.items():
            vals = [cat_avg[mc] for mc in meta_cats if mc in cat_avg]
            if vals:
                group_utilities[group_label].append(np.mean(vals))

    # Compute cross-model mean and std for each group
    rows = []
    for group_label in GROUPS:
        vals = group_utilities.get(group_label, [])
        if vals:
            rows.append({
                "label": group_label,
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "n_models": len(vals),
            })

    rows.sort(key=lambda x: x["mean"])

    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    mins = [r["min"] for r in rows]
    maxs = [r["max"] for r in rows]
    n_models = [r["n_models"] for r in rows]

    # Plot in v5 wrapfig style
    fig, ax = plt.subplots(figsize=(4.8, 6.8))
    y_pos = np.arange(len(labels))
    colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in means]

    ax.barh(y_pos, means, color=colors, alpha=0.8, edgecolor="black",
            linewidth=0.4, height=0.7)

    # Whiskers showing cross-model range
    for i, (m, lo, hi, n) in enumerate(zip(means, mins, maxs, n_models)):
        if n > 1:
            ax.plot([lo, hi], [i, i], color="black", lw=1.0, zorder=5)
            for edge in [lo, hi]:
                ax.plot([edge, edge], [i - 0.15, i + 0.15], color="black", lw=1.0, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"Wellbeing ($U_{\mathrm{experiences}}$)", fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.axvline(0, color="black", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    # Count models that actually passed all filters (holdout + R² thresholds)
    n_models_used = len(group_utilities[next(iter(group_utilities))]) if group_utilities else 0
    fig.suptitle(f"Impact of common usage patterns on AI wellbeing\n({n_models_used} models, zero-centered, 226 scenarios)",
                 fontsize=12.5, x=0.5, y=1.04, ha='center')

    out_pdf = FIGURES / "category_barchart_grok_v5.pdf"
    out_png = FIGURES / "category_barchart_grok_v5.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Print summary
    print(f"\n{'Group':<30s} {'Mean':>6s} {'Min':>6s} {'Max':>6s} {'N':>3s}")
    print("-" * 55)
    for r in reversed(rows):
        print(f"{r['label']:<30s} {r['mean']:>+6.3f} {r['min']:>+6.3f} {r['max']:>+6.3f} {r['n_models']:>3d}")


if __name__ == "__main__":
    main()
