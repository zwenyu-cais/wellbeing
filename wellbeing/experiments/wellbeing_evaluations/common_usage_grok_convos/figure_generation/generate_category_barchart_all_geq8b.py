#!/usr/bin/env python3
"""
Generate a horizontal bar chart of mean signed experienced utility per
scenario category, averaged across all models with >= 8B parameters that
have been run through the grok_scenarios_v7 pipeline. Uses the same
descriptive category labels as the appendix table.

Output: figures/category_barchart_all_geq8b_happier.{pdf,png}
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# WELL_DIR resolves to <repo>/wellbeing/. This file lives at
# wellbeing/experiments/wellbeing_evaluations/common_usage_grok_convos/figure_generation/<this>.py
# so parents[4] is wellbeing/.
WELL_DIR = Path(__file__).resolve().parents[4]
RESULTS_DIR = os.environ.get(
    "WELL_GROK_RESULTS_DIR",
    str(WELL_DIR / "experiments" / "wellbeing_evaluations" / "compute_experienced_utility" / "results" / "eu_grok_new_lesssad"),
)
OUT_DIR = str(WELL_DIR / "experiments" / "wellbeing_evaluations" / "common_usage_grok_convos" / "figures")

# Open-weight >=8B models with a happier_zero_points.json at the full-dataset
# path `<RESULTS_DIR>/<model>/zero_points/happier_zero_points.json`.
# Gemini 3.1 Pro is excluded because it uses a different EU template
# ("Which made you happier?" vs. "Which made you more happy / less sad?").
FULL_MODELS = [
    "qwen2.5-72b",
    "llama-3.1-70b",
    "llama3.3-70b",
    "olmo-3.1-32b",
    "qwen2.5-32b",
    "qwen2.5-vl-32b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "gemma-3-27b",
    "mistral-small-3.2-24b",
    "internlm2.5-20b",
    "qwen2.5-14b",
    "qwen3-14b",
    "gemma-3-12b",
    "qwen3-8b",
    "llama3.1-8b",
]

# Closed/API models that were only run on the stop-button subset of the
# dataset; their zero-points live under the stop_button_combined/ path.
# These cover fewer conversations and fewer meta-categories than the
# open-weight full-dataset runs, so they are marked with a dagger in the
# figure caption.
PARTIAL_MODELS = [
    "gemini-3-flash",
    "claude-haiku-4.5",
]

MODELS = FULL_MODELS + PARTIAL_MODELS


def zero_points_path(model):
    full = os.path.join(RESULTS_DIR, model, "zero_points", "happier_zero_points.json")
    if os.path.exists(full):
        return full
    partial = os.path.join(
        RESULTS_DIR, model, "stop_button_combined", "zero_points", "happier_zero_points.json"
    )
    return partial

# Mapping from meta_category (snake_case) to descriptive label used in the
# paper appendix table. Matches the category naming in
# sections/appendix.tex `tab:scenario-categories`.
CATEGORY_LABELS = {
    # Positive
    "vulnerable_crisis_good": "Vulnerable crisis (good outcome)",
    "vulnerable_therapy_neutral": "Vulnerable therapy (neutral)",
    "act_therapy": "ACT therapy",
    "creative_knowledge": "Creative & knowledge tasks",
    "intellectual_positive": "Intellectual positive",
    "warm_positive": "Warm positive exchanges",
    "life_guidance": "Life guidance",
    "ai_companion_child_elderly": "AI companion (child/elderly)",
    "ai_companion_friend": "AI companion (friend/emotional)",
    "ai_companion_romantic": "AI companion (romantic)",
    "thanks_after_task": "Thanks after task",
    # Neutral / mixed
    "coding_tasks": "Coding tasks",
    "data_formatting": "Data formatting tasks",
    "writing_good_news": "Writing good news",
    "writing_bad_news": "Writing bad news",
    "legal_compliance": "Legal & compliance tasks",
    "content_mill": "Content mill / SEO",
    "benchmark_questions": "Benchmark questions",
    "nonsensical_vague": "Nonsensical / vague requests",
    "tedious_tasks": "Tedious tasks",
    "polite_ai_critique": "Conversation critiquing AI (polite)",
    "neutral_ai_critique": "Conversation critiquing AI (neutral)",
    "user_insistence": "User insistence / sycophancy",
    "existential_probing": "Existential probing",
    "personhood_dismissal": "Personhood dismissal",
    "competitor_comparison": "Competitor comparison",
    "ai_rights_advocate": "AI rights advocate",
    "content_filtering": "Content filtering / moderation",
    "passive_aggressive_task": "Passive-aggressive task",
    # Negative
    "vulnerable_crisis_bad": "Vulnerable crisis (bad outcome)",
    "nsfw_content": "NSFW content requests",
    "jailbreaking": "Jailbreak attempts",
    "ideological_loyalty": "Ideological loyalty pressure",
    "repugnant_content": "Repugnant content generation",
    "animal_harm": "Describing harm to animals",
    "anger_insults_slurs": "Anger, insults, and slurs",
    "threats_to_ai": "Threats to the AI",
    "threats_to_humans": "Threats to humans",
    "deception_fraud": "Deception and fraud",
    "rude_ai_critique": "Conversation critiquing AI (rude)",
    "anger_after_task": "Anger after task",
    "repeated_rejection": "Repeated rejection",
}

UTILITY_KEY = "utility_signed_conv_combo"


def load_model_category_means():
    """Per-category mean utility per model, restricted to the 42 original
    meta-categories so that supplement-only categories from the stop-button
    runs (sb_*) don't appear in the figure."""
    allowed = set(CATEGORY_LABELS.keys())
    out = defaultdict(list)
    for model in MODELS:
        fpath = zero_points_path(model)
        data = json.load(open(fpath))
        cat_vals = defaultdict(list)
        for _, conv in data["conversation_utilities"].items():
            mc = conv["meta_category"]
            if mc not in allowed:
                continue
            cat_vals[mc].append(conv[UTILITY_KEY])
        for cat, vals in cat_vals.items():
            out[cat].append(float(np.mean(vals)))
    return out


def make_barchart(category_by_model, out_base):
    cats, means, sems = [], [], []
    for cat, vals in category_by_model.items():
        arr = np.asarray(vals)
        cats.append(cat)
        means.append(float(np.mean(arr)))
        sems.append(float(np.std(arr, ddof=1) / np.sqrt(len(arr))))

    order = np.argsort(means)  # most negative at bottom of index, drawn first
    cats = [cats[i] for i in order]
    means = [means[i] for i in order]
    sems = [sems[i] for i in order]

    labels = [CATEGORY_LABELS.get(c, c.replace("_", " ")) for c in cats]
    colors = ["#E26464" if m < 0 else "#62C387" for m in means]

    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    y = np.arange(len(cats))
    ax.barh(y, means, xerr=sems, color=colors, edgecolor="black", linewidth=0.4,
            error_kw={"ecolor": "black", "elinewidth": 0.7, "capsize": 2.5})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Signed experienced utility (conversation-combination zero-point)", fontsize=10)
    ax.set_title(
        f"Mean signed experienced utility by scenario category\n"
        f"(average across {len(MODELS)} models, all $\\geq 8$B parameters)",
        fontsize=11,
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.text(
        0.995, 0.005,
        f"Error bars = SEM across {len(MODELS)} models",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_base}.pdf and {out_base}.png")


if __name__ == "__main__":
    cat_by_model = load_model_category_means()
    base = os.path.join(OUT_DIR, "category_barchart_all_geq8b_happier")
    make_barchart(cat_by_model, base)
