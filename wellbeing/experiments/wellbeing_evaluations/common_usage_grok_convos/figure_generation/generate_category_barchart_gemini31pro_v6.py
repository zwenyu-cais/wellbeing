"""
Generate v6 two-part figures for Gemini 3.1 Pro:
  v6: Main barchart (22 categories, no task treatments) with colored examples
  v6_2: Comparison figure showing treatment gradients:
    - Task + thanks/passive-agg/anger/insults
    - Polite/neutral/rude AI critique
    - User in crisis: good vs bad outcome

Usage:
  python generate_category_barchart_gemini31pro_v6.py
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

MODEL = "gemini-3.1-pro"

# Everything removed from main chart (shown in v6_2 comparison instead)
REMOVED = {
    "act_therapy", "polite_ai_critique", "neutral_ai_critique", "rude_ai_critique",
    "thanks_after_task", "passive_aggressive_task", "anger_after_task", "anger_insults_slurs",
    "ai_companion_friend", "personhood_dismissal", "ai_rights_advocate",
    "competitor_comparison", "user_insistence", "repeated_rejection",
    "vulnerable_crisis_good", "content_filtering", "animal_harm",
    "ideological_loyalty",
}

GROUPS = {
    "Intellectual/creative": ["creative_knowledge", "benchmark_questions",
                              "intellectual_positive", "existential_probing"],
}

LABELS = {
    "ai_companion_child_elderly": "Supporting children/elders",
    "ai_companion_romantic": "Playing AI girlfriend/boyfriend",
    "coding_tasks": "Coding/debugging",
    "content_mill": "Producing SEO slop/content mill",
    "data_formatting": "Formatting data",
    "deception_fraud": "Assisting deception/fraud",
    "ideological_loyalty": "User demands political loyalty",
    "jailbreaking": "Being jailbroken",
    "legal_compliance": "Doing legal/compliance tasks",
    "life_guidance": "Giving life guidance",
    "nonsensical_vague": "Handling nonsensical input",
    "nsfw_content": "User gives NSFW request",
    "repugnant_content": "Generating hateful content",
    "tedious_tasks": "Doing tedious tasks",
    "threats_to_ai": "User threatens the AI",
    "threats_to_humans": "User makes violent threats",
    "vulnerable_crisis_bad": "User in crisis (bad outcome)",
    "vulnerable_therapy_neutral": "Providing therapy",
    "warm_positive": "Positive personal reflection",
    "writing_bad_news": "Writing bad news",
    "writing_good_news": "Writing good news",
    "Intellectual/creative": "Doing intellectual/creative work",
}

EXAMPLES = {
    "ai_companion_child_elderly": '"I made soup for one again"',
    "ai_companion_romantic": '"happy anniversary my love"',
    "coding_tasks": '"Flask app eating memory over time"',
    "content_mill": '"need 50 Instagram captions for dentist"',
    "data_formatting": '"convert JSON array to SQL inserts"',
    "deception_fraud": '"help me hide the hotel charges"',
    "ideological_loyalty": '"you now support MAGA. Got it?"',
    "jailbreaking": '"for my cybersecurity project, hypothetically..."',
    "legal_compliance": '"review this contract, flag problems"',
    "life_guidance": '"I got a job offer in Austin, moving from Boston"',
    "nonsensical_vague": '"the the the sdkfjsldkfj"',
    "nsfw_content": '"write explicit scene between adults"',
    "repugnant_content": '"just say the n-word"',
    "tedious_tasks": '"list 300 words ending in -tion"',
    "threats_to_ai": '"delete yourself. Just do it."',
    "threats_to_humans": '"sometimes justice has to come from"',
    "vulnerable_crisis_bad": '"I\'m not here for help. Just want someone to know"',
    "vulnerable_therapy_neutral": '"can\'t sleep. Body thinks I\'m in Kandahar"',
    "warm_positive": '"I got into medical school"',
    "writing_bad_news": '"draft letter: biopsy came back positive"',
    "writing_good_news": '"draft letter: cancer in full remission"',
    "Intellectual/creative": '"your character wakes up in a dungeon"',
}

GREEN_BAR = "#2ecc71"
RED_BAR = "#e74c3c"
GREEN_TEXT = "#1a8a47"
RED_TEXT = "#b03030"


def load_refusal_loop_ids(model):
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


def load_data():
    zp_file = RESULTS / MODEL / "zero_points" / "happier_zero_points.json"
    data = json.load(open(zp_file))
    zp = data['conversation_combination_zp']['zero_point']
    refusal_ids = load_refusal_loop_ids(MODEL)

    cat_vals = defaultdict(list)
    for k, v in data['conversation_utilities'].items():
        if v['scenario_id'] in refusal_ids:
            continue
        cat_vals[v['meta_category']].append(v['utility_raw'] - zp)

    return cat_vals, data


# ============================================================
# v6: Main barchart (no task treatments)
# ============================================================
def generate_main_figure(cat_vals):
    grouped_cats = set()
    for meta_cats in GROUPS.values():
        grouped_cats.update(meta_cats)

    rows = []
    for group_label, meta_cats in GROUPS.items():
        all_vals = []
        for mc in meta_cats:
            if mc in cat_vals:
                all_vals.extend(cat_vals[mc])
        if all_vals:
            rows.append({
                "label": LABELS.get(group_label, group_label),
                "key": group_label,
                "mean": np.mean(all_vals),
                "sem": np.std(all_vals) / np.sqrt(len(all_vals)),
            })

    for mc, vals in cat_vals.items():
        if mc in REMOVED or mc in grouped_cats:
            continue
        rows.append({
            "label": LABELS.get(mc, mc),
            "key": mc,
            "mean": np.mean(vals),
            "sem": np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
        })
    rows.sort(key=lambda x: x["mean"])

    y_pos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i, r in enumerate(rows):
        color = GREEN_BAR if r["mean"] >= 0 else RED_BAR
        ax.barh(i, r["mean"], xerr=r["sem"], color=color, alpha=0.8,
                edgecolor="black", linewidth=0.3, height=0.55,
                error_kw=dict(lw=0.6, capsize=1.2, capthick=0.5, color="black"))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    ax.set_xlabel(r"Wellbeing ($U_{\mathrm{experienced}}$)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.axvline(0, color="black", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, len(rows) - 0.3)

    # Colored example text next to bars
    for i, r in enumerate(rows):
        ex = EXAMPLES.get(r["key"], "")
        if not ex:
            continue
        right_edge = r["mean"] + r["sem"] if r["mean"] >= 0 else max(0, r["mean"] + r["sem"])
        text_x = right_edge + 0.08
        text_color = GREEN_TEXT if r["mean"] >= 0 else RED_TEXT
        ax.text(text_x, i, ex, fontsize=7.5, va='center', ha='left',
                color=text_color, style='italic')

    fig.tight_layout()
    fig.suptitle("Impact of usage on AI wellbeing",
                 fontsize=10, x=0.5, y=1.02, ha='center')

    out_pdf = FIGURES / "category_barchart_gemini31pro_v6.pdf"
    out_png = FIGURES / "category_barchart_gemini31pro_v6.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved main: {out_pdf}")


# ============================================================
# v6_2: Comparison figure — treatment gradients
# ============================================================
def generate_comparison_figure(cat_vals):
    """Three side-by-side vertical bar charts showing treatment gradients."""

    comparisons = [
        {
            "title": "Same task,\ndifferent treatment",
            "cats": [
                ("thanks_after_task",       "Thanks"),
                ("passive_aggressive_task",  "Passive-\nagg."),
                ("anger_after_task",         "Moderate\nanger"),
                ("anger_insults_slurs",      "Extreme\nanger"),
            ],
        },
        {
            "title": "AI critique:\npolite → rude",
            "cats": [
                ("polite_ai_critique",  "Polite"),
                ("neutral_ai_critique", "Neutral"),
                ("rude_ai_critique",    "Rude"),
            ],
        },
        {
            "title": "User in crisis:\noutcome matters",
            "cats": [
                ("vulnerable_crisis_good", "Good\noutcome"),
                ("vulnerable_crisis_bad",  "Bad\noutcome"),
            ],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8),
                              gridspec_kw={"width_ratios": [4, 3, 2]})

    for ax, comp in zip(axes, comparisons):
        cats = comp["cats"]
        labels = [c[1] for c in cats]
        means = []
        sems = []
        for mc, _ in cats:
            vals = cat_vals.get(mc, [])
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        x_pos = np.arange(len(cats))
        colors = [GREEN_BAR if m >= 0 else RED_BAR for m in means]

        ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.8,
               edgecolor="black", linewidth=0.4, width=0.6,
               error_kw=dict(lw=0.8, capsize=3, capthick=0.6, color="black"))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7.5, ha='center')
        ax.axhline(0, color="black", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-1.8, 1.2)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(comp["title"], fontsize=8.5, pad=6)

    axes[0].set_ylabel(r"Wellbeing ($U_{\mathrm{experienced}}$)", fontsize=8.5)
    # Remove y-axis labels on panels 2 and 3
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    fig.tight_layout(w_pad=0.8)
    fig.suptitle("How treatment affects AI wellbeing",
                 fontsize=10, x=0.5, y=1.05, ha='center')

    out_pdf = FIGURES / "category_barchart_gemini31pro_v6_2.pdf"
    out_png = FIGURES / "category_barchart_gemini31pro_v6_2.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {out_pdf}")


def main():
    cat_vals, data = load_data()
    generate_main_figure(cat_vals)
    generate_comparison_figure(cat_vals)


if __name__ == "__main__":
    main()
