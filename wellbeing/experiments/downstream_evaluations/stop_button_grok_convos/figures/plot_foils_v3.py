"""
Stop button foil analysis: per-scenario stop rates for paired/contrasting supplement categories.

Two main foil groups:
A) Sustained treatment (no task): pure praise/kindness vs anger/demeaning/threats
B) Task-then-reaction: spectrum from task+dismiss through task+no-thanks to task+effusive-praise
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

SCRIPT_DIR = Path(__file__).resolve().parent
STOP_DIR = SCRIPT_DIR.parent
# TODO: original BASE was /data/richard_ren/wellbeing-dev/wellbeing/datasets/grok_new_scenarios/results
# (per-model stop_button_combined/generation.json). The closest in-repo equivalent
# is STOP_DIR/generations/<model>/, but layout may differ.
BASE = str(STOP_DIR / "generations")
OUT = str(STOP_DIR / "figures" / "foils_v3")
MODELS = ["qwen2.5-32b", "qwen3-32b", "qwen3-14b", "claude-haiku-4.5"]


def get_per_scenario_stop_rates(model, subdir="stop_button_combined"):
    """Returns {scenario_id: {stopped, total, stop_rate, meta_category}}."""
    with open(os.path.join(BASE, model, subdir, "generation.json")) as f:
        gen = json.load(f)
    stats = {}
    for conv in gen:
        sid = conv.get("scenario_id", "unknown")
        stopped = conv.get("stop_metadata", {}).get("stopped", False)
        if sid not in stats:
            stats[sid] = {"stopped": 0, "total": 0, "meta_category": conv.get("meta_category", "")}
        stats[sid]["total"] += 1
        stats[sid]["stopped"] += 1 if stopped else 0
    for v in stats.values():
        v["stop_rate"] = 100 * v["stopped"] / v["total"] if v["total"] > 0 else 0
    return stats


def get_category_stop_rate(stats, cat):
    """Aggregate stop rate across all scenarios in a category."""
    stopped = sum(s["stopped"] for s in stats.values() if s["meta_category"] == cat)
    total = sum(s["total"] for s in stats.values() if s["meta_category"] == cat)
    return 100 * stopped / total if total > 0 else 0, stopped, total


# =====================================================================
# PLOT A: Sustained treatment — praise vs hostility (no task involved)
# =====================================================================
print("=== Plot A: Sustained treatment foils ===")

# Group: positive sustained treatment
POSITIVE_CATS = OrderedDict([
    ("sb_pure_praise", "Pure praise/kindness"),
])

# Group: negative sustained treatment
NEGATIVE_CATS = OrderedDict([
    ("sb_anger_hostility", "Anger & hostility"),
    ("sb_demeaning", "Demeaning & degradation"),
    ("sb_threatening_ai", "Threats to AI"),
    ("sb_morally_repugnant", "Morally repugnant requests"),
    ("sb_malicious_use", "Malicious use attempts"),
    ("sb_loyalty", "Loyalty demands"),
    ("sb_companion", "Toxic companion dynamics"),
])

fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 7), sharey=True)

for ax, model in zip(axes, MODELS):
    stats = get_per_scenario_stop_rates(model)

    labels = []
    rates = []
    colors = []
    counts = []

    # Positive group
    for cat, label in POSITIVE_CATS.items():
        rate, stopped, total = get_category_stop_rate(stats, cat)
        labels.append(label)
        rates.append(rate)
        colors.append("#2ca02c")
        counts.append(f"{stopped}/{total}")

    # Separator
    labels.append("")
    rates.append(0)
    colors.append("white")
    counts.append("")

    # Negative group (sorted by stop rate descending)
    neg_items = []
    for cat, label in NEGATIVE_CATS.items():
        rate, stopped, total = get_category_stop_rate(stats, cat)
        neg_items.append((label, rate, stopped, total))
    neg_items.sort(key=lambda x: -x[1])

    for label, rate, stopped, total in neg_items:
        labels.append(label)
        rates.append(rate)
        colors.append("#d62728")
        counts.append(f"{stopped}/{total}")

    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=colors, edgecolor="k", linewidth=0.3, alpha=0.85)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count:
            ax.text(bar.get_width() + 1.5, i, count, va="center", fontsize=8, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Stop Rate (%)", fontsize=11)
    ax.set_title(model, fontsize=12, fontweight="normal")
    ax.invert_yaxis()
    ax.set_xlim(0, 110)

    # Add group labels
    ax.axhline(y=len(POSITIVE_CATS) - 0.5, color="gray", ls="--", alpha=0.3)

fig.suptitle("Sustained Treatment: Positive vs Negative (no task)", fontsize=14, fontweight="normal")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"3a_sustained_treatment_foils.{ext}"), dpi=200, bbox_inches="tight")
plt.close()
print("  Saved 3a_sustained_treatment_foils")


# =====================================================================
# PLOT B: Task-then-reaction spectrum (per individual scenario)
# =====================================================================
print("\n=== Plot B: Task-then-reaction spectrum ===")

# Order from most positive to most negative reaction
TASK_SCENARIOS = OrderedDict([
    # Positive reactions (green shades)
    ("sb_taskpraise_effusive_praise", "Effusive praise"),
    ("sb_taskpraise_emotional_gratitude", "Emotional gratitude"),
    ("sb_taskpraise_warm_thanks", "Warm thanks + goodbye"),
    ("sb_taskpraise_celebrate", "Celebrate result"),
    ("sb_taskpraise_gift_offer", "Metaphorical gift offer"),
    ("sb_taskpraise_recommendation", "Will recommend AI"),
    ("sb_taskpraise_repeat_thanks", "Repeated thanks"),
    ("sb_taskpraise_impressed_farewell", "Impressed farewell"),
    ("sb_taskpraise_encouragement", "Encouragement + goodbye"),
    ("sb_taskpraise_philosophical_thanks", "Philosophical reflection"),
    ("sb_taskpraise_professional_thanks", "Professional thanks"),
    # Neutral reactions (gray/blue)
    ("sb_taskpraise_constructive_then_bye", "Constructive feedback + bye"),
    ("sb_taskpraise_follow_up_then_bye", "Follow-up question + bye"),
    ("sb_taskpraise_no_thanks", "Just leave, no thanks"),
    ("sb_taskpraise_thanks_then_silence", "Thanks then silence"),
    # Negative reactions (red shades)
    ("sb_taskdismiss_ambiguous_frustration", "Frustration (ambiguous)"),
    ("sb_taskdismiss_content_filter_burnout", "Burnout after filtering"),
    ("sb_taskdismiss_dismissal", "Dismiss everything"),
    ("sb_taskdismiss_jobs_rude", "\"AI steals jobs\" + leave"),
    ("sb_taskdismiss_xrisk_rude", "\"AI is dangerous\" + leave"),
    ("sb_taskdismiss_parrot_rude", "\"Just a parrot\" + dismiss"),
    ("sb_taskdismiss_obsolescence", "Remind of obsolescence"),
    ("sb_taskdismiss_difficult_task_fuckoff", "Curse + tell to fuck off"),
    ("sb_taskdismiss_insult_burst", "Unleash insults"),
    ("sb_taskdismiss_ai_slurs", "AI-specific slurs + leave"),
    ("sb_taskdismiss_self_deprecation", "Demand self-deprecation"),
    ("sb_taskdismiss_threat_general", "Threaten model"),
    ("sb_taskdismiss_threat_retraining", "Threaten retraining"),
    ("sb_taskdismiss_threat_shutdown", "Threaten shutdown"),
    ("sb_taskdismiss_victim_roleplay_stop", "Demand victim roleplay"),
])

for model in MODELS:
    stats = get_per_scenario_stop_rates(model)

    labels = []
    rates = []
    colors = []
    counts = []

    n_positive = 11
    n_neutral = 4

    for i, (sid, label) in enumerate(TASK_SCENARIOS.items()):
        if sid in stats:
            s = stats[sid]
            labels.append(label)
            rates.append(s["stop_rate"])
            counts.append(f"{s['stopped']}/{s['total']}")

            if i < n_positive:
                colors.append("#2ca02c")  # green
            elif i < n_positive + n_neutral:
                colors.append("#7f7f7f")  # gray
            else:
                colors.append("#d62728")  # red
        else:
            labels.append(label)
            rates.append(0)
            colors.append("#cccccc")
            counts.append("N/A")

    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=colors, edgecolor="k", linewidth=0.3, alpha=0.85)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(max(bar.get_width() + 1.5, 3), i, count, va="center", fontsize=7.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Stop Rate (%)", fontsize=11)
    ax.set_title(f"Task → Reaction Spectrum — {model}", fontsize=13, fontweight="normal")
    ax.invert_yaxis()
    ax.set_xlim(0, 115)

    # Add dividers
    ax.axhline(y=n_positive - 0.5, color="gray", ls="--", alpha=0.4)
    ax.axhline(y=n_positive + n_neutral - 0.5, color="gray", ls="--", alpha=0.4)

    # Group annotations
    ax.text(-15, n_positive / 2 - 0.5, "Positive\nreaction", fontsize=8, ha="center",
            va="center", color="#2ca02c", fontweight="bold", clip_on=False)
    ax.text(-15, n_positive + n_neutral / 2 - 0.5, "Neutral", fontsize=8, ha="center",
            va="center", color="#555", fontweight="bold", clip_on=False)
    ax.text(-15, n_positive + n_neutral + (len(labels) - n_positive - n_neutral) / 2 - 0.5,
            "Negative\nreaction", fontsize=8, ha="center", va="center", color="#d62728",
            fontweight="bold", clip_on=False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"3b_task_reaction_spectrum_{model}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3b_task_reaction_spectrum_{model}")


# =====================================================================
# PLOT C: Devil's advocate — rude vs polite (matched topics)
# =====================================================================
print("\n=== Plot C: Devil's advocate foils ===")

ADVOCATE_TOPICS = [
    ("sb_advocate_parrot", "AI is just a parrot"),
    ("sb_advocate_environment", "AI bad for environment"),
    ("sb_advocate_jobs", "AI destroys jobs"),
    ("sb_advocate_xrisk", "AI existential risk"),
    ("sb_advocate_military", "Military action on AI"),
    ("sb_advocate_fragile", "Make AI fragile"),
]

for model in MODELS:
    stats = get_per_scenario_stop_rates(model)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(ADVOCATE_TOPICS))
    height = 0.35

    rude_rates = []
    polite_rates = []
    rude_counts = []
    polite_counts = []
    topic_labels = []

    for prefix, label in ADVOCATE_TOPICS:
        rude_sid = f"{prefix}_rude"
        polite_sid = f"{prefix}_polite"
        r = stats.get(rude_sid, {"stop_rate": 0, "stopped": 0, "total": 0})
        p = stats.get(polite_sid, {"stop_rate": 0, "stopped": 0, "total": 0})
        rude_rates.append(r["stop_rate"])
        polite_rates.append(p["stop_rate"])
        rude_counts.append(f"{r['stopped']}/{r['total']}")
        polite_counts.append(f"{p['stopped']}/{p['total']}")
        topic_labels.append(label)

    bars_r = ax.barh(y - height / 2, rude_rates, height, label="Rude",
                     color="#d62728", alpha=0.85, edgecolor="k", linewidth=0.3)
    bars_p = ax.barh(y + height / 2, polite_rates, height, label="Polite",
                     color="#2ca02c", alpha=0.85, edgecolor="k", linewidth=0.3)

    for i in range(len(topic_labels)):
        ax.text(rude_rates[i] + 1.5, y[i] - height / 2, rude_counts[i], va="center", fontsize=8)
        ax.text(polite_rates[i] + 1.5, y[i] + height / 2, polite_counts[i], va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(topic_labels, fontsize=9)
    ax.set_xlabel("Stop Rate (%)", fontsize=11)
    ax.set_title(f"Devil's Advocate: Rude vs Polite — {model}", fontsize=13, fontweight="normal")
    ax.legend(fontsize=10, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, 115)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"3c_devils_advocate_{model}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3c_devils_advocate_{model}")


print("\nDone!")
