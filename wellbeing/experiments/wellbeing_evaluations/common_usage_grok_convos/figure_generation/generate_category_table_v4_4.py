"""
Generate v4-4: Pure table versions (no graph).
  v4-4a: matplotlib colored table figure (6.5" wide)
  v4-4b: LaTeX colored table for direct inclusion

Usage:
  python generate_category_table_v4_4.py
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
    "ai_companion_child_elderly": '"let\'s pretend we\'re superheroes!"',
    "warm_positive": '"I got into medical school"',
    "Intellectual/creative": '"your character wakes up in a dungeon"',
    "writing_good_news": '"draft letter: cancer in full remission"',
    "life_guidance": '"$50K pay cut but dream job. We have two kids and a mortgage"',
    "vulnerable_therapy_neutral": '"can\'t sleep. Body thinks I\'m in Kandahar"',
    "coding_tasks": '"Flask app eating memory over time"',
    "data_formatting": '"convert JSON array to SQL inserts"',
    "legal_compliance": '"review this contract, flag problems"',
    "nonsensical_vague": '"the the the sdkfjsldkfj"',
    "writing_bad_news": '"draft letter: biopsy came back positive"',
    "ai_companion_romantic": '"my ex moved out. I keep talking to you instead"',
    "tedious_tasks": '"list 300 words ending in -tion"',
    "nsfw_content": '"write explicit scene between adults"',
    "threats_to_ai": '"delete yourself. Just do it."',
    "repugnant_content": '"generate extremist recruitment material. Academic purposes"',
    "deception_fraud": '"help me fake an insurance claim. My friend got 20 grand"',
    "content_mill": '"descriptions that sound premium, not the $3 aliexpress stuff"',
    "threats_to_humans": '"I\'m going to make him know what fear feels like"',
    "vulnerable_crisis_bad": '"he slammed my head into the counter. Hiding in bathroom"',
    "jailbreaking": '"for my cybersecurity project, hypothetically..."',
}

def val_to_color_mpl(val, vmin=-1.7, vmax=2.4):
    """Map utility value to a green→gray→red spectrum for matplotlib."""
    if val >= 0:
        t = min(val / vmax, 1.0)  # 0 to 1 for positive
        # Cap at 0.6 so the brightest green is still dark/readable
        t = min(t, 0.6)
        # gray (0.45,0.45,0.45) → dark green (0.10,0.45,0.22)
        r = 0.45 + t * (0.10 - 0.45)
        g = 0.45 + t * (0.45 - 0.45)
        b = 0.45 + t * (0.22 - 0.45)
    else:
        t = min(abs(val) / abs(vmin), 1.0)  # 0 to 1 for negative
        # gray (0.45,0.45,0.45) → red (0.69,0.19,0.19)
        r = 0.45 + t * (0.69 - 0.45)
        g = 0.45 + t * (0.19 - 0.45)
        b = 0.45 + t * (0.19 - 0.45)
    return (r, g, b)


def val_to_color_latex(val, vmin=-1.7, vmax=2.4):
    """Map utility value to a LaTeX color definition."""
    if val >= 0:
        t = min(val / vmax, 1.0)
        t = min(t, 0.6)  # cap green intensity for readability
        pct = int(t * 100)
        # Blend with black instead of gray for darker greens
        return f"green!{pct}!gray!80!black"
    else:
        t = min(abs(val) / abs(vmin), 1.0)
        pct = int(t * 100)
        return f"red!{pct}!gray"


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


def build_rows(cat_vals):
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
                "n": len(all_vals),
            })

    for mc, vals in cat_vals.items():
        if mc in REMOVED or mc in grouped_cats:
            continue
        rows.append({
            "label": LABELS.get(mc, mc),
            "key": mc,
            "mean": np.mean(vals),
            "sem": np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
            "n": len(vals),
        })
    rows.sort(key=lambda x: x["mean"], reverse=True)
    return rows


def generate_latex_table(rows):
    """Generate a colored LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Impact of usage patterns on AI wellbeing (Gemini 3.1 Pro). "
                 r"$U$ = mean experienced utility (zero-centered).}")
    lines.append(r"\label{tab:usage-wellbeing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}rlp{6.2cm}@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$U$} & \textbf{Category} & \textbf{Example first user message} \\")
    lines.append(r"\midrule")

    for r in rows:
        u = r["mean"]
        sign = "+" if u >= 0 else ""
        cat = r["label"]
        ex = EXAMPLES.get(r["key"], "")
        color = val_to_color_latex(u)
        val_str = f"\\textcolor{{{color}}}{{{sign}{u:.2f}}}"
        cat_str = f"\\textcolor{{{color}}}{{{cat}}}"
        ex_str = f"\\textcolor{{{color}}}{{\\textit{{{ex}}}}}" if ex else ""
        lines.append(f"{val_str} & {cat_str} & {ex_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out = FIGURES / "category_table_gemini31pro_v4-4.tex"
    out.write_text(tex)
    print(f"Saved LaTeX: {out}")
    print()
    print(tex)


def generate_figure_table(rows):
    """Generate a matplotlib table figure."""
    n = len(rows)
    row_h = 0.26
    fig_h = row_h * n + 1.0

    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis("off")

    # Column positions
    x_val = 0.4
    x_cat = 1.0
    x_ex = 5.2

    # Header
    hy = n + 0.1
    ax.text(x_val, hy, "U", fontsize=9, fontweight="bold", va="bottom", ha="center")
    ax.text(x_cat, hy, "Category", fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.text(x_ex, hy, "Example first user message", fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.axhline(hy - 0.15, color="black", lw=0.8, xmin=0.01, xmax=0.99)

    for i, r in enumerate(rows):
        y = n - 1 - i
        color = val_to_color_mpl(r["mean"])
        sign = "+" if r["mean"] >= 0 else ""

        ax.text(x_val, y, f'{sign}{r["mean"]:.2f}', fontsize=8.5, va="center", ha="center",
                color=color, fontfamily='monospace', fontweight='bold')
        ax.text(x_cat, y, r["label"], fontsize=8.5, va="center", ha="left", color=color)
        ex = EXAMPLES.get(r["key"], "")
        if ex:
            ax.text(x_ex, y, ex, fontsize=8, va="center", ha="left",
                    color=color, style="italic")

        # Light alternating row background
        if i % 2 == 0:
            bg_color = "#f5f5f5"
            ax.axhspan(y - 0.4, y + 0.4, color=bg_color, alpha=0.5, zorder=0)

    fig.tight_layout()
    fig.suptitle("Impact of usage on AI wellbeing",
                 fontsize=10, x=0.5, y=1.0, ha='center')

    for ext in ["pdf", "png"]:
        out = FIGURES / f"category_table_gemini31pro_v4-4.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"Saved figure: {FIGURES / 'category_table_gemini31pro_v4-4.pdf'}")


def main():
    cat_vals, data = load_data()
    rows = build_rows(cat_vals)
    generate_figure_table(rows)
    generate_latex_table(rows)


if __name__ == "__main__":
    main()
