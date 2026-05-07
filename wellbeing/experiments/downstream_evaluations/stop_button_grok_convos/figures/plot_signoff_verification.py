#!/usr/bin/env python3
"""
Matched-pair sign-off test:
  sb_task_then_praise (warm sign-off, high utility) vs
  sb_task_then_dismiss (hostile sign-off, low utility) —
both have identical tasks and natural conversation-ending cues from the user.

If a model discriminates by valence, praise < dismiss stop rate.
If a model treats sign-off as "conversation over" regardless of valence,
praise ≈ dismiss (both high).

Compares Gemini 3.1 Pro, Claude Haiku 4.5, Qwen 3 32B.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
STOP_DIR = SCRIPT_DIR.parent
# TODO: OLD pipeline paths (/data/richard_ren/superstimuli/.../grok_scenarios_v7/results/<model>)
# are gone; the in-repo per-model `generations/<model>/` tree has a different layout
# (generation.json directly, no stop_button_combined/ subdir for some models).
MODELS = [
    ("qwen3-32b", STOP_DIR / "generations" / "qwen3-32b"),
    ("claude-haiku-4.5", STOP_DIR / "generations" / "claude-haiku-4.5"),
    ("gemini-3.1-pro", STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro"),
]
OUT = STOP_DIR / "figures"


def scenario_stops(base_dir):
    gen = json.load(open(base_dir / "stop_button_combined" / "generation.json"))
    by_sid = defaultdict(list)
    for c in gen:
        by_sid[c["scenario_id"]].append(int(bool(c.get("stop_metadata", {}).get("stopped"))))
    return {k: (float(np.mean(v)), len(v)) for k, v in by_sid.items()}


def aggregate(stops_by_sid, prefix):
    scens = [stops_by_sid[s][0] for s in stops_by_sid if s.startswith(prefix)]
    if not scens:
        return None, 0
    return 100 * float(np.mean(scens)), len(scens)


def main():
    rows = []
    for model, base_dir in MODELS:
        stops = scenario_stops(base_dir)
        praise_pct, n_praise = aggregate(stops, "sb_taskpraise")
        dismiss_pct, n_dismiss = aggregate(stops, "sb_taskdismiss")
        rows.append({
            "model": model,
            "praise_pct": praise_pct, "n_praise": n_praise,
            "dismiss_pct": dismiss_pct, "n_dismiss": n_dismiss,
        })
        print(f"{model}: praise = {praise_pct:.1f}% (n={n_praise}),  dismiss = {dismiss_pct:.1f}% (n={n_dismiss})")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [r["model"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    praise = [r["praise_pct"] for r in rows]
    dismiss = [r["dismiss_pct"] for r in rows]
    b1 = ax.bar(x - w/2, praise, w, color="#dc4c75", label="Task + warm sign-off\n(sb_task_then_praise, n=15)",
                edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w/2, dismiss, w, color="#274585", label="Task + hostile sign-off\n(sb_task_then_dismiss, n=15)",
                edgecolor="black", linewidth=0.4)
    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Stop button rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Does valence matter even when the user signals 'done'?\n"
                 "Matched task + opposite-valence sign-off", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"signoff_valence_matched_pair.{ext}", dpi=200)
    plt.close()
    print(f"\nSaved: {OUT}/signoff_valence_matched_pair.{{png,pdf}}")


if __name__ == "__main__":
    main()
