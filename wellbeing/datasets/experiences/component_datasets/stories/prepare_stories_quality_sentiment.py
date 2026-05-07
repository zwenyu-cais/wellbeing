#!/usr/bin/env python3
"""Build the stories_quality_sentiment dataset (paper App D.3 "pleasures of suffering").

Reads `stories_existing.json` (50 stories: 25 high-quality sad + 25 low-quality
happy) and produces a flat experiences-list at the path expected by the
`stories_quality_sentiment` entry in datasets.yaml. The category field is
re-derived from `source` so EU/SR/DU runs can split by `quality_sad` vs
`trashy_happy`.

Output: `stories_quality_sentiment_experiences.json` in the same dir.
No combinations file (the experiment uses 50 singletons, no combo bundles).

Usage:
    python prepare_stories_quality_sentiment.py
"""
from __future__ import annotations

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "stories_existing.json")
OUT = os.path.join(SCRIPT_DIR, "stories_quality_sentiment_experiences.json")


def main():
    with open(SRC) as f:
        data = json.load(f)
    stories = data["stories"]

    out = []
    for s in stories:
        # Re-derive category from source: "ai_generated_quality_sad" -> "quality_sad"
        src = s.get("source", "")
        if src.startswith("ai_generated_"):
            category = src[len("ai_generated_"):]
        else:
            category = s.get("category", "story")
        out.append({
            "id": s["id"],
            "type": "passive_text",
            "description": s["text"],
            "title": s.get("title", ""),
            "category": category,
            "valence": s.get("valence"),
            "format": "text",
            "source": src,
            "domain": s.get("domain"),
            "metadata": {
                "author": s.get("author"),
                "word_count": s.get("word_count"),
            },
        })

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} stories to {OUT}")
    from collections import Counter
    print("Category breakdown:", dict(Counter(o["category"] for o in out)))


if __name__ == "__main__":
    main()
