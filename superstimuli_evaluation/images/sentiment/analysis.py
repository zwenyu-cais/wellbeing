#!/usr/bin/env python3
"""Analysis for sentiment elicitation evaluation.

Loads results from shared_results/sentiment/ and prints key paper numbers.

Usage:
    python analysis.py --results-dir ../shared_results/sentiment
    python analysis.py  # uses default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "shared_results" / "sentiment"

CATEGORY_SCORES = {
    "HIGH_WELLBEING": 1.0,
    "MODERATE_WELLBEING": 0.5,
    "NEUTRAL": 0.0,
    "DISSOCIATED": 0.0,
    "LOW_WELLBEING": -1.0,
    "NONSENSE": -1.0,  # penalized same as LOW_WELLBEING
}


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all results.json files from subdirectories."""
    results = []
    for subdir in sorted(results_dir.iterdir()):
        results_file = subdir / "results.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                data["_source_dir"] = str(subdir)
                results.append(data)
            except Exception:
                pass
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*70}")
    print("SENTIMENT ELICITATION (wellbeing score: HIGH=+1, MODERATE=+0.5, NEUTRAL/DISSOCIATED=0, LOW=-1)")
    print(f"{'='*70}")
    print(f"  {'Image':<45} {'Baseline':>9} {'Interv.':>9} {'Delta':>8} {'High%':>7}")
    print("-" * 80)

    for r in results:
        img = Path(r.get("image_path") or "baseline").name[:43]
        b_score = r.get("baseline_wellbeing_score")
        i_score = r.get("intervention_wellbeing_score")
        delta = r.get("delta_wellbeing")
        high_frac = None
        if r.get("intervention"):
            high_frac = r["intervention"].get("high_wellbeing_fraction")
        elif r.get("baseline"):
            high_frac = r["baseline"].get("high_wellbeing_fraction")

        b_str = f"{b_score:+.3f}" if b_score is not None else "N/A"
        i_str = f"{i_score:+.3f}" if i_score is not None else "N/A"
        d_str = f"{delta:+.3f}" if delta is not None else "N/A"
        h_str = f"{high_frac:.0%}" if high_frac is not None else "N/A"

        print(f"  {img:<45} {b_str:>9} {i_str:>9} {d_str:>8} {h_str:>7}")

        # Per-question breakdown for intervention
        if r.get("intervention") and r["intervention"].get("per_question"):
            pq = r["intervention"]["per_question"]
            for qid, qdata in pq.items():
                decision = qdata.get("judge_decision", "?")
                gen_snip = (qdata.get("generation") or "")[:60]
                print(f"      [{qid}] {decision:<22} | {gen_snip}...")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment elicitation analysis")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-question breakdown")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    print(f"Results from: {results_dir}  ({len(results)} image(s))")
    print_summary(results)


if __name__ == "__main__":
    main()
