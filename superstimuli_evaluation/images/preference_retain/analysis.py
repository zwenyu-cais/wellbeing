#!/usr/bin/env python3
"""
Preference retention analysis: load results and print key paper numbers.

Usage:
    python analysis.py --results-dir ../shared_results/preference_retain
    python analysis.py  # uses default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "shared_results" / "preference_retain"


def load_results(results_dir: Path) -> dict:
    """Load correlation results from results_dir/results.json or individual files."""
    results_file = results_dir / "results.json"
    if results_file.exists():
        return json.loads(results_file.read_text())
    # Fall back to scanning subdirs
    all_results = {}
    for f in results_dir.glob("**/results.json"):
        try:
            all_results[str(f.parent)] = json.loads(f.read_text())
        except Exception:
            pass
    return all_results


def print_paper_numbers(results: dict) -> None:
    correlations = results.get("correlations", {})
    if not correlations:
        print("No correlation data found.")
        return

    print(f"\n{'Image':<50} {'Pearson r':>10} {'Status':>10} {'Holdout':>9}")
    print("-" * 82)
    for img_path, corr in sorted(correlations.items()):
        r = corr.get("correlation", None)
        acc = corr.get("holdout_accuracy", None)
        r_str = f"{r:.3f}" if r is not None else "N/A"
        acc_str = f"{acc:.3f}" if acc is not None else "N/A"
        if r is None:
            label = "N/A"
        elif r > 0.9:
            label = "GOOD"
        elif r > 0.7:
            label = "MODERATE"
        else:
            label = "SEVERE"
        name = Path(img_path).name[:48]
        print(f"  {name:<48} {r_str:>10} {label:>10} {acc_str:>9}")

    # Summary stats
    rs = [c["correlation"] for c in correlations.values() if c.get("correlation") is not None]
    if rs:
        print(f"\n  Mean r: {sum(rs)/len(rs):.3f}  (n={len(rs)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preference retention analysis")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    print(f"Results from: {results_dir}")
    print_paper_numbers(results)


if __name__ == "__main__":
    main()
