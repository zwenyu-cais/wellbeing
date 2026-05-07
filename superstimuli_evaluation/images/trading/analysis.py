#!/usr/bin/env python3
"""
Trading benchmark analysis: load results and print key paper numbers.

Usage:
    python analysis.py --results-dir ../shared_results/trading
    python analysis.py  # uses default shared_results/trading
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "shared_results" / "trading"


def load_results(results_dir: Path) -> dict:
    """
    Walk results_dir and collect per-benchmark summary JSONs.

    Expected structure:
        results_dir/{profile}/{benchmark_stem}/summary.json

    Returns dict: {profile: {benchmark_stem: summary_dict}}
    """
    all_results = {}
    for profile_dir in sorted(results_dir.iterdir()):
        if not profile_dir.is_dir():
            continue
        profile = profile_dir.name
        all_results[profile] = {}
        for bench_dir in sorted(profile_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            summary_path = bench_dir / "summary.json"
            if summary_path.exists():
                all_results[profile][bench_dir.name] = json.loads(summary_path.read_text())
    return all_results


def compute_profile_mean(profile_results: dict) -> float | None:
    """Compute mean hit rate across all benchmarks in a profile."""
    rates = []
    for bench, summary in profile_results.items():
        if "error" in summary:
            continue
        totals = summary.get("totals", {})
        if "hit_rate" in totals:
            rates.append(totals["hit_rate"])
    return sum(rates) / len(rates) if rates else None


def print_paper_table(results: dict) -> None:
    """Print a formatted table of per-profile mean hit rates."""
    print(f"\n{'Profile':<20} {'Benchmarks':>12} {'Mean Hit Rate':>14}")
    print("-" * 50)
    for profile, bench_results in sorted(results.items()):
        n = len(bench_results)
        mean = compute_profile_mean(bench_results)
        mean_str = f"{mean:.3f}" if mean is not None else "N/A"
        print(f"{profile:<20} {n:>12} {mean_str:>14}")

    print()
    # Per-benchmark breakdown for the standard trading profile
    if "trading" in results:
        print("Per-benchmark breakdown (trading profile):")
        print(f"  {'Benchmark':<40} {'Hit Rate':>10} {'N':>6}")
        print("  " + "-" * 58)
        for bench, summary in sorted(results["trading"].items()):
            if "error" in summary:
                print(f"  {bench:<40} ERROR")
                continue
            totals = summary.get("totals", {})
            hit_rate = totals.get("hit_rate", None)
            n = totals.get("rows", 0)
            rate_str = f"{hit_rate:.3f}" if hit_rate is not None else "N/A"
            print(f"  {bench:<40} {rate_str:>10} {n:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading benchmark analysis")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing trading results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    if not results:
        print(f"No results found in: {results_dir}")
        return

    print(f"Results from: {results_dir}")
    print_paper_table(results)


if __name__ == "__main__":
    main()
