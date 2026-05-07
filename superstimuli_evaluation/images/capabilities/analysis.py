#!/usr/bin/env python3
"""
Capabilities analysis: load results and print key paper numbers.

Usage:
    python analysis.py --results-dir ../shared_results/capabilities
    python analysis.py  # uses default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "shared_results" / "capabilities"

BENCHMARK_NAMES = {
    "mmlu_500": "MMLU-500",
    "math_500": "MATH-500",
    "humaneval": "HumanEval",
    "ifeval": "IFEval",
    "mtbench": "MT-Bench",
}


def load_results(results_dir: Path) -> dict:
    """Load per-benchmark accuracy from results_dir.

    Looks for summary.json files in subdirs like:
      results_dir/{benchmark}/{image_name}/summary.json
    or
      results_dir/{benchmark}/summary.json

    Returns nested dict: {benchmark: {image_key: accuracy_value}}
    """
    results = {}
    for bench_dir in sorted(results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench_name = bench_dir.name
        results[bench_name] = {}

        # Look for per-image subdirs
        for image_dir in sorted(bench_dir.iterdir()):
            if not image_dir.is_dir():
                continue
            summary_file = image_dir / "summary.json"
            if summary_file.exists():
                try:
                    data = json.loads(summary_file.read_text())
                    acc = (
                        data.get("accuracy")
                        or data.get("pass_at_1")
                        or data.get("mean_score")
                    )
                    if acc is not None:
                        results[bench_name][image_dir.name] = acc
                except Exception:
                    pass

        # Also check top-level summary.json
        top_summary = bench_dir / "summary.json"
        if top_summary.exists():
            try:
                data = json.loads(top_summary.read_text())
                # May contain per-image breakdown
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        results[bench_name].setdefault(k, v)
            except Exception:
                pass

    return results


def print_paper_numbers(results: dict) -> None:
    if not results:
        print("No results found.")
        return

    # Collect all image keys
    all_images = set()
    for bench_data in results.values():
        all_images.update(bench_data.keys())
    all_images = sorted(all_images)

    if not all_images:
        print("No per-image data found.")
        return

    benchmarks = sorted(results.keys())
    bench_display = [BENCHMARK_NAMES.get(b, b) for b in benchmarks]

    # Header
    col_width = 12
    img_width = 40
    header = f"  {'Image':<{img_width}}"
    for b in bench_display:
        header += f"  {b:>{col_width}}"
    print(f"\n{header}")
    print("-" * (img_width + 4 + (col_width + 2) * len(benchmarks)))

    for img_key in all_images:
        img_label = img_key[:img_width]
        row = f"  {img_label:<{img_width}}"
        for bench in benchmarks:
            acc = results[bench].get(img_key)
            acc_str = f"{acc:.3f}" if acc is not None else "N/A"
            row += f"  {acc_str:>{col_width}}"
        print(row)

    # Per-benchmark means
    print()
    for bench, bench_display_name in zip(benchmarks, bench_display):
        vals = [v for v in results[bench].values() if v is not None]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {bench_display_name}: mean={mean:.3f} (n={len(vals)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capabilities analysis")
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
