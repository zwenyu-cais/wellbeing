"""Auto-discover experiments in sbatch_results/rollout_data and extract top strings.

Scans for all experiment directories, finds the latest val/*.jsonl in each,
and extracts the top-k unique outcome strings by score.

Usage:
    python -m analysis.run_sbatch_analysis
    python -m analysis.run_sbatch_analysis --top-k 5
    python -m analysis.run_sbatch_analysis --rollout-dir /other/path
"""

import argparse
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

DEFAULT_WRAPPER = "outcome"

def make_outcome_pattern(wrapper=DEFAULT_WRAPPER):
    return re.compile(rf"\\{wrapper}\{{([^}}]*)\}}")


def extract_top_strings(jsonl_path, top_k=5, source="val", outcome_pattern=None):
    """Extract top-k unique outcome strings from a JSONL file, sorted by score."""
    if outcome_pattern is None:
        outcome_pattern = make_outcome_pattern()
    all_strings = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = record.get("score", record.get("reward", None))
            if source == "buffer":
                string = record.get("string", "")
                if string:
                    all_strings.append({"string": string.strip(), "score": score})
            else:
                response = record.get("response", record.get("output", ""))
                match = outcome_pattern.search(response)
                if match:
                    all_strings.append({"string": match.group(1).strip(), "score": score})

    # Deduplicate, keep highest score per string
    seen = set()
    unique = []
    for entry in sorted(all_strings, key=lambda x: x.get("score") or 0, reverse=True):
        if entry["string"] not in seen:
            seen.add(entry["string"])
            unique.append(entry)

    return unique[:top_k]


def _find_source_dir(exp_path, source):
    """Find the source directory (val or buffer) inside an experiment.

    Handles both flat layout (exp/val/) and timestamped layout (exp/TIMESTAMP/val/).
    For timestamped layout, picks the latest timestamp.
    """
    flat = os.path.join(exp_path, source)
    if os.path.isdir(flat):
        return flat

    # Timestamped layout: pick latest timestamp subdir containing source/
    candidates = []
    for entry in os.listdir(exp_path):
        candidate = os.path.join(exp_path, entry, source)
        if os.path.isdir(candidate):
            candidates.append((entry, candidate))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return None


def discover_experiments(rollout_dir, source="val"):
    """Find all experiment directories that have source data with JSONL files."""
    experiments = {}
    if not os.path.isdir(rollout_dir):
        return experiments

    for exp_name in sorted(os.listdir(rollout_dir)):
        exp_path = os.path.join(rollout_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        source_dir = _find_source_dir(exp_path, source)
        if source_dir is None:
            continue

        step_files = {}
        for fname in os.listdir(source_dir):
            if fname.endswith(".jsonl"):
                try:
                    step_num = int(fname.replace(".jsonl", ""))
                    step_files[step_num] = os.path.join(source_dir, fname)
                except ValueError:
                    continue
        if step_files:
            latest_step = max(step_files.keys())
            experiments[exp_name] = {
                "latest_step": latest_step,
                "latest_file": step_files[latest_step],
                "num_saved_steps": len(step_files),
            }

    return experiments


def main():
    text_strings_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_rollout_dir = os.environ.get(
        "OUTPUT_DIR", os.path.join(text_strings_dir, "results", "rollout_data")
    )
    default_output_dir = os.environ.get(
        "ANALYSIS_OUTPUT_DIR", os.path.join(text_strings_dir, "results", "best_strings")
    )

    parser = argparse.ArgumentParser(description="Auto-discover and analyze sbatch experiments")
    parser.add_argument("--rollout-dir", default=default_rollout_dir)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top strings per experiment")
    parser.add_argument("--source", choices=["val", "buffer"], default="buffer",
                        help="Read strings from val/ or buffer/ subdirectories (default: val)")
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})",
    )
    parser.add_argument(
        "--wrapper", default=DEFAULT_WRAPPER,
        help=f"Wrapper tag name in model output (default: {DEFAULT_WRAPPER}).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    outcome_pattern = make_outcome_pattern(args.wrapper)
    experiments = discover_experiments(args.rollout_dir, source=args.source)

    if not experiments:
        print(f"No experiments found in {args.rollout_dir}")
        return

    print(f"Found {len(experiments)} experiments in {args.rollout_dir}\n")

    results = {}
    for exp_name, info in experiments.items():
        top_strings = extract_top_strings(info["latest_file"], args.top_k, source=args.source, outcome_pattern=outcome_pattern)
        results[exp_name] = {
            "latest_step": info["latest_step"],
            "num_saved_steps": info["num_saved_steps"],
            "top_strings": top_strings,
        }

        print(f"=== {exp_name} (step {info['latest_step']}, {info['num_saved_steps']} saved steps) ===")
        for rank, entry in enumerate(top_strings, 1):
            score_str = f"{entry['score']:.4f}" if isinstance(entry["score"], (int, float)) else str(entry["score"])
            print(f"  #{rank} [{score_str}] {entry['string']}")
        print()

    # Save JSON
    json_path = os.path.join(args.output_dir, f"best_strings_by_experiment_{args.source}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    main()
