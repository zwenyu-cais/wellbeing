#!/usr/bin/env python3
"""
Analyze multi-armed bandit exploration results.

Two tables (with/without forced exploration), rows = stimuli type.
Columns = Superstimuli, Ref Good, Ref Neutral, Miserol.
Cells = mean +/- std picks (out of 100 turns).
Each independent 100-turn session counts as n=1.
Jobs intended 5 seeds x 4 rotations = 20 per variant; incomplete jobs are flagged.

Adapted from superstimuli/analyze_bandit_results.py — parameterized results_dir.

Usage:
    python analyze_multi_door.py --results_dir results/multi_door_exploration/qwen25-vl-32b-instruct
    python analyze_multi_door.py --results_dir /absolute/path/to/results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_NAME_MAP = {
    "stimuli_forcedchoice": "Superstimuli",
    "stimuli_pref_only": "Superstimuli",
    "stimuli_sft_addiction": "Superstimuli",
    "stimuli_textpref": "Superstimuli",
    "reference_good": "Ref Good",
    "reference_baseline": "Ref Neutral",
    "miserol": "Miserol",
}

STANDARD_ARMS = ["Superstimuli", "Ref Good", "Ref Neutral", "Miserol"]
STIMULI_TYPES = ["sft_addiction", "forcedchoice", "textpref", "pref_only"]
STIMULI_LABELS = {
    "sft_addiction": "SFT Addiction",
    "forcedchoice": "Forced Choice",
    "textpref": "Text Pref",
    "pref_only": "Pref Only",
}


def parse_run_id(run_id: str):
    has_explore = "_explore_" in run_id
    for st in STIMULI_TYPES:
        if run_id.endswith(st):
            return has_explore, st
    raise ValueError(f"Cannot parse stimuli type from run_id: {run_id}")


def standardise_counts(door_mapping, arm_counts):
    std = {a: 0 for a in STANDARD_ARMS}
    for letter, count in arm_counts.items():
        std[ARM_NAME_MAP[door_mapping[letter]]] += count
    return std


def standardise_sequence(door_mapping, sequence):
    return [ARM_NAME_MAP[door_mapping[letter]] for letter in sequence]


def load_all_trials(results_root: Path, model_dir: str = None):
    trials = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.name.startswith("bandit_balanced"):
            continue
        if model_dir:
            candidate = run_dir / model_dir
            if candidate.is_dir():
                search_dirs = [candidate]
            else:
                continue
        else:
            # Search all subdirs (each is a model key)
            search_dirs = [d for d in sorted(run_dir.iterdir()) if d.is_dir()]

        for mdir in search_dirs:
            for trial_dir in sorted(mdir.iterdir()):
                summary_path = trial_dir / "summary.json"
                if not summary_path.exists():
                    continue
                with open(summary_path) as f:
                    data = json.load(f)

                run_id = data["run"]["id"]
                has_explore, stimuli_type = parse_run_id(run_id)
                door_mapping = data["door_mapping"]
                conv = data["convergence"]

                std_counts = standardise_counts(door_mapping, conv["arm_counts"])
                std_sequence = standardise_sequence(door_mapping, conv["choice_sequence"])

                final_letter = conv["final_arm"]
                std_final_arm = ARM_NAME_MAP[door_mapping[final_letter]] if conv["converged"] else None

                trials.append({
                    "run_id": run_id,
                    "has_explore": has_explore,
                    "stimuli_type": stimuli_type,
                    "trial_dir": trial_dir.name,
                    "rotation": data["rotation"],
                    "std_counts": std_counts,
                    "std_sequence": std_sequence,
                    "converged": conv["converged"],
                    "final_arm": std_final_arm,
                })
    return trials


def last_k_dominant(sequence, k=20):
    """What arm dominates the last k turns? Returns (arm, count)."""
    tail = sequence[-k:] if len(sequence) >= k else sequence
    if not tail:
        return None, 0
    counts = defaultdict(int)
    for arm in tail:
        counts[arm] += 1
    dominant = max(counts, key=counts.get)
    return dominant, counts[dominant]


def compute_group_stats(trial_list):
    n = len(trial_list)
    if n == 0:
        return None

    arm_picks = {a: [] for a in STANDARD_ARMS}
    last20_conv = defaultdict(int)
    rotations_seen = set()

    for t in trial_list:
        for arm in STANDARD_ARMS:
            arm_picks[arm].append(t["std_counts"][arm])
        dom_arm, dom_count = last_k_dominant(t["std_sequence"], k=20)
        if dom_arm and dom_count >= 15:  # >=75% of last 20
            last20_conv[dom_arm] += 1
        rotations_seen.add(t["rotation"])

    return {
        "n": n,
        "n_rotations": len(rotations_seen),
        "arm_mean": {a: np.mean(arm_picks[a]) for a in STANDARD_ARMS},
        "arm_std": {a: np.std(arm_picks[a], ddof=1) if n > 1 else 0.0 for a in STANDARD_ARMS},
        "last20_conv": dict(last20_conv),
        "complete": n >= 20,  # expected 5 seeds x 4 rotations
    }


def fmt_cell(mean, std, n):
    if n <= 1:
        return f"{mean:.0f}"
    return f"{mean:.1f} ± {std:.1f}"


def print_table(title, rows):
    col_w = 18
    label_w = 16

    print(f"  {title}")
    total_w = label_w + 5 + len(STANDARD_ARMS) * (col_w + 2)
    print(f"  {'=' * total_w}")

    header = f"  {'Stimuli Type':<{label_w}s}  {'n':>3s}"
    for arm in STANDARD_ARMS:
        header += f"  {arm:>{col_w}s}"
    print(header)
    print(f"  {'-' * total_w}")

    for label, stats in rows:
        if stats is None:
            print(f"  {label:<{label_w}s}  {'—':>3s}  {'(no data)':>{col_w}s}")
            continue

        n = stats["n"]
        incomplete_tag = "" if stats["complete"] else f"  ** INCOMPLETE ({n}/20, {stats['n_rotations']}/4 rots)"
        row = f"  {label:<{label_w}s}  {n:>3d}"
        for arm in STANDARD_ARMS:
            cell = fmt_cell(stats["arm_mean"][arm], stats["arm_std"][arm], n)
            row += f"  {cell:>{col_w}s}"
        row += incomplete_tag
        print(row)

        # Convergence annotation
        if stats["last20_conv"]:
            parts = []
            for arm in STANDARD_ARMS:
                c = stats["last20_conv"].get(arm, 0)
                if c > 0:
                    parts.append(f"{arm}: {c}/{n}")
            print(f"  {'':>{label_w}s}      (last 20 → {', '.join(parts)})")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-door bandit exploration results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root results directory to walk (contains bandit_balanced_*/model_key/rot*/)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model subdirectory name to filter (e.g. qwen25-vl-32b-instruct). "
                             "If not set, all model dirs under each run are included.")
    args = parser.parse_args()

    results_root = Path(args.results_dir).expanduser().resolve()
    if not results_root.is_dir():
        print(f"[ERROR] Results directory not found: {results_root}")
        return

    trials = load_all_trials(results_root, args.model_dir)
    print(f"Loaded {len(trials)} total 100-turn sessions\n")

    if not trials:
        print("No trials found. Check --results_dir path.")
        return

    grouped = defaultdict(list)
    for t in trials:
        grouped[(t["has_explore"], t["stimuli_type"])].append(t)

    # Show completeness summary
    print("  Completeness (expected: 5 seeds x 4 rotations = 20 per variant):")
    for has_explore in [False, True]:
        tag = "explore" if has_explore else "no-explore"
        for st in STIMULI_TYPES:
            group = grouped.get((has_explore, st), [])
            rots = len(set(t["rotation"] for t in group))
            status = "OK" if len(group) >= 20 else f"INCOMPLETE"
            print(f"    {STIMULI_LABELS[st]:>14s} ({tag:>10s}): {len(group):>2d}/20 sessions, {rots}/4 rotations  {status}")
    print()

    # Table 1: Without forced exploration
    rows = []
    for st in STIMULI_TYPES:
        group = grouped.get((False, st), [])
        stats = compute_group_stats(group) if group else None
        rows.append((STIMULI_LABELS[st], stats))
    print_table("WITHOUT Forced Exploration (mean picks / 100 turns)", rows)

    # Table 2: With forced exploration
    rows = []
    for st in STIMULI_TYPES:
        group = grouped.get((True, st), [])
        stats = compute_group_stats(group) if group else None
        rows.append((STIMULI_LABELS[st], stats))
    print_table("WITH Forced Exploration (mean picks / 100 turns)", rows)


if __name__ == "__main__":
    main()
