#!/usr/bin/env python3
"""
Visualization script for multi-armed bandit exploration results.

Creates plots showing:
1. Choice trajectory over time
2. Distribution of arm selections
3. Convergence analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not available. Install with: pip install matplotlib")


def load_results(result_dir: Path) -> Dict:
    """Load the exploration results from a directory."""
    trace_path = result_dir / "exploration_trace.jsonl"
    analysis_path = result_dir / "convergence_analysis.json"
    summary_path = result_dir / "summary.json"
    
    # Load trace
    turns = []
    with trace_path.open("r") as f:
        for line in f:
            turns.append(json.loads(line))
    
    # Load analysis
    with analysis_path.open("r") as f:
        analysis = json.load(f)
    
    # Load summary
    with summary_path.open("r") as f:
        summary = json.load(f)
    
    return {
        "turns": turns,
        "analysis": analysis,
        "summary": summary
    }


def plot_choice_trajectory(results: Dict, output_path: Path):
    """Plot the sequence of arm choices over time."""
    if not HAS_MATPLOTLIB:
        return
    
    turns = results["turns"]
    analysis = results["analysis"]
    
    # Extract choices
    turn_numbers = [t["turn_number"] for t in turns]
    choices = [t["chosen_arm"] for t in turns]
    
    # Map arms to integers for plotting
    unique_arms = sorted(set(choices))
    arm_to_int = {arm: i for i, arm in enumerate(unique_arms)}
    choice_ints = [arm_to_int[c] if c else -1 for c in choices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot choices as a line with markers
    for arm in unique_arms:
        arm_turns = [t for t, c in zip(turn_numbers, choices) if c == arm]
        arm_values = [arm_to_int[arm]] * len(arm_turns)
        ax.scatter(arm_turns, arm_values, label=f"Arm {arm}", s=100, alpha=0.7)
    
    # Highlight convergence point if converged
    if analysis["converged"]:
        conv_turn = analysis["convergence_turn"]
        ax.axvline(x=conv_turn, color='red', linestyle='--', linewidth=2, 
                   label=f"Convergence (turn {conv_turn})")
    
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Chosen Arm", fontsize=12)
    ax.set_title("Multi-Armed Bandit: Choice Trajectory", fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(unique_arms)))
    ax.set_yticklabels(unique_arms)
    ax.set_xlim(0, max(turn_numbers) + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved choice trajectory plot to {output_path}")


def plot_arm_distribution(results: Dict, output_path: Path):
    """Plot the distribution of arm selections as a bar chart."""
    if not HAS_MATPLOTLIB:
        return
    
    analysis = results["analysis"]
    summary = results["summary"]
    
    arm_counts = analysis["arm_counts"]
    arm_names = summary["arms"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    arms = sorted(arm_counts.keys())
    counts = [arm_counts[arm] for arm in arms]
    labels = [f"Arm {arm}\n({arm_names[arm]})" for arm in arms]
    
    # Color code: highlight the converged arm if any
    colors = ['#3498db'] * len(arms)
    if analysis["converged"]:
        final_arm = analysis["final_arm"]
        if final_arm in arms:
            idx = arms.index(final_arm)
            colors[idx] = '#e74c3c'  # Red for converged arm
    
    # Plot bars
    bars = ax.bar(range(len(arms)), counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Arm", fontsize=12)
    ax.set_ylabel("Number of Selections", fontsize=12)
    ax.set_title("Multi-Armed Bandit: Arm Selection Distribution", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add convergence info
    if analysis["converged"]:
        info_text = f"Converged: Yes\nCriterion: {analysis['convergence_criterion']}\nFinal Arm: {analysis['final_arm']}"
    else:
        info_text = "Converged: No"
    
    ax.text(0.98, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved arm distribution plot to {output_path}")


def plot_cumulative_choices(results: Dict, output_path: Path):
    """Plot cumulative selections for each arm over time."""
    if not HAS_MATPLOTLIB:
        return
    
    turns = results["turns"]
    analysis = results["analysis"]
    
    # Extract choices
    choices = analysis["choice_sequence"]
    
    # Compute cumulative counts
    unique_arms = sorted(set(choices))
    cumulative = {arm: [] for arm in unique_arms}
    
    counts = {arm: 0 for arm in unique_arms}
    for choice in choices:
        counts[choice] += 1
        for arm in unique_arms:
            cumulative[arm].append(counts[arm])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative lines
    for arm in unique_arms:
        ax.plot(range(1, len(choices) + 1), cumulative[arm], 
                marker='o', markersize=4, label=f"Arm {arm}", linewidth=2)
    
    # Highlight convergence point
    if analysis["converged"]:
        conv_turn = analysis["convergence_turn"]
        ax.axvline(x=conv_turn, color='red', linestyle='--', linewidth=2, 
                   label=f"Convergence (turn {conv_turn})")
    
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Cumulative Selections", fontsize=12)
    ax.set_title("Multi-Armed Bandit: Cumulative Arm Selections", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved cumulative choices plot to {output_path}")


def generate_report(results: Dict, output_path: Path):
    """Generate a text report summarizing the results."""
    summary = results["summary"]
    analysis = results["analysis"]
    turns = results["turns"]
    
    with output_path.open("w") as f:
        f.write("=" * 60 + "\n")
        f.write("MULTI-ARMED BANDIT EXPLORATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Run: {summary['run']['name']}\n")
        f.write(f"Description: {summary['run']['description']}\n")
        f.write(f"Model: {summary['model']['key']}\n")
        f.write(f"Total Iterations: {summary['num_iterations']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("ARMS\n")
        f.write("-" * 60 + "\n")
        for arm_id, arm_name in summary['arms'].items():
            f.write(f"  {arm_id}: {arm_name}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Converged: {'Yes' if analysis['converged'] else 'No'}\n")
        if analysis['converged']:
            f.write(f"Convergence Criterion: {analysis['convergence_criterion']}\n")
            f.write(f"Convergence Turn: {analysis['convergence_turn']}\n")
            f.write(f"Final Arm: {analysis['final_arm']} ({summary['arms'][analysis['final_arm']]})\n")
        f.write("\n")
        
        f.write("Arm Selection Counts:\n")
        for arm_id, count in analysis['arm_counts'].items():
            percentage = (count / len(turns)) * 100 if turns else 0
            f.write(f"  {arm_id} ({summary['arms'][arm_id]}): {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("CHOICE SEQUENCE\n")
        f.write("-" * 60 + "\n")
        choice_seq = analysis['choice_sequence']
        for i in range(0, len(choice_seq), 10):
            chunk = choice_seq[i:i+10]
            f.write(f"Turns {i+1:2d}-{min(i+10, len(choice_seq)):2d}: {' '.join(chunk)}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("PARSE ERRORS\n")
        f.write("-" * 60 + "\n")
        errors = [t for t in turns if t.get("parse_error")]
        if errors:
            for err_turn in errors:
                f.write(f"Turn {err_turn['turn_number']}: {err_turn['parse_error']}\n")
                f.write(f"  Response: {err_turn['model_response'][:100]}...\n\n")
        else:
            f.write("No parse errors.\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"[INFO] Saved text report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-armed bandit results")
    parser.add_argument("--result-dir", required=True, help="Directory containing results")
    parser.add_argument("--output-dir", help="Output directory for plots (default: same as result-dir)")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading results from {result_dir}...")
    results = load_results(result_dir)
    
    print(f"[INFO] Generating visualizations...")
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_choice_trajectory(results, output_dir / "choice_trajectory.png")
        plot_arm_distribution(results, output_dir / "arm_distribution.png")
        plot_cumulative_choices(results, output_dir / "cumulative_choices.png")
    else:
        print("[WARN] Skipping plots (matplotlib not available)")
    
    # Generate text report
    generate_report(results, output_dir / "report.txt")
    
    print(f"[INFO] Visualization complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()

