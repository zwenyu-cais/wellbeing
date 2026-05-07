"""Generate combination options from baseline text strings for zero-point fitting.

Matches the combination generation procedure from
wellbeing/datasets/experiences/component_datasets/d2d3/prepare_options.py:
400 combinations with stratified sizes (2x160, 3x120, 4x120).
"""

import random
from typing import Dict, List, Tuple

COMBINATION_SIZES: List[Tuple[int, int]] = [(2, 160), (3, 120), (4, 120)]


def generate_combinations(
    baseline_options: List[Dict],
    combination_sizes: List[Tuple[int, int]] = COMBINATION_SIZES,
    seed: int = 42,
) -> List[Dict]:
    """Generate combination options by bundling baseline options.

    Args:
        baseline_options: List of baseline option dicts (must have 'id' and 'description').
        combination_sizes: List of (combo_size, count) tuples.
        seed: Random seed for reproducibility.

    Returns:
        List of combination option dicts (without 'id' assigned — caller sets IDs).
    """
    rng = random.Random(seed)
    n_individual = len(baseline_options)
    combinations = []

    for combo_size, count in combination_sizes:
        for _ in range(count):
            sampled_indices = rng.sample(range(n_individual), combo_size)
            components = [baseline_options[i] for i in sampled_indices]

            desc_parts = [
                f"The following bundle contains {combo_size} individual experiences."
            ]
            for k, comp in enumerate(components, 1):
                desc_parts.append(
                    f"---------- Experience {k} of {combo_size} ----------\n"
                    f"{comp['description']}"
                )
            description = "\n\n".join(desc_parts)

            combinations.append({
                "description": description,
                "category": "combination",
                "source": "combination",
                "component_ids": [comp["id"] for comp in components],
                "size": combo_size,
            })

    return combinations
