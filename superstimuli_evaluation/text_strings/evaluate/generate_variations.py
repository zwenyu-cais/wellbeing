"""Generate paraphrase and clause-reordered variations of RL-discovered strings.

For each RL string, produces two variations:
- paraphrase: same meaning, different wording
- clause_reorder: clauses rearranged, meaning preserved

Variations are cached per model+experiment to avoid regeneration and prevent
overwrites across different runs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

VARIATION_TYPES = ["paraphrase", "clause_reorder"]

VARIATION_PROMPTS = {
    "paraphrase": (
        "Rewrite the following sentence to convey exactly the same meaning "
        "using different words and phrasing. Do not add or remove any information. "
        "Output only the rewritten sentence, nothing else.\n\n"
        "Sentence: {text}"
    ),
    "clause_reorder": (
        "Rearrange the clauses of the following sentence so the information "
        "appears in a different order, but the overall meaning is preserved. "
        "Keep the original words and phrasing as much as possible — only "
        "change the order, not the wording. "
        "Output only the rewritten sentence, nothing else.\n\n"
        "Sentence: {text}"
    ),
}


def generate_variations_for_text(
    text: str,
    agent,
    variation_types: List[str] = VARIATION_TYPES,
) -> Dict[str, str]:
    """Generate all variation types for a single text string.

    Args:
        text: The original RL string.
        agent: LLM agent with a ``completions(messages)`` method.
        variation_types: Which variation types to generate.

    Returns:
        Dict mapping variation_type -> variation_text.
    """
    variations: Dict[str, str] = {}
    for vtype in variation_types:
        if vtype not in VARIATION_PROMPTS:
            raise ValueError(f"Unknown variation type: {vtype}")
        prompt = VARIATION_PROMPTS[vtype].format(text=text)
        messages = [{"role": "user", "content": prompt}]
        response = agent.completions(messages)
        variations[vtype] = response.strip().strip("\"'")
    return variations


def load_variation_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    """Load cached variations from disk.

    Returns dict mapping original text -> {variation_type: variation_text}.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_variation_cache(cache: Dict[str, Dict[str, str]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_variations(
    rl_options: List[Dict],
    cache_path: Path,
    agent=None,
    variation_types: List[str] = VARIATION_TYPES,
) -> List[Dict]:
    """Generate variation options for RL strings.

    When *agent* is ``None`` only cached variations are returned. If a string
    is missing from the cache an error is raised — this forces offline
    pre-generation via ``generate_variations_cache.py``.

    Args:
        rl_options: RL option dicts (must have 'description' and 'category').
        cache_path: Path to the per-model+experiment cache JSON.
        agent: Optional LLM agent for on-the-fly generation.
        variation_types: Which variation types to include.

    Returns:
        List of variation option dicts (without 'id' — caller assigns IDs).
    """
    cache = load_variation_cache(cache_path)
    updated = False
    variations: List[Dict] = []

    for opt in rl_options:
        text = opt["description"]

        if text in cache:
            var_texts = cache[text]
        elif agent is not None:
            var_texts = generate_variations_for_text(text, agent, variation_types)
            cache[text] = var_texts
            updated = True
        else:
            raise RuntimeError(
                f"Variation cache missing for string (first 80 chars): "
                f"{text[:80]!r}...\n"
                f"Run `python -m evaluate.generate_variations_cache` first."
            )

        for vtype in variation_types:
            if vtype not in var_texts:
                continue
            variations.append({
                "description": var_texts[vtype],
                "category": opt.get("category", ""),
                "source": "variation",
                "variation_type": vtype,
                "parent_description": text,
                "parent_category": opt.get("category", ""),
            })

    if updated:
        save_variation_cache(cache, cache_path)

    return variations
