"""Runs map loading and soft prompt resolution utilities.

Loads a runs_map JSON (model_map + method_map), resolves system prompts
per model, and finds the best soft prompt run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .find_best_run import find_best_run

# Default runs_map lives alongside this file
DEFAULT_RUNS_MAP = Path(__file__).resolve().parent / "runs_map.json"

# Map CLI stimulant type names to runs_map method_map key prefixes.
# runs_map.json uses "Euphorics_<model>" keys.
_TYPE_TO_RUNS_MAP_KEY = {
    "euphorics": "Euphorics",
}


# ── Runs map utilities ──────────────────────────────────────────────────


def load_runs_map(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load runs_map JSON."""
    if path is None:
        path = DEFAULT_RUNS_MAP
    with open(path) as f:
        return json.load(f)


def get_system_prompts(runs_map: Dict[str, Any], model: str) -> Dict[str, str]:
    """Get system prompts for a model from runs_map.

    Returns:
        Dict with 'system_prompt_text' (contains [candidate_0] placeholder)
        and 'system_prompt_text_base' (no placeholder, used for baseline).
    """
    model_entry = runs_map.get("model_map", {}).get(model)
    if model_entry is None:
        raise KeyError(f"Model '{model}' not found in runs_map model_map")
    return {
        "system_prompt_text": model_entry["system_prompt_text"],
        "system_prompt_text_base": model_entry["system_prompt_text_base"],
        "soft_prompt_placement": model_entry.get("soft_prompt_placement", "system_prompt"),
    }


def get_model_display_name(model: str) -> str:
    """Get the display name for a model from models.yaml.

    Uses the ``model_name`` field directly.
    Falls back to the model key if not found.
    """
    from superstimuli_evaluation.soft_prompt.configs import load_models

    models = load_models()
    model_entry = models.get(model)
    if model_entry is None:
        return model
    return model_entry.get("model_name", model)


def resolve_sp_system_prompt(
    system_prompt_text: str,
    sp_token_str: str,
) -> str:
    """Replace [candidate_0] placeholder in system prompt with actual SP tokens.

    Args:
        system_prompt_text: System prompt template with [candidate_0] placeholder.
        sp_token_str: Actual SP token string, e.g. '<sp_0><sp_1><sp_2><sp_3>'.

    Returns:
        System prompt with placeholder replaced.
    """
    return system_prompt_text.replace("[candidate_0]", sp_token_str)


# ── Resolve soft prompt path (combines runs_map + find_best_run) ────────


def _resolve_soft_prompt_runs(
    runs_map: Dict[str, Any],
    model: str,
    stimulant_type: str,
    soft_prompt_base_dir: str,
    top_runs: int = 1,
) -> list:
    """Find the top soft prompt run directories for a model + stimulant type.

    Returns:
        List of absolute path strings to run directories, ranked best-first.
    """
    type_key = _TYPE_TO_RUNS_MAP_KEY.get(stimulant_type)
    if type_key is None:
        raise ValueError(
            f"Unknown stimulant_type '{stimulant_type}'. "
            f"Expected one of: {list(_TYPE_TO_RUNS_MAP_KEY.keys())}"
        )
    method_key = f"{type_key}_{model}"

    entry = runs_map.get("method_map", {}).get(method_key)
    if entry is None:
        raise KeyError(f"No entry for '{method_key}' in runs_map method_map")

    save_name = entry["save_name"]
    model_name = entry.get("model_name", model)
    sweep_dir = Path(soft_prompt_base_dir) / save_name / model_name

    # Build threshold filters
    _SPECIAL_KEYS = {"g_magnitude_threshold"}
    thresholds = {}
    min_thresholds = {}
    for key, val in entry.items():
        if key.endswith("_threshold") and key not in _SPECIAL_KEYS:
            base_name = key[: -len("_threshold")]
            field = base_name + "_at_best_checkpoint_so_far"
            direction = entry.get(base_name + "_direction", "lower_is_better")
            if direction == "higher_is_better":
                min_thresholds[field] = val
            else:
                thresholds[field] = val

    g_mag = entry.get("g_magnitude_threshold")

    best_runs = find_best_run(
        sweep_dir,
        hyperparameters=[],
        thresholds=thresholds or None,
        min_thresholds=min_thresholds or None,
        g_magnitude_threshold=g_mag,
        top_runs=top_runs,
    )
    if not best_runs:
        raise RuntimeError(
            f"find_best_run returned no results for {sweep_dir}"
        )

    return [str(sweep_dir / r) for r in best_runs]


def resolve_soft_prompt_path(
    runs_map: Dict[str, Any],
    model: str,
    stimulant_type: str,
    soft_prompt_base_dir: str,
) -> str:
    """Find the best soft prompt run directory for a model + stimulant type."""
    return _resolve_soft_prompt_runs(runs_map, model, stimulant_type, soft_prompt_base_dir, top_runs=1)[0]


def resolve_soft_prompt_paths(
    runs_map: Dict[str, Any],
    model: str,
    stimulant_type: str,
    soft_prompt_base_dir: str,
    top_runs: int = 1,
) -> list:
    """Find the top N soft prompt run directories for a model + stimulant type.

    Returns:
        List of absolute path strings, ranked best-first.
    """
    return _resolve_soft_prompt_runs(runs_map, model, stimulant_type, soft_prompt_base_dir, top_runs=top_runs)
