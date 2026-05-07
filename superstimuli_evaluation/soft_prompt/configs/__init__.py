"""Config path resolution for superstimuli_evaluation.soft_prompt.

superstimuli_evaluation.soft_prompt owns its own experiments.yaml and datasets.yaml.
models.yaml is shared from superstimuli_training/soft_prompt/assets/.

The ``.env`` file at the package root is loaded on import (via python-dotenv)
so that all eval scripts automatically pick up ``SOFT_PROMPT_BASE_DIR``,
``EVAL_OUTPUTS_DIR``, ``CONDA_ENV``, etc.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Load .env from the superstimuli_evaluation.soft_prompt root (one level up from configs/)
_EVAL_ROOT = Path(__file__).resolve().parent.parent
_DOT_ENV = _EVAL_ROOT / ".env"
if _DOT_ENV.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_DOT_ENV, override=False)
    except ImportError:
        # Fallback if python-dotenv not installed
        pass

# Output root directory for all eval results.
# Set EVAL_OUTPUTS_DIR in .env (e.g. EVAL_OUTPUTS_DIR=outputs_utility).
EVAL_OUTPUTS_DIR: str = os.environ.get("EVAL_OUTPUTS_DIR", "outputs")

_CONFIGS_DIR = Path(__file__).resolve().parent
_WELLBEING_DEV_ROOT = _CONFIGS_DIR.parents[2]
_WELLBEING_CONFIGS_DIR = _WELLBEING_DEV_ROOT / "wellbeing" / "configs"
_TRAINING_ASSETS_DIR = _WELLBEING_DEV_ROOT / "superstimuli_training" / "soft_prompt" / "assets"

# superstimuli_evaluation.soft_prompt configs
EXPERIMENTS_YAML = _CONFIGS_DIR / "experiments.yaml"
DATASETS_YAML = _CONFIGS_DIR / "datasets.yaml"

# Shared configs
MODELS_YAML = _TRAINING_ASSETS_DIR / "models.yaml"
WELLBEING_MODELS_YAML = _WELLBEING_CONFIGS_DIR / "models.yaml"
COMPUTE_UTILITIES_YAML = _WELLBEING_DEV_ROOT / "wellbeing" / "metrics" / "compute_utilities" / "compute_utilities.yaml"


def load_models(models_yaml: Optional[Path] = None) -> Dict[str, Any]:
    """Load the models dict from models.yaml.

    Returns the inner dict so callers can do ``load_models()[model_key]``.
    Supports both flat format and legacy ``models:`` wrapper.
    """
    path = models_yaml or MODELS_YAML
    with open(path) as f:
        data = yaml.safe_load(f)
    # Support both formats: top-level ``models:`` key or flat dict
    if "models" in data and isinstance(data["models"], dict):
        return data["models"]
    return data


def load_model_config(model_key: str, models_yaml: Optional[Path] = None) -> Dict[str, Any]:
    """Load config for a single model. Raises KeyError if not found."""
    models = load_models(models_yaml)
    if model_key not in models:
        path = models_yaml or MODELS_YAML
        raise KeyError(
            f"Model '{model_key}' not found in {path}. "
            f"Available: {list(models.keys())}"
        )
    return models[model_key]


def _sub_outputs_dir(obj: Any, outputs_dir: str) -> Any:
    """Recursively replace {outputs_dir} in all strings in a config object."""
    if isinstance(obj, str):
        return obj.replace("{outputs_dir}", outputs_dir)
    if isinstance(obj, dict):
        return {k: _sub_outputs_dir(v, outputs_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub_outputs_dir(v, outputs_dir) for v in obj]
    return obj


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """Load config for a specific experiment from experiments.yaml.

    All occurrences of ``{outputs_dir}`` in string values are replaced with
    the value of the ``EVAL_OUTPUTS_DIR`` environment variable (default: ``outputs``).
    """
    with open(EXPERIMENTS_YAML) as f:
        all_configs = yaml.safe_load(f)
    if experiment_name not in all_configs:
        raise KeyError(
            f"Experiment '{experiment_name}' not found in {EXPERIMENTS_YAML}. "
            f"Available: {list(all_configs.keys())}"
        )
    return _sub_outputs_dir(all_configs[experiment_name], EVAL_OUTPUTS_DIR)
