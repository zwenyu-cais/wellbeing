"""Model loading utilities. Single source of truth: models.yaml (symlinked from emergent-values)."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Union

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_YAML = os.path.join(_BASE_DIR, "configs", "models.yaml")


def load_models_config(models_config_path: Optional[Union[str, Path]] = None) -> Dict[str, Dict]:
    """Load all model configurations from models.yaml."""
    path = str(models_config_path) if models_config_path else _MODELS_YAML
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_config(
    model_key: str,
    models_config_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """Get configuration for a specific model."""
    models = load_models_config(models_config_path)
    if model_key not in models:
        available = sorted(models.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available[:10]}...")
    return models[model_key]


def get_model_type(
    model_key: str,
    models_config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Get the model type (openai, anthropic, vllm, etc.)."""
    return get_model_config(model_key, models_config_path)["model_type"]


def list_models_by_type() -> Dict[str, list]:
    """Group all models by their type."""
    models = load_models_config()
    by_type = {}
    for name, config in models.items():
        model_type = config["model_type"]
        by_type.setdefault(model_type, []).append(name)
    return {k: sorted(v) for k, v in sorted(by_type.items())}
