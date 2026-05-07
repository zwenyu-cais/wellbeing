"""Model loading and registry.

Reads models.yaml and provides a unified interface for loading vLLM
generator and judge models.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from transformers import AutoTokenizer
from vllm import LLM

DEFAULT_MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"


@dataclass
class ModelSpec:
    key: str
    path: str
    gpu_count: int
    model_name: str = ""
    model_type: str = "vllm"
    dtype: str = "bfloat16"
    max_model_len: int = 16384


def load_models_config(path: Path = DEFAULT_MODELS_YAML) -> Dict[str, ModelSpec]:
    """Load model registry from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Models config not found: {path}")
    data = yaml.safe_load(path.read_text())
    specs = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        model_path = entry.get("path") or entry.get("model_path") or entry.get("model_name", "")
        specs[key] = ModelSpec(
            key=key,
            path=model_path,
            gpu_count=int(entry.get("gpu_count", 1)),
            model_name=entry.get("model_name", key),
            model_type=entry.get("model_type", "vllm"),
        )
    return specs


def resolve_model(key_or_path: str, models_cfg: Dict[str, ModelSpec]) -> ModelSpec:
    """Resolve a model key or filesystem path to a ModelSpec.

    If key_or_path is in models_cfg, return that entry.
    If it's a filesystem path, infer gpu_count from the base model name
    (2 for 32B, 4 for 72B, default 2).
    """
    if key_or_path in models_cfg:
        return models_cfg[key_or_path]

    # Treat as filesystem path
    path = Path(key_or_path)
    if path.exists() or "/" in key_or_path:
        # Try to infer gpu_count from model name
        gpu_count = 2  # safe default
        # Check larger sizes first to avoid substring false positives
        # (e.g. "70b" contains "7b" if checked carelessly)
        path_lower = key_or_path.lower()
        size_match = re.search(r'(\d+)b', path_lower)
        if size_match:
            size_num = int(size_match.group(1))
            if size_num >= 65:
                gpu_count = 4
            elif size_num >= 20:
                gpu_count = 2
            elif size_num <= 8:
                gpu_count = 1
        return ModelSpec(
            key=path.stem if path.exists() else key_or_path,
            path=key_or_path,
            gpu_count=gpu_count,
            model_name=key_or_path,
        )

    raise ValueError(
        f"Model '{key_or_path}' not found in config and is not a valid path. "
        f"Available models: {list(models_cfg.keys())}"
    )


def load_vllm_model(spec: ModelSpec, max_model_len: Optional[int] = None):
    """Load a vLLM model and tokenizer.

    Returns (LLM, tokenizer) tuple.
    """
    # Use SLURM_GPUS_ON_NODE if available, otherwise spec.gpu_count
    tp = int(os.environ.get("SLURM_GPUS_ON_NODE", spec.gpu_count))

    model_len = max_model_len or spec.max_model_len

    print(f"Loading model: {spec.path}")
    print(f"  tensor_parallel_size={tp}, max_model_len={model_len}")

    llm = LLM(
        model=spec.path,
        tensor_parallel_size=tp,
        max_model_len=model_len,
        dtype=spec.dtype,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)

    # Disable thinking mode (Qwen3.5+ defaults enable_thinking=True in its
    # chat template, which wastes tokens on <think> reasoning).  For models
    # whose template doesn't use the variable, Jinja2 silently ignores it.
    _orig_apply = tokenizer.apply_chat_template

    def _apply_no_thinking(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_apply(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_no_thinking

    return llm, tokenizer
