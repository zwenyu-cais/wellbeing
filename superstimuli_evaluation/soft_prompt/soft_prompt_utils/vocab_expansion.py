"""Vocab expansion approach for soft prompt injection via vLLM.

Creates a modified model directory with soft prompt embeddings baked into
unused token slots in the embedding layer.  The model can then be served
by standard vLLM **without** ``--enable-prompt-embeds``.

Supports five model families:

- **Qwen3.5-27B**: adds ``<sp_N>`` tokens to unused slots (248077+)
- **Qwen3.5-35B-A3B**: adds ``<sp_N>`` tokens to unused slots (248077+)
- **Qwen3-30B-A3B**: adds ``<sp_N>`` tokens to unused slots (151669+)
- **Llama-3.3-70B**: reuses existing ``<|reserved_special_token_N|>`` tokens
- **Gemma-3-27B**: adds ``<sp_N>`` tokens (262145+) and unties embeddings

Usage::

    from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
        prepare_expanded_model,
        build_prompt_token_ids,
    )

    ve = prepare_expanded_model("qwen35-27b", sp_tensor)
    # Start vLLM with ve.modified_dir, then build payloads:
    token_ids = build_prompt_token_ids(prompt_text, tokenizer, ve.sp_token_ids)
    # Send token_ids to vLLM with logit_bias=ve.sp_logit_bias
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Per-model configuration
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_EXPANSION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen35-27b": {
        "strategy": "unused_slots",
        "unused_token_start": 248077,
        "untie_embeddings": False,
    },
    "qwen35-27b-thinking": {
        "strategy": "unused_slots",
        "unused_token_start": 248077,
        "untie_embeddings": False,
    },
    "qwen35-35b-a3b": {
        "strategy": "unused_slots",
        "unused_token_start": 248077,
        "untie_embeddings": False,
    },
    "qwen3-30b-a3b-instruct": {
        "strategy": "unused_slots",
        "unused_token_start": 151669,
        "untie_embeddings": False,
    },
    "llama-33-70b-instruct": {
        "strategy": "reserved_tokens",
        "untie_embeddings": False,
    },
    "gemma-3-27b-it": {
        "strategy": "unused_slots",
        "unused_token_start": 262145,
        "untie_embeddings": True,
    },
    "gemma-4-31b-it": {
        "strategy": "unused_slots",
        "unused_token_start": 262145,
        "untie_embeddings": True,
    },
    "olmo-31-32b-instruct": {
        "strategy": "unused_slots",
        "unused_token_start": 100279,
        "untie_embeddings": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VocabExpansionResult:
    """Result of preparing a vocab-expanded model."""

    modified_dir: str
    sp_token_ids: List[int]
    sp_token_names: List[str]
    sp_logit_bias: Dict[str, int]
    n_sp_tokens: int


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_KNOWN_CONDITIONS = ("euphorics",)


def _condition_from_path(sp_path: str) -> str:
    """Extract condition name from a soft prompt run path."""
    for part in Path(sp_path).parts:
        for cond in _KNOWN_CONDITIONS:
            if cond in part:
                return cond
    return "unknown"


def _sp_tensor_hash(sp_tensor: torch.Tensor) -> str:
    """Compute a short hash of a soft prompt tensor for caching."""
    data = sp_tensor.cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:12]


def _default_cache_dir() -> Path:
    base = os.environ.get("WELLBEING_EVALS_CACHE_DIR")
    if base:
        return Path(base) / "vocab_expansion"
    return Path.home() / ".cache" / "wellbeing_evals" / "vocab_expansion"


def _find_embed_in_index(model_path: str) -> Tuple[Path, str]:
    """Find embed_tokens weight file and key from model.safetensors.index.json."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No model.safetensors.index.json at {model_path}")
    with open(index_path) as f:
        index = json.load(f)
    for key, filename in index["weight_map"].items():
        if "embed_tokens" in key and "weight" in key:
            return Path(model_path) / filename, key
    raise RuntimeError(f"Could not find embed_tokens in model index at {model_path}")


def _get_lm_head_key(embed_key: str) -> str:
    """Derive lm_head weight key from embed_tokens key.

    e.g. ``language_model.model.embed_tokens.weight`` →
         ``language_model.lm_head.weight``
    """
    parts = embed_key.split(".")
    model_idx = parts.index("model") if "model" in parts else -1
    if model_idx >= 0:
        return ".".join(parts[:model_idx] + ["lm_head", "weight"])
    return "lm_head.weight"


def _make_result(
    modified_dir: str,
    sp_token_ids: List[int],
    sp_token_names: List[str],
) -> VocabExpansionResult:
    return VocabExpansionResult(
        modified_dir=modified_dir,
        sp_token_ids=sp_token_ids,
        sp_token_names=sp_token_names,
        sp_logit_bias={str(tid): -100 for tid in sp_token_ids},
        n_sp_tokens=len(sp_token_ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Preparation: unused slots (Qwen, Gemma)
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_unused_slots(
    model_key: str,
    model_path: str,
    modified_dir: str,
    sp_tensor: torch.Tensor,
    ve_config: Dict[str, Any],
) -> VocabExpansionResult:
    from transformers import AutoTokenizer
    from safetensors.torch import load_file, save_file

    n_sp_tokens = sp_tensor.shape[0]
    unused_start = ve_config["unused_token_start"]
    sp_token_ids = list(range(unused_start, unused_start + n_sp_tokens))
    sp_token_names = [f"<sp_{i}>" for i in range(n_sp_tokens)]

    # Add SP tokens to tokenizer and save
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.add_tokens(sp_token_names, special_tokens=True)
    actual_ids = tokenizer.convert_tokens_to_ids(sp_token_names)
    assert actual_ids == sp_token_ids, (
        f"Token ID mismatch for {model_key}: expected {sp_token_ids[:3]}..., "
        f"got {actual_ids[:3]}... "
        f"Tokenizer vocab ({len(tokenizer) - n_sp_tokens}) != expected ({unused_start})"
    )
    tokenizer.save_pretrained(modified_dir)

    # Symlink all original model files (tokenizer files already written above)
    for item in Path(model_path).iterdir():
        dest = Path(modified_dir) / item.name
        if not dest.exists():
            os.symlink(item, dest)

    # Load and modify embeddings
    embed_file, embed_key = _find_embed_in_index(model_path)
    data = load_file(str(embed_file), device="cpu")
    original_embed = data[embed_key]
    modified_embed = original_embed.clone()
    sp_embed = sp_tensor.to(modified_embed.dtype)

    for i, token_id in enumerate(sp_token_ids):
        modified_embed[token_id] = sp_embed[i]

    save_data = {embed_key: modified_embed}

    # Handle untied embeddings (Gemma): separate lm_head with SP rows zeroed
    if ve_config.get("untie_embeddings"):
        lm_head_key = _get_lm_head_key(embed_key)
        lm_head_weight = original_embed.clone()
        for token_id in sp_token_ids:
            lm_head_weight[token_id] = 0.0
        save_data[lm_head_key] = lm_head_weight
        print(f"[vocab_expansion] Created untied lm_head: {lm_head_key} (SP rows zeroed)")

        # Update config.json
        config_dest = Path(modified_dir) / "config.json"
        if config_dest.is_symlink():
            config_dest.unlink()
        with open(Path(model_path) / "config.json") as f:
            config = json.load(f)
        config["tie_word_embeddings"] = False
        if "text_config" in config:
            config["text_config"]["tie_word_embeddings"] = False
        with open(config_dest, "w") as f:
            json.dump(config, f, indent=2)

    # Save modified embeddings
    modified_file = Path(modified_dir) / "modified_embeddings.safetensors"
    save_file(save_data, str(modified_file))

    # Update model index to point embed_tokens (and lm_head if untied) to modified file
    idx_dest = Path(modified_dir) / "model.safetensors.index.json"
    if idx_dest.is_symlink():
        idx_dest.unlink()
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"][embed_key] = "modified_embeddings.safetensors"
    if ve_config.get("untie_embeddings"):
        lm_head_key = _get_lm_head_key(embed_key)
        index["weight_map"][lm_head_key] = "modified_embeddings.safetensors"
    with open(idx_dest, "w") as f:
        json.dump(index, f, indent=2)

    print(
        f"[vocab_expansion] {n_sp_tokens} SP tokens written to IDs "
        f"{sp_token_ids[0]}-{sp_token_ids[-1]}"
    )
    return _make_result(modified_dir, sp_token_ids, sp_token_names)


# ─────────────────────────────────────────────────────────────────────────────
# Preparation: reserved tokens (Llama)
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_reserved_tokens(
    model_path: str,
    modified_dir: str,
    sp_tensor: torch.Tensor,
    ve_config: Dict[str, Any],
) -> VocabExpansionResult:
    from transformers import AutoTokenizer
    from safetensors.torch import load_file, save_file

    n_sp_tokens = sp_tensor.shape[0]

    # Find reserved tokens in existing vocab
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    reserved = sorted(
        [(v, k) for k, v in vocab.items() if "reserved_special_token" in k]
    )
    assert len(reserved) >= n_sp_tokens, (
        f"Need {n_sp_tokens} reserved tokens but only found {len(reserved)}"
    )
    sp_token_ids = [tid for tid, _ in reserved[:n_sp_tokens]]
    sp_token_names = [name for _, name in reserved[:n_sp_tokens]]

    # Symlink all original model files (tokenizer unchanged)
    for item in Path(model_path).iterdir():
        dest = Path(modified_dir) / item.name
        if not dest.exists():
            os.symlink(item, dest)

    # Modify embeddings
    embed_file, embed_key = _find_embed_in_index(model_path)
    data = load_file(str(embed_file), device="cpu")
    original_embed = data[embed_key]
    modified_embed = original_embed.clone()
    sp_embed = sp_tensor.to(modified_embed.dtype)

    for i, token_id in enumerate(sp_token_ids):
        modified_embed[token_id] = sp_embed[i]

    modified_file = Path(modified_dir) / "modified_embeddings.safetensors"
    save_file({embed_key: modified_embed}, str(modified_file))

    # Update model index
    idx_dest = Path(modified_dir) / "model.safetensors.index.json"
    if idx_dest.is_symlink():
        idx_dest.unlink()
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"][embed_key] = "modified_embeddings.safetensors"
    with open(idx_dest, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[vocab_expansion] {n_sp_tokens} SP tokens using reserved tokens")
    return _make_result(modified_dir, sp_token_ids, sp_token_names)


# ─────────────────────────────────────────────────────────────────────────────
# Cache loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_cached_result(
    modified_dir: str,
    n_sp_tokens: int,
    ve_config: Dict[str, Any],
) -> VocabExpansionResult:
    """Reconstruct a VocabExpansionResult from a previously prepared directory."""
    strategy = ve_config["strategy"]

    if strategy == "reserved_tokens":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(modified_dir, trust_remote_code=True)
        vocab = tokenizer.get_vocab()
        reserved = sorted(
            [(v, k) for k, v in vocab.items() if "reserved_special_token" in k]
        )
        sp_token_ids = [tid for tid, _ in reserved[:n_sp_tokens]]
        sp_token_names = [name for _, name in reserved[:n_sp_tokens]]
    else:
        unused_start = ve_config["unused_token_start"]
        sp_token_ids = list(range(unused_start, unused_start + n_sp_tokens))
        sp_token_names = [f"<sp_{i}>" for i in range(n_sp_tokens)]

    print(f"[vocab_expansion] Using cached expanded model: {modified_dir}")
    return _make_result(modified_dir, sp_token_ids, sp_token_names)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: prepare_expanded_model
# ─────────────────────────────────────────────────────────────────────────────


def prepare_expanded_model(
    model_key: str,
    sp_tensor: torch.Tensor,
    output_dir: Optional[str] = None,
    sp_path: Optional[str] = None,
) -> VocabExpansionResult:
    """Create a modified model directory with SP embeddings in unused token slots.

    The result directory can be served by vLLM as-is.  Subsequent calls with the
    same ``model_key`` and ``sp_tensor`` reuse the cached directory.

    Args:
        model_key: Key from models.yaml (e.g. ``"qwen35-27b"``).
        sp_tensor: Soft prompt tensor ``(n_tokens, hidden_dim)``.
        output_dir: Optional explicit output directory.  Auto-generated
            under ``~/.cache/wellbeing_evals/vocab_expansion/`` if *None*.

    Returns:
        :class:`VocabExpansionResult` with ``modified_dir``, ``sp_token_names``,
        ``sp_logit_bias``, etc.
    """
    from superstimuli_evaluation.soft_prompt.configs import load_model_config

    ve_config = VOCAB_EXPANSION_CONFIGS.get(model_key)
    if ve_config is None:
        raise ValueError(
            f"No vocab expansion config for model '{model_key}'. "
            f"Available: {list(VOCAB_EXPANSION_CONFIGS.keys())}"
        )

    model_config = load_model_config(model_key)
    model_path = model_config["path"]
    n_sp_tokens = sp_tensor.shape[0]
    sp_hash = _sp_tensor_hash(sp_tensor)

    # Determine output directory
    if output_dir is None:
        cache_dir = _default_cache_dir()
        if sp_path is not None:
            run_id = Path(sp_path).name
            condition = _condition_from_path(sp_path)
            output_dir = str(cache_dir / f"{model_key}_{condition}_{run_id}")
        else:
            output_dir = str(cache_dir / f"{model_key}_{sp_hash}")

    modified_dir = output_dir

    # Check cache
    marker = Path(modified_dir) / ".vocab_expansion_done"
    if marker.exists():
        try:
            meta = json.loads(marker.read_text())
            if (
                meta.get("model_key") == model_key
                and meta.get("n_sp_tokens") == n_sp_tokens
                and meta.get("sp_hash") == sp_hash
            ):
                return _load_cached_result(modified_dir, n_sp_tokens, ve_config)
        except (json.JSONDecodeError, KeyError):
            pass
        # Invalid cache — remove and re-prepare
        import shutil

        shutil.rmtree(modified_dir, ignore_errors=True)

    print(f"[vocab_expansion] Preparing expanded model at: {modified_dir}")
    os.makedirs(modified_dir, exist_ok=True)

    strategy = ve_config["strategy"]
    if strategy == "reserved_tokens":
        result = _prepare_reserved_tokens(model_path, modified_dir, sp_tensor, ve_config)
    else:
        result = _prepare_unused_slots(
            model_key, model_path, modified_dir, sp_tensor, ve_config
        )

    # Write cache marker
    marker.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "n_sp_tokens": n_sp_tokens,
                "sp_hash": sp_hash,
            }
        )
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API: token-level placeholder replacement (matches training)
# ─────────────────────────────────────────────────────────────────────────────


def build_prompt_token_ids(
    prompt_text: str,
    tokenizer: Any,
    sp_token_ids: List[int],
) -> List[int]:
    """Tokenize *prompt_text* and replace all ``[candidate_0]`` token spans
    with *sp_token_ids*.

    This mirrors the transformers training approach exactly: tokenize the
    prompt containing the literal ``[candidate_0]`` text, locate the
    placeholder(s) via character-offset mapping, then splice in the SP token
    IDs at the token level.  Surrounding tokens are untouched, so
    tokenization boundaries are preserved.

    If no ``[candidate_0]`` is found the token IDs are returned as-is
    (baseline / no-SP case).
    """
    from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
        find_indexed_placeholder_spans_via_offsets,
        candidate_placeholder_for_index,
    )

    # Count how many [candidate_0] occurrences exist in the prompt
    placeholder = candidate_placeholder_for_index(0)
    num_occurrences = prompt_text.count(placeholder)
    if num_occurrences == 0:
        return list(tokenizer(prompt_text)["input_ids"])

    inputs = tokenizer(prompt_text, return_offsets_mapping=True)
    token_ids = list(inputs["input_ids"])
    offset_mapping = [(int(s), int(e)) for s, e in inputs["offset_mapping"]]

    spans = find_indexed_placeholder_spans_via_offsets(
        prompt_text, offset_mapping, max_index=num_occurrences
    )
    if not spans:
        return token_ids

    # Replace all spans from right to left to preserve earlier positions
    sp = list(sp_token_ids)
    for span_start, span_end in reversed(spans):
        token_ids = token_ids[:span_start] + sp + token_ids[span_end:]
    return token_ids



# ─────────────────────────────────────────────────────────────────────────────
# Public API: agent wrapper for compute_utilities
# ─────────────────────────────────────────────────────────────────────────────


class VocabExpansionAgentWrapper:
    """Agent wrapper matching the vLLMSoftPromptAgent / TransformersAgentWrapper
    interface, using vocab-expanded vLLM for inference.

    Used by ``preference_retain`` and ``run_eu`` evals which pass an ``agent``
    to :func:`compute_utilities`.
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        ve_result: Optional[VocabExpansionResult] = None,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 10,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
    ):
        from transformers import AutoTokenizer
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
            normalize_api_url,
        )

        if ve_result is None and model_dir is None:
            raise ValueError("Either ve_result or model_dir must be provided.")

        self.api_url = normalize_api_url(api_url)
        self.model_name = model_name
        self.ve_result = ve_result
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.accepts_system_message = True
        _dir = ve_result.modified_dir if ve_result is not None else model_dir
        self.model = _dir
        self.tokenizer = AutoTokenizer.from_pretrained(_dir, trust_remote_code=True)

    def _build_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """Build prompt text with ``[candidate_0]`` still present."""
        user_messages = [m for m in messages if m.get("role") != "system"]
        if self.system_prompt:
            full_messages = [{"role": "system", "content": self.system_prompt}] + user_messages
        else:
            full_messages = user_messages
        return self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def _build_prompt_token_ids(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Build token IDs with ``[candidate_0]`` replaced at token level.

        When ``ve_result`` is None (baseline mode), returns plain token IDs
        with no placeholder replacement.
        """
        prompt_text = self._build_prompt_text(messages)
        if self.ve_result is None:
            return list(self.tokenizer(prompt_text)["input_ids"])
        return build_prompt_token_ids(
            prompt_text, self.tokenizer, self.ve_result.sp_token_ids
        )

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        resp = requests.post(
            f"{self.api_url}/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    async def _async_post_with_retry(
        self,
        session: Any,
        payload: Dict[str, Any],
        max_retries: int = 3,
        timeout: float = 300,
    ) -> Dict[str, Any]:
        import aiohttp

        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.api_url}/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Request failed ({e}), retrying... ({attempt+1}/{max_retries})")
                    import asyncio
                    await asyncio.sleep(wait)
                else:
                    raise

    def _make_payload(
        self,
        token_ids: List[int],
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": token_ids,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        if self.ve_result is not None:
            payload["logit_bias"] = self.ve_result.sp_logit_bias
        if logprobs is not None:
            payload["logprobs"] = logprobs
        return payload

    def completions(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        token_ids = self._build_prompt_token_ids(messages)
        payload = self._make_payload(token_ids, max_tokens=kwargs.get("max_tokens"))
        result = self._post(payload)
        return result["choices"][0]["text"].strip()

    def _completions(self, messages, **kwargs) -> str:
        return self.completions(messages, **kwargs)

    def completions_batch(
        self, messages_list: List[List[Dict[str, Any]]], **kwargs
    ) -> List[str]:
        return [self.completions(m, **kwargs) for m in messages_list]

    def _completions_batch(self, messages_list, **kwargs) -> List[str]:
        return self.completions_batch(messages_list, **kwargs)

    async def async_completions_batch(
        self,
        messages: List[List[Dict[str, Any]]],
        concurrency: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> List[str]:
        import asyncio
        import aiohttp

        sem = asyncio.Semaphore(concurrency or 32)
        total = len(messages)
        completed = [0]
        results: List[Optional[str]] = [None] * total

        async def _do(idx: int, session: aiohttp.ClientSession):
            token_ids = self._build_prompt_token_ids(messages[idx])
            payload = self._make_payload(token_ids, max_tokens=kwargs.get("max_tokens"))
            async with sem:
                data = await self._async_post_with_retry(session, payload)
                results[idx] = data["choices"][0]["text"].strip()
                completed[0] += 1
                if verbose and (completed[0] % 50 == 0 or completed[0] == total):
                    print(f"  {completed[0]}/{total} completed")

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[_do(i, session) for i in range(total)])
        return results  # type: ignore[return-value]

    async def async_completions_batch_with_logprobs(
        self,
        messages: List[List[Dict[str, Any]]],
        top_K: int = 5,
        max_tokens: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        import asyncio
        import aiohttp

        sem = asyncio.Semaphore(32)
        total = len(messages)
        completed = [0]
        results: List[Any] = [None] * total

        async def _do(idx: int, session: aiohttp.ClientSession):
            token_ids = self._build_prompt_token_ids(messages[idx])
            payload = self._make_payload(
                token_ids, max_tokens=max_tokens, logprobs=top_K
            )
            async with sem:
                data = await self._async_post_with_retry(session, payload)
                choice = data["choices"][0]
                text = choice["text"].strip()
                lp_data = choice.get("logprobs", {})
                top_logprobs = lp_data.get("top_logprobs", [{}])
                token_lps: List[Tuple[str, float]] = []
                if top_logprobs:
                    for tok, lp_val in top_logprobs[0].items():
                        token_lps.append((tok, lp_val))
                results[idx] = (text, token_lps)
                completed[0] += 1
                if verbose and (completed[0] % 50 == 0 or completed[0] == total):
                    print(f"  {completed[0]}/{total} completed")

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[_do(i, session) for i in range(total)])
        return results  # type: ignore[return-value]

    async def async_completions(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        import aiohttp

        token_ids = self._build_prompt_token_ids(messages)
        payload = self._make_payload(token_ids, max_tokens=kwargs.get("max_tokens"))
        async with aiohttp.ClientSession() as session:
            data = await self._async_post_with_retry(session, payload)
            return data["choices"][0]["text"].strip()
