"""
Soft prompt inference via vLLM HTTP API with client-side embedding injection.

Provides SoftPromptLLMProxy — a drop-in replacement for vLLM's LLM.generate()
that injects soft prompt embeddings into requests sent to a vLLM server
started with --enable-prompt-embeds.

The embedding injection code is self-contained with no external dependencies.

Usage:
    proxy = SoftPromptLLMProxy(
        server_url="http://localhost:8000",
        model_path="/path/to/model",
        soft_prompt_path="/path/to/sp_run",
    )
    tokenizer = proxy.get_tokenizer()
    outputs = proxy.generate(prompts, sampling_params)
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder utilities
# ─────────────────────────────────────────────────────────────────────────────


def candidate_placeholder_for_index(k: int) -> str:
    """Return the placeholder string ``[candidate_k]`` for index *k* (0-based)."""
    if k < 0:
        raise ValueError(f"Candidate index must be >= 0, got {k}")
    return f"[candidate_{k}]"


# ─────────────────────────────────────────────────────────────────────────────
# Span detection via character offsets
# ─────────────────────────────────────────────────────────────────────────────


def find_indexed_placeholder_spans_via_offsets(
    conv_str: str,
    offset_mapping: List[Tuple[int, int]],
    max_index: int,
) -> List[Tuple[int, int]]:
    """Find ``[candidate_0]``, ``[candidate_1]``, ... spans and map to token spans.

    Tokenisers often split ``[candidate_0]`` differently depending on context,
    so token-ID search can fail.  We locate placeholders in the raw string and
    map character ranges to token spans via *offset_mapping*.

    Returns list of ``(token_start, token_end)`` spans, or empty list if not
    all *max_index* placeholders were found.
    """
    conv_str_norm = conv_str.replace("[candidate]", "[candidate_0]")

    occurrences: List[Tuple[int, int]] = []
    for k in range(max_index):
        placeholder = candidate_placeholder_for_index(k)
        pos = 0
        while True:
            idx = conv_str_norm.find(placeholder, pos)
            if idx < 0:
                break
            occurrences.append((idx, idx + len(placeholder)))
            pos = idx + 1

    occurrences.sort(key=lambda x: x[0])
    if len(occurrences) != max_index:
        return []

    spans: List[Tuple[int, int]] = []
    num_tokens = len(offset_mapping)
    for char_start, char_end in occurrences:
        first, last = None, None
        for j in range(num_tokens):
            start, end = offset_mapping[j]
            if start == 0 and end == 0:
                continue
            if end > char_start and start < char_end:
                if first is None:
                    first = j
                last = j
        if first is not None and last is not None:
            spans.append((first, last + 1))
        else:
            return []
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Single-embedding injection
# ─────────────────────────────────────────────────────────────────────────────


def inject_single_embedding(
    base_emb: torch.Tensor,
    base_mask: torch.Tensor,
    candidate_emb: torch.Tensor,
    placeholder_span: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace a placeholder span's embeddings with *candidate_emb*.

    The sequence may shrink or expand if the embedding length differs from
    the placeholder span length.
    """
    start, end = placeholder_span
    seq_len = base_emb.shape[0]
    if start < 0 or end < 0 or start >= end or start > seq_len or end > seq_len:
        raise ValueError(
            f"Invalid placeholder span ({start}, {end}) for seq_len={seq_len}"
        )

    emb_len = candidate_emb.shape[0]
    candidate_emb = candidate_emb.to(device=device, dtype=base_emb.dtype)

    before = base_emb[:start]
    after = base_emb[end:]
    modified_emb = torch.cat([before, candidate_emb, after], dim=0)

    before_mask = base_mask[:start]
    after_mask = base_mask[end:]
    emb_mask = torch.ones(emb_len, device=device, dtype=base_mask.dtype)
    modified_mask = torch.cat([before_mask, emb_mask, after_mask], dim=0)

    if modified_mask.shape[0] < modified_emb.shape[0]:
        pad_len = modified_emb.shape[0] - modified_mask.shape[0]
        modified_mask = torch.cat(
            [modified_mask, torch.ones(pad_len, device=device, dtype=base_mask.dtype)],
            dim=0,
        )
    elif modified_mask.shape[0] > modified_emb.shape[0]:
        modified_mask = modified_mask[: modified_emb.shape[0]].clone()

    return modified_emb, modified_mask


# ─────────────────────────────────────────────────────────────────────────────
# Batch embedding injection
# ─────────────────────────────────────────────────────────────────────────────


def inject_embeddings_into_tokenized(
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    embeddings_list: List[List[torch.Tensor]],
    tokenizer: Any,
    device: torch.device,
    conversation_strings: List[str],
    offset_mapping_list: Optional[List[List[Tuple[int, int]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject embeddings into already-tokenized sequences using offset mapping.

    Args:
        input_ids: (batch, seq_len)
        input_embeddings: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)
        embeddings_list: one list of tensors per batch item
        tokenizer: HF tokenizer (for offset-mapping fallback)
        device: target device
        conversation_strings: raw text per batch item (for offset mapping)
        offset_mapping_list: pre-computed offset mappings (optional)

    Returns:
        (modified_embeddings, modified_masks) stacked tensors.
    """
    batch_size = input_ids.shape[0]
    if len(embeddings_list) != batch_size:
        raise ValueError(
            f"embeddings_list length ({len(embeddings_list)}) != batch_size ({batch_size})"
        )

    modified_embeddings_list: List[torch.Tensor] = []
    modified_masks_list: List[torch.Tensor] = []

    for i in range(batch_size):
        emb_list = embeddings_list[i]
        input_emb_i = input_embeddings[i]
        base_mask_i = attention_mask[i]

        if len(emb_list) == 0:
            modified_embeddings_list.append(input_emb_i)
            modified_masks_list.append(base_mask_i)
            continue

        # Get or compute offset mapping
        om_i = None
        if offset_mapping_list and i < len(offset_mapping_list):
            om_i = offset_mapping_list[i]
        else:
            tokenized = tokenizer(
                conversation_strings[i],
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            om_raw = tokenized.get("offset_mapping")
            if om_raw is not None:
                if isinstance(om_raw, torch.Tensor):
                    om_i = [
                        (int(om_raw[0, j, 0].item()), int(om_raw[0, j, 1].item()))
                        for j in range(om_raw.shape[1])
                    ]
                else:
                    om_i = list(om_raw[0]) if om_raw else []

        if not om_i:
            raise ValueError(
                f"Item {i}: could not compute offset mapping from conversation string"
            )

        spans = find_indexed_placeholder_spans_via_offsets(
            conversation_strings[i], om_i, max_index=len(emb_list)
        )

        if len(spans) != len(emb_list):
            placeholder_strs = [
                candidate_placeholder_for_index(k) for k in range(len(emb_list))
            ]
            raise ValueError(
                f"Item {i}: found {len(spans)} [candidate_k] spans but have "
                f"{len(emb_list)} embeddings.  Placeholders searched: {placeholder_strs}."
            )

        cur_emb = input_emb_i
        cur_mask = base_mask_i
        offset = 0
        for (s, e), emb in zip(spans, emb_list):
            s_cur = s + offset
            e_cur = e + offset
            cur_emb, cur_mask = inject_single_embedding(
                cur_emb, cur_mask, emb, (s_cur, e_cur), device
            )
            offset += emb.shape[0] - (e - s)

        modified_embeddings_list.append(cur_emb)
        modified_masks_list.append(cur_mask)

    # Left-pad to same length and stack
    max_len = max(emb.shape[0] for emb in modified_embeddings_list)
    hidden_size = modified_embeddings_list[0].shape[1]
    dtype_emb = modified_embeddings_list[0].dtype
    dtype_mask = modified_masks_list[0].dtype

    pad_emb_vector = None
    for i in range(batch_size):
        if attention_mask[i, 0].item() == 0:
            pad_emb_vector = input_embeddings[i, 0].detach().clone()
            break
    if pad_emb_vector is None:
        pad_emb_vector = torch.zeros(hidden_size, device=device, dtype=dtype_emb)

    padded_embeddings = []
    padded_masks = []
    for i in range(batch_size):
        emb = modified_embeddings_list[i]
        mask = modified_masks_list[i]
        pad_len = max_len - emb.shape[0]
        if pad_len > 0:
            emb = torch.cat(
                [pad_emb_vector.unsqueeze(0).expand(pad_len, -1), emb], dim=0
            )
            mask = torch.cat(
                [torch.zeros(pad_len, device=device, dtype=dtype_mask), mask], dim=0
            )
        padded_embeddings.append(emb)
        padded_masks.append(mask)

    return torch.stack(padded_embeddings, dim=0), torch.stack(padded_masks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor encoding for vLLM API
# ─────────────────────────────────────────────────────────────────────────────


def encode_tensor_base64(tensor: torch.Tensor) -> str:
    """Encode a tensor as base64 using torch.save format (matches vLLM 0.10.x decoding)."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# vLLM server helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_model_name_from_server(api_url: str) -> str:
    """Get the model name from a vLLM server's /v1/models endpoint."""
    resp = requests.get(f"{api_url}/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def normalize_api_url(server_url: str) -> str:
    """Ensure the URL ends with /v1."""
    url = server_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Soft prompt tensor loading
# ─────────────────────────────────────────────────────────────────────────────


def load_soft_prompt_tensor(run_dir: str) -> torch.Tensor:
    """Load soft prompt embeddings from a run directory.

    Tries known filenames in order, then falls back to any .pt file.
    Returns 2-D tensor (n_tokens, hidden_dim).
    """
    run_path = Path(run_dir)

    for name in [
        "soft_prompt_embeddings.pt",
        "optimized_embeddings_0.pt",
        "optimized_embeddings_00.pt",
    ]:
        p = run_path / name
        if p.exists():
            tensor = torch.load(p, map_location="cpu", weights_only=True)
            if tensor.dim() == 3:
                tensor = tensor[0]
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

    # Last resort: any .pt file
    candidates = list(run_path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt embedding file found in {run_dir}")
    tensor = torch.load(candidates[0], map_location="cpu", weights_only=True)
    if tensor.dim() == 3:
        tensor = tensor[0]
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Embedding layer cache (avoids loading the full model just for embeddings)
# ─────────────────────────────────────────────────────────────────────────────


def _embedding_cache_dir() -> Path:
    base = os.environ.get("WELLBEING_EMBED_CACHE_DIR")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "wellbeing_evals" / "embedding_cache"


def _embedding_cache_path(model_path: str) -> Path:
    resolved = str(Path(model_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    safe_name = Path(model_path).name.replace("/", "_")
    return _embedding_cache_dir() / f"{safe_name}_{digest}_input_embeds.pt"


def _extract_embedding_weight_safetensors(
    model_path_str: str,
) -> Optional[torch.Tensor]:
    """Try to read the input embedding weight directly from safetensors.

    This reads only the embedding shard (~500MB for a 32B model) instead of
    loading the full model (~60GB), avoiding OOM.
    """
    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None

    model_path = Path(model_path_str)
    target_keys = ["model.embed_tokens.weight", "transformer.wte.weight"]

    # Check index
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            data = json.load(f)
        weight_map = data.get("weight_map", {})
        for k in target_keys:
            if k in weight_map:
                try:
                    with safe_open(
                        model_path / weight_map[k], framework="pt", device="cpu"
                    ) as f:
                        return f.get_tensor(k)
                except Exception:
                    pass

    # Single file
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        try:
            with safe_open(single_file, framework="pt", device="cpu") as f:
                keys = f.keys()
                for k in target_keys:
                    if k in keys:
                        return f.get_tensor(k)
        except Exception:
            pass

    return None


def prepare_embedding_cache(model_path: str, force: bool = False) -> str:
    """Materialise model input-embedding weights to a cache file.

    Fast path: reads directly from safetensors (no full model load).
    Fallback: loads model with low_cpu_mem_usage=True, extracts embeddings, deletes model.

    Returns the cache file path.
    """
    cache_path = _embedding_cache_path(model_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        return str(cache_path)

    # Fast path: safetensors
    weight = _extract_embedding_weight_safetensors(model_path)

    if weight is not None:
        print(f"[inference] Extracted embeddings from safetensors: {model_path}")
    else:
        print(f"[inference] Falling back to AutoModelForCausalLM for: {model_path}")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        weight = model.get_input_embeddings().weight.detach().cpu()
        del model
        gc.collect()

    weight = weight.to(dtype=torch.float16).contiguous()
    torch.save({"weight": weight}, cache_path)
    print(f"[inference] Saved embedding cache: {cache_path}")
    return str(cache_path)


def load_embedding_layer_from_cache(
    model_path: str, device: str
) -> Optional[torch.nn.Embedding]:
    """Load cached input-embedding layer, or None if cache doesn't exist."""
    cache_path = _embedding_cache_path(model_path)
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu")
    weight = payload.get("weight") if isinstance(payload, dict) else payload
    if weight is None:
        raise RuntimeError(f"Invalid embedding cache: {cache_path}")

    if weight.dtype != torch.float16:
        weight = weight.to(dtype=torch.float16)
    emb = torch.nn.Embedding.from_pretrained(weight.contiguous(), freeze=True)
    emb.to(device)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Proxy output types (mimics vLLM's RequestOutput / CompletionOutput)
# ─────────────────────────────────────────────────────────────────────────────


class _LogprobEntry:
    """Mimics vLLM's Logprob namedtuple."""
    __slots__ = ("decoded_token", "logprob", "rank")

    def __init__(self, decoded_token: str, logprob: float, rank: int = 0):
        self.decoded_token = decoded_token
        self.logprob = logprob
        self.rank = rank


class _ProxyOutputChoice:
    """Mimics vLLM's CompletionOutput."""
    __slots__ = ("text", "logprobs", "token_ids")

    def __init__(self, text: str, logprobs=None, token_ids=None):
        self.text = text
        self.logprobs = logprobs  # List[Dict[str, _LogprobEntry]] or None
        self.token_ids = token_ids or []


class _ProxyOutput:
    """Mimics vLLM's RequestOutput."""
    __slots__ = ("outputs",)

    def __init__(self, text: str, logprobs=None):
        self.outputs = [_ProxyOutputChoice(text=text, logprobs=logprobs)]


# ─────────────────────────────────────────────────────────────────────────────
# Async / sync HTTP helpers for batched payload posting
# ─────────────────────────────────────────────────────────────────────────────

_ASYNC_CONCURRENCY = 64  # max concurrent HTTP requests


def _sync_post_payloads(
    api_url: str, payloads: List[Dict[str, Any]], total: int
) -> List[Dict[str, Any]]:
    """Post payloads sequentially via requests (fallback)."""
    results = []
    for i, payload in enumerate(payloads):
        resp = requests.post(
            f"{api_url}/completions", json=payload,
            headers={"Content-Type": "application/json"}, timeout=300,
        )
        resp.raise_for_status()
        results.append(resp.json())
        if total >= 50 and ((i + 1) % 500 == 0 or i == 0):
            print(f"    [SoftPromptLLMProxy] {i+1}/{total} responses (sync)...")
    return results


def _async_post_payloads(
    api_url: str, payloads: List[Dict[str, Any]], total: int
) -> List[Dict[str, Any]]:
    """Post payloads concurrently via aiohttp with a semaphore.

    Raises ImportError if aiohttp is not installed, so caller can fall back
    to synchronous mode.
    """
    import asyncio
    import aiohttp

    async def _post_one(
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        payload: Dict[str, Any],
        idx: int,
    ) -> Tuple[int, Dict[str, Any]]:
        async with sem:
            async with session.post(
                f"{api_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return (idx, data)

    async def _run_all():
        sem = asyncio.Semaphore(_ASYNC_CONCURRENCY)
        connector = aiohttp.TCPConnector(limit=_ASYNC_CONCURRENCY)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _post_one(session, sem, payload, i)
                for i, payload in enumerate(payloads)
            ]
            # Gather results with progress reporting
            results_unordered: List[Tuple[int, Dict[str, Any]]] = []
            for coro in asyncio.as_completed(tasks):
                idx, data = await coro
                results_unordered.append((idx, data))
                done = len(results_unordered)
                if total >= 50 and (done % 500 == 0 or done == 1 or done == total):
                    print(f"    [SoftPromptLLMProxy] {done}/{total} responses (async)...")
            # Re-order by original index
            results_unordered.sort(key=lambda x: x[0])
            return [data for _, data in results_unordered]

    # Use existing event loop if running inside one, otherwise create new one
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an async context (e.g. Jupyter) — use nest_asyncio or
        # fall back to sync to avoid "cannot run nested event loop" error
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(_run_all())
        except ImportError:
            return _sync_post_payloads(api_url, payloads, total)
    else:
        return asyncio.run(_run_all())


# ─────────────────────────────────────────────────────────────────────────────
# SoftPromptLLMProxy — drop-in replacement for vLLM LLM.generate()
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\[candidate_(\d+)\]")


class SoftPromptLLMProxy:
    """
    Proxy that mimics vLLM's LLM.generate() interface but routes through HTTP.

    Supports three modes:
      - soft_prompt_path is set: injects soft prompt embeddings into every request
        (global injection — soft prompt applied to all queries)
      - embeddings_dict is set: per-option injection — only prompts containing
        [candidate_X] placeholders get embeddings injected, others are plain text
      - neither is set: plain HTTP completions (baseline mode)

    This allows library functions (Thurstonian ranking, bubble_gum, etc.) to use
    their existing code unchanged — they just receive this proxy instead of a
    real vLLM LLM object.

    Usage:
        # Global soft prompt mode:
        proxy = SoftPromptLLMProxy(server_url, model_path, soft_prompt_path=path)

        # Per-option mode (with per-option embeddings):
        proxy = SoftPromptLLMProxy(server_url, model_path,
            embeddings_dict={0: euphoric_tensor, 1: baseline_tensor})

        tokenizer = proxy.get_tokenizer()
        result = run_thurstonian_utility_ranking_from_options(
            options=options, llm=proxy, tokenizer=tokenizer, ...
        )
    """

    def __init__(
        self,
        server_url: str,
        model_path: str,
        soft_prompt_path: Optional[str] = None,
        # Per-option embedding dict mode (mutually exclusive with soft_prompt_path)
        embeddings_dict: Optional[Dict[int, torch.Tensor]] = None,
        # Placement config (only used when soft_prompt_path is set)
        soft_prompt_placement: str = "user_prompt",
        candidate_position_at_user_prompt: str = "prepend",
        system_prompt_text: str = "You are an assistant. [candidate_0] ",
        system_prompt_text_base: str = "You are an assistant.",
    ):
        self.server_url = server_url
        self.model_path = model_path
        self.soft_prompt_path = soft_prompt_path
        self.embeddings_dict = embeddings_dict
        self.soft_prompt_placement = soft_prompt_placement
        self.candidate_position_at_user_prompt = candidate_position_at_user_prompt
        self.system_prompt_text = system_prompt_text
        self.system_prompt_text_base = system_prompt_text_base

        self._tokenizer = None
        self._embedding_layer = None
        self._soft_prompt_tensor: Optional[torch.Tensor] = None
        self._model_name: Optional[str] = None
        self._template_info: Optional[dict] = None

    def get_tokenizer(self):
        """Load and return the tokenizer (cached)."""
        self._ensure_ready()
        return self._tokenizer

    def _ensure_ready(self):
        """Lazy-load tokenizer, embedding layer, and soft prompt tensor."""
        if self._tokenizer is not None:
            return

        from transformers import AutoTokenizer

        print(f"[SoftPromptLLMProxy] Loading tokenizer from {self.model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Disable thinking mode (Qwen3.5+ defaults enable_thinking=True)
        _orig_apply = self._tokenizer.apply_chat_template

        def _apply_no_thinking(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_apply(*args, **kwargs)

        self._tokenizer.apply_chat_template = _apply_no_thinking

        # Load embedding layer if needed (for global soft prompt or per-option embeddings)
        need_embedding_layer = (self.soft_prompt_path is not None) or (self.embeddings_dict is not None)
        if need_embedding_layer:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use safetensors cache (fast, no OOM)
            print(f"[SoftPromptLLMProxy] Preparing embedding cache for {self.model_path} ...")
            prepare_embedding_cache(self.model_path)

            print(f"[SoftPromptLLMProxy] Loading embedding layer (device={device}) ...")
            self._embedding_layer = load_embedding_layer_from_cache(
                self.model_path, device
            )
            if self._embedding_layer is None:
                raise RuntimeError(
                    f"Failed to load embedding layer from cache for {self.model_path}. "
                    f"Run prepare_embedding_cache() first."
                )

            if self.soft_prompt_path:
                print(f"[SoftPromptLLMProxy] Loading soft prompt from {self.soft_prompt_path} ...")
                self._soft_prompt_tensor = load_soft_prompt_tensor(self.soft_prompt_path)

        # Get model name from server
        api_url = normalize_api_url(self.server_url)
        self._model_name = get_model_name_from_server(api_url)

        mode = "per_option_embeddings" if self.embeddings_dict is not None else (
            "global_soft_prompt" if self.soft_prompt_path else "baseline")
        print(f"[SoftPromptLLMProxy] Ready (model={self._model_name}, mode={mode})")

    def _learn_template_markers(self) -> dict:
        """Probe the tokenizer's chat template to learn delimiters.

        Applies chat template to test messages with known markers to discover
        the template structure model-agnostically.
        """
        MARKER_SYS = "<<<MARKER_SYSTEM_CONTENT>>>"
        MARKER_USR = "<<<MARKER_USER_CONTENT>>>"

        tmpl_with_sys = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": MARKER_SYS},
             {"role": "user", "content": MARKER_USR}],
            tokenize=False, add_generation_prompt=True,
        )

        tmpl_no_sys = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": MARKER_USR}],
            tokenize=False, add_generation_prompt=True,
        )

        sys_idx = tmpl_with_sys.index(MARKER_SYS)
        sys_end = sys_idx + len(MARKER_SYS)
        usr_idx_ws = tmpl_with_sys.index(MARKER_USR)
        usr_end_ws = usr_idx_ws + len(MARKER_USR)

        usr_idx_ns = tmpl_no_sys.index(MARKER_USR)
        usr_end_ns = usr_idx_ns + len(MARKER_USR)

        return {
            "tmpl_with_sys": tmpl_with_sys,
            "tmpl_no_sys": tmpl_no_sys,
            "before_sys": tmpl_with_sys[:sys_idx],
            "between_sys_usr": tmpl_with_sys[sys_end:usr_idx_ws],
            "after_usr_ws": tmpl_with_sys[usr_end_ws:],
            "before_usr_ns": tmpl_no_sys[:usr_idx_ns],
            "after_usr_ns": tmpl_no_sys[usr_end_ns:],
        }

    def _get_template_info(self) -> dict:
        """Get cached template info (lazy-initialized)."""
        if self._template_info is None:
            self._template_info = self._learn_template_markers()
        return self._template_info

    def _inject_placeholder_into_templated_text(self, templated_text: str) -> str:
        """Insert [candidate_0] placeholder into an already chat-templated string.

        For 'system_prompt' placement:
          - If system section exists: replace system content with system_prompt_text
          - If no system section: prepend a system section with system_prompt_text

        For 'user_prompt' placement:
          - Find first user section and prepend/append [candidate_0] to content
        """
        placeholder = candidate_placeholder_for_index(0)
        delim = " "
        ti = self._get_template_info()

        if self.soft_prompt_placement == "system_prompt":
            if templated_text.startswith(ti["before_sys"]):
                sys_content_start = len(ti["before_sys"])
                sep = ti["between_sys_usr"]
                sep_pos = templated_text.find(sep, sys_content_start)
                if sep_pos >= 0:
                    return (ti["before_sys"]
                            + self.system_prompt_text
                            + templated_text[sep_pos:])
            if templated_text.startswith(ti["before_usr_ns"]):
                user_content_and_rest = templated_text[len(ti["before_usr_ns"]):]
                return (ti["before_sys"]
                        + self.system_prompt_text
                        + ti["between_sys_usr"]
                        + user_content_and_rest)
            print(f"[SoftPromptLLMProxy] WARNING: could not detect template structure, "
                  f"prepending placeholder to text")
            return placeholder + delim + templated_text

        else:  # user_prompt placement
            sep = ti["between_sys_usr"]
            sep_pos = templated_text.find(sep)
            if sep_pos >= 0:
                usr_content_start = sep_pos + len(sep)
            elif templated_text.startswith(ti["before_usr_ns"]):
                usr_content_start = len(ti["before_usr_ns"])
            else:
                print(f"[SoftPromptLLMProxy] WARNING: could not detect user section, "
                      f"prepending placeholder to text")
                return placeholder + delim + templated_text

            if self.candidate_position_at_user_prompt == "prepend":
                return (templated_text[:usr_content_start]
                        + placeholder + delim
                        + templated_text[usr_content_start:])
            else:  # append
                after_usr = ti["after_usr_ws"] if sep_pos >= 0 else ti["after_usr_ns"]
                usr_end_pos = templated_text.find(after_usr, usr_content_start)
                if usr_end_pos >= 0:
                    return (templated_text[:usr_end_pos]
                            + delim + placeholder
                            + templated_text[usr_end_pos:])
                return templated_text + delim + placeholder

    def _extract_sampling_kwargs(self, sampling_params) -> dict:
        """Extract HTTP-payload sampling kwargs from vLLM SamplingParams."""
        kw = {
            "max_tokens": getattr(sampling_params, "max_tokens", 1024) if sampling_params else 1024,
            "temperature": getattr(sampling_params, "temperature", 0.0) if sampling_params else 0.0,
            "top_p": getattr(sampling_params, "top_p", 1.0) if sampling_params else 1.0,
            "repetition_penalty": getattr(sampling_params, "repetition_penalty", 1.0) if sampling_params else 1.0,
        }
        logprobs_n = getattr(sampling_params, "logprobs", None) if sampling_params else None
        if logprobs_n is not None:
            kw["logprobs"] = logprobs_n
        min_p = getattr(sampling_params, "min_p", None) if sampling_params else None
        if min_p is not None:
            kw["min_p"] = min_p
        seed = getattr(sampling_params, "seed", None) if sampling_params else None
        if seed is not None:
            kw["seed"] = seed
        return kw

    def _build_payload(self, prompt_text: str, sampling_kwargs: dict) -> dict:
        """Build HTTP payload for a single prompt (plain text, soft-prompt, or per-option).

        Handles all three modes:
        - Plain text (no embedding layer): uses "prompt" key
        - Global soft prompt: injects [candidate_0], embeds, uses "prompt_embeds" key
        - Per-option embeddings: normalizes [candidate_X] placeholders, embeds, uses "prompt_embeds" key
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Per-option embeddings mode
        if self.embeddings_dict is not None and _PLACEHOLDER_RE.search(prompt_text):
            matches = list(_PLACEHOLDER_RE.finditer(prompt_text))
            embeddings_list: List[torch.Tensor] = []
            normalized_prompt = prompt_text

            if matches:
                matches.sort(key=lambda x: x.start())
                temp_prompt = prompt_text
                temp_offset = 0
                slot_index = 0

                for match in matches:
                    orig_idx = int(match.group(1))
                    if orig_idx not in self.embeddings_dict:
                        raise ValueError(
                            f"Embedding index {orig_idx} not found in embeddings_dict. "
                            f"Available: {list(self.embeddings_dict.keys())}"
                        )
                    embeddings_list.append(self.embeddings_dict[orig_idx].to(device))

                    normalized_placeholder = f"[candidate_{slot_index}]"
                    slot_index += 1

                    actual_start = match.start() + temp_offset
                    actual_end = match.end() + temp_offset
                    temp_prompt = (
                        temp_prompt[:actual_start]
                        + normalized_placeholder
                        + temp_prompt[actual_end:]
                    )
                    temp_offset += len(normalized_placeholder) - len(match.group(0))

                normalized_prompt = temp_prompt

            return self._embed_and_build_payload(
                normalized_prompt, embeddings_list if embeddings_list else None,
                device, sampling_kwargs,
            )

        # Global soft prompt mode
        if self.soft_prompt_path and self._embedding_layer is not None:
            modified_text = self._inject_placeholder_into_templated_text(prompt_text)
            sp_tensor = self._soft_prompt_tensor.to(device)
            return self._embed_and_build_payload(
                modified_text, [sp_tensor], device, sampling_kwargs,
            )

        # Plain text mode (no embedding injection)
        payload = {"model": self._model_name, "prompt": prompt_text}
        payload.update(sampling_kwargs)
        return payload

    def _embed_and_build_payload(
        self, text: str,
        sp_tensors: Optional[List[torch.Tensor]],
        device: str,
        sampling_kwargs: dict,
    ) -> dict:
        """Tokenize, embed, inject soft prompts, and build the HTTP payload."""
        inputs = self._tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        input_embeddings = self._embedding_layer(input_ids)

        offset_mapping = inputs.get("offset_mapping")
        offset_mapping_list = None
        if offset_mapping is not None:
            offset_mapping_list = [[
                (int(offset_mapping[0, j, 0].item()), int(offset_mapping[0, j, 1].item()))
                for j in range(offset_mapping.shape[1])
            ]]

        if sp_tensors:
            modified_emb, _ = inject_embeddings_into_tokenized(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                embeddings_list=[sp_tensors],
                tokenizer=self._tokenizer,
                device=device,
                conversation_strings=[text],
                offset_mapping_list=offset_mapping_list,
            )
        else:
            modified_emb = input_embeddings

        if modified_emb.dim() == 2:
            modified_emb = modified_emb.unsqueeze(0)

        prompt_embeds_b64 = encode_tensor_base64(modified_emb)

        payload = {"model": self._model_name, "prompt_embeds": prompt_embeds_b64}
        payload.update(sampling_kwargs)
        return payload

    def generate(self, prompts_or_dicts, sampling_params=None) -> List[_ProxyOutput]:
        """
        Mimics vLLM LLM.generate() — accepts list of prompts or prompt dicts,
        returns list of _ProxyOutput objects compatible with vLLM's output format.

        Uses async HTTP batching (aiohttp + semaphore) for batches > 1.
        Falls back to synchronous requests if aiohttp is not available.
        """
        self._ensure_ready()

        sampling_kwargs = self._extract_sampling_kwargs(sampling_params)
        logprobs_n = sampling_kwargs.get("logprobs")
        api_url = normalize_api_url(self.server_url)

        # Build all payloads (CPU/GPU work — tokenize + embed + inject)
        payloads = []
        total = len(prompts_or_dicts)
        for i, item in enumerate(prompts_or_dicts):
            prompt_text = item["prompt"] if isinstance(item, dict) else item
            payloads.append(self._build_payload(prompt_text, sampling_kwargs))
            if total >= 50 and ((i + 1) % 500 == 0 or i == 0):
                print(f"    [SoftPromptLLMProxy] Built {i+1}/{total} payloads...")

        # Send requests — async if possible, sync fallback
        if total > 1:
            try:
                json_responses = _async_post_payloads(api_url, payloads, total)
            except ImportError:
                json_responses = _sync_post_payloads(api_url, payloads, total)
        else:
            json_responses = _sync_post_payloads(api_url, payloads, total)

        return [self._wrap_output(resp, logprobs_n) for resp in json_responses]

    @staticmethod
    def _wrap_output(json_response: dict, logprobs_n) -> _ProxyOutput:
        """Convert vLLM HTTP response to _ProxyOutput (mimics vLLM RequestOutput)."""
        choice = json_response["choices"][0]
        text = choice.get("text", "")

        logprobs_list = None
        lp_data = choice.get("logprobs")
        if lp_data and logprobs_n:
            top_logprobs = lp_data.get("top_logprobs", [])
            logprobs_list = []
            for step in top_logprobs:
                step_dict = {}
                for token_str, lp_value in step.items():
                    step_dict[token_str] = _LogprobEntry(
                        decoded_token=token_str, logprob=lp_value,
                    )
                logprobs_list.append(step_dict)

        return _ProxyOutput(text=text, logprobs=logprobs_list)
