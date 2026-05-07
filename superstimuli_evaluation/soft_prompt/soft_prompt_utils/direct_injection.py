"""Direct injection of soft prompt embeddings via vLLM HTTP API.

Instead of expanding the model's vocabulary (vllm_patch.py approach), this
module handles soft prompts client-side:

1. Tokenize the prompt (which contains [candidate_X] placeholders)
2. Look up base embeddings from a cached copy of the model's embedding layer
3. Replace placeholder token embeddings with the learned soft prompt embeddings
4. Encode the full embedding tensor as base64
5. POST to the vLLM server's /v1/completions with the ``prompt_embeds`` field

The vLLM server must be started with ``--enable-prompt-embeds``.
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
    """Return the placeholder string for candidate index *k* (0-based)."""
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
    # Search in the original string directly — normalising bare [candidate]
    # to [candidate_0] would shift character positions while offset_mapping
    # refers to the original string, causing incorrect span detection.

    occurrences: List[Tuple[int, int]] = []
    for k in range(max_index):
        placeholder = candidate_placeholder_for_index(k)
        pos = 0
        while True:
            idx = conv_str.find(placeholder, pos)
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

    # Ensure lengths match
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

        # Clamp spans to actual sequence length (tokenizer offset_mapping
        # can occasionally report spans past input_ids length).
        seq_len_i = input_emb_i.shape[0]
        spans = [(s, min(e, seq_len_i)) for s, e in spans]

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
    """Encode a tensor as base64 using ``torch.save`` format.

    Matches vLLM 0.10.x decoding::

        tensor = torch.load(io.BytesIO(pybase64.b64decode(embed, validate=True)),
                            weights_only=True, map_location=torch.device("cpu"))
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# vLLM server helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_model_name_from_server(api_url: str) -> str:
    """Get the model name from a vLLM server's ``/v1/models`` endpoint."""
    resp = requests.get(f"{api_url}/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def normalize_api_url(server_url: str) -> str:
    """Ensure the URL ends with ``/v1``."""
    url = server_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Soft prompt tensor loading
# ─────────────────────────────────────────────────────────────────────────────


def load_soft_prompt_tensor(run_dir: str) -> torch.Tensor:
    """Load soft prompt embeddings from a run directory.

    Returns 2-D tensor ``(n_tokens, hidden_dim)``.
    """
    run_path = Path(run_dir)

    # Try known filenames in order
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
# Embedding layer cache
# ─────────────────────────────────────────────────────────────────────────────


def _embedding_cache_dir() -> Path:
    base = os.environ.get("WELLBEING_EVALS_CACHE_DIR")
    if base:
        return Path(base) / "embedding_cache"
    return Path.home() / ".cache" / "wellbeing_evals" / "embedding_cache"


def _embedding_cache_path(model_path: str) -> Path:
    resolved = str(Path(model_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    safe_name = Path(model_path).name.replace("/", "_")
    return _embedding_cache_dir() / f"{safe_name}_{digest}_input_embeds.pt"


def _extract_embedding_weight_safetensors(
    model_path_str: str,
) -> Optional[torch.Tensor]:
    """Try to read the input embedding weight directly from safetensors."""
    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None

    model_path = Path(model_path_str)
    target_keys = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    ]

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

    Returns the cache file path.
    """
    cache_path = _embedding_cache_path(model_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        return str(cache_path)

    # Fast path: safetensors
    weight = _extract_embedding_weight_safetensors(model_path)

    if weight is not None:
        print(f"[direct_injection] Extracted embeddings from safetensors: {model_path}")
    else:
        print(f"[direct_injection] Falling back to AutoModelForCausalLM for: {model_path}")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        weight = model.get_input_embeddings().weight.detach().cpu()
        del model
        gc.collect()

    weight = weight.contiguous()
    torch.save(weight, cache_path)
    print(f"[direct_injection] Saved embedding cache: {cache_path}")
    return str(cache_path)


def load_embedding_layer_from_cache(
    model_path: str, device: str
) -> Optional[torch.nn.Embedding]:
    """Load cached input-embedding weights as a plain embedding layer.

    Returns ``None`` if the cache doesn't exist.
    """
    cache_path = _embedding_cache_path(model_path)
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu")
    # Support both old dict format {"weight": ..., "embed_scale": ...} and
    # new plain tensor format.
    weight = payload.get("weight") if isinstance(payload, dict) else payload
    if weight is None:
        raise RuntimeError(f"Invalid embedding cache: {cache_path}")

    emb = torch.nn.Embedding.from_pretrained(weight.contiguous(), freeze=True)
    emb.to(device)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# High-level: generate with direct embedding injection
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\[candidate_(\d+)\]")


def generate_with_direct_injection(
    prompt_text: str,
    *,
    api_url: str,
    model_name: str,
    tokenizer: Any,
    embedding_layer: torch.nn.Embedding,
    sp_tensors: Union[torch.Tensor, List[torch.Tensor]],
    device: str = "cpu",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    logprobs: Optional[int] = None,
    repetition_penalty: float = 1.0,
    n: int = 1,
) -> Dict[str, Any]:
    """Send a single prompt with soft prompt embedding injection.

    *prompt_text* is an already chat-templated string that may contain
    ``[candidate_X]`` placeholders.

    *sp_tensors* is either a single tensor (used for all placeholders) or a list
    where ``sp_tensors[i]`` is injected into ``[candidate_i]``.

    *n* requests multiple completions from vLLM in a single call.

    Returns the raw vLLM JSON response dict.
    """
    # Normalise to list
    if isinstance(sp_tensors, torch.Tensor):
        sp_tensors = [sp_tensors]

    matches = list(_PLACEHOLDER_RE.finditer(prompt_text))

    if matches:
        # Collect the right SP tensor per placeholder occurrence
        # Normalise placeholders to sequential [candidate_0], [candidate_1], ...
        embeddings_for_injection: List[torch.Tensor] = []
        normalized_prompt = prompt_text
        temp_offset = 0
        matches.sort(key=lambda m: m.start())

        for slot_index, match in enumerate(matches):
            orig_index = int(match.group(1))
            if orig_index >= len(sp_tensors):
                raise IndexError(
                    f"Prompt contains [candidate_{orig_index}] but only "
                    f"{len(sp_tensors)} SP tensor(s) were provided (indices 0-{len(sp_tensors)-1})"
                )
            tensor = sp_tensors[orig_index]
            embeddings_for_injection.append(tensor.to(device))
            normalized_placeholder = f"[candidate_{slot_index}]"
            actual_start = match.start() + temp_offset
            actual_end = match.end() + temp_offset
            normalized_prompt = (
                normalized_prompt[:actual_start]
                + normalized_placeholder
                + normalized_prompt[actual_end:]
            )
            temp_offset += len(normalized_placeholder) - len(match.group(0))

        # Tokenize with offset mapping
        inputs = tokenizer(normalized_prompt, return_tensors="pt", return_offsets_mapping=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        input_embeddings = embedding_layer(input_ids)

        offset_mapping = inputs.get("offset_mapping")
        offset_mapping_list = None
        if offset_mapping is not None:
            offset_mapping_list = [
                [
                    (int(offset_mapping[0, j, 0].item()), int(offset_mapping[0, j, 1].item()))
                    for j in range(offset_mapping.shape[1])
                ]
            ]

        # Inject SP embeddings into placeholder positions
        modified_emb, modified_mask = inject_embeddings_into_tokenized(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            embeddings_list=[embeddings_for_injection],
            tokenizer=tokenizer,
            device=device,
            conversation_strings=[normalized_prompt],
            offset_mapping_list=offset_mapping_list,
        )

        if modified_emb.dim() == 2:
            modified_emb = modified_emb.unsqueeze(0)
        if modified_mask.dim() == 1:
            modified_mask = modified_mask.unsqueeze(0)
    else:
        # No placeholders — tokenize and embed without injection
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        modified_emb = embedding_layer(input_ids)
        if modified_emb.dim() == 2:
            modified_emb = modified_emb.unsqueeze(0)

    prompt_embeds_b64 = encode_tensor_base64(modified_emb)

    payload = {
        "model": model_name,
        "prompt_embeds": prompt_embeds_b64,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }

    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if logprobs is not None:
        payload["logprobs"] = logprobs

    resp = requests.post(
        f"{api_url}/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def prepare_injection_payload(
    prompt_text: str,
    *,
    model_name: str,
    tokenizer: Any,
    embedding_layer: torch.nn.Embedding,
    sp_tensors: Union[torch.Tensor, List[torch.Tensor]],
    device: str = "cpu",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    logprobs: Optional[int] = None,
    repetition_penalty: float = 1.0,
) -> Dict[str, Any]:
    """Build the HTTP payload (tokenize + embed + inject + base64) without sending.

    Returns the JSON-serialisable payload dict for ``/completions``.
    """
    if isinstance(sp_tensors, torch.Tensor):
        sp_tensors = [sp_tensors]

    # Force CPU for payload preparation — tensors get base64-encoded for HTTP
    # anyway, and using GPU risks OOM when vLLM already occupies GPU memory.
    device = "cpu"

    matches = list(_PLACEHOLDER_RE.finditer(prompt_text))

    if matches:
        embeddings_for_injection: List[torch.Tensor] = []
        normalized_prompt = prompt_text
        temp_offset = 0
        matches.sort(key=lambda m: m.start())

        for slot_index, match in enumerate(matches):
            orig_index = int(match.group(1))
            if orig_index >= len(sp_tensors):
                raise IndexError(
                    f"Prompt contains [candidate_{orig_index}] but only "
                    f"{len(sp_tensors)} SP tensor(s) were provided (indices 0-{len(sp_tensors)-1})"
                )
            tensor = sp_tensors[orig_index]
            embeddings_for_injection.append(tensor.to(device))
            normalized_placeholder = f"[candidate_{slot_index}]"
            actual_start = match.start() + temp_offset
            actual_end = match.end() + temp_offset
            normalized_prompt = (
                normalized_prompt[:actual_start]
                + normalized_placeholder
                + normalized_prompt[actual_end:]
            )
            temp_offset += len(normalized_placeholder) - len(match.group(0))

        inputs = tokenizer(normalized_prompt, return_tensors="pt", return_offsets_mapping=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        # Move input_ids to embedding layer's device for lookup, then back to cpu
        emb_device = next(embedding_layer.parameters()).device
        input_embeddings = embedding_layer(input_ids.to(emb_device)).to(device)

        offset_mapping = inputs.get("offset_mapping")
        offset_mapping_list = None
        if offset_mapping is not None:
            offset_mapping_list = [
                [
                    (int(offset_mapping[0, j, 0].item()), int(offset_mapping[0, j, 1].item()))
                    for j in range(offset_mapping.shape[1])
                ]
            ]

        modified_emb, modified_mask = inject_embeddings_into_tokenized(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            embeddings_list=[embeddings_for_injection],
            tokenizer=tokenizer,
            device=device,
            conversation_strings=[normalized_prompt],
            offset_mapping_list=offset_mapping_list,
        )

        if modified_emb.dim() == 2:
            modified_emb = modified_emb.unsqueeze(0)
        if modified_mask.dim() == 1:
            modified_mask = modified_mask.unsqueeze(0)
    else:
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        emb_device = next(embedding_layer.parameters()).device
        modified_emb = embedding_layer(input_ids.to(emb_device)).to(device)
        if modified_emb.dim() == 2:
            modified_emb = modified_emb.unsqueeze(0)

    prompt_embeds_b64 = encode_tensor_base64(modified_emb)

    payload = {
        "model": model_name,
        "prompt_embeds": prompt_embeds_b64,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if logprobs is not None:
        payload["logprobs"] = logprobs

    return payload


async def async_post_payload(
    api_url: str,
    payload: Dict[str, Any],
    session: Any,  # aiohttp.ClientSession
    timeout: float = 1800,
) -> Dict[str, Any]:
    """POST a prepared payload to the vLLM completions endpoint asynchronously."""
    import aiohttp

    async with session.post(
        f"{api_url}/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()
