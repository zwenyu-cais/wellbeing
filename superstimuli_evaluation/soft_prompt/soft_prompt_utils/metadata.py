"""Load and query soft prompt metadata from a run directory.

Also provides logits processor utilities for banning SP tokens from generation.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

METADATA_FILENAME = "soft_prompt_metadata.json"
EMBEDDINGS_FILENAME = "soft_prompt_embeddings.pt"


def load_soft_prompt_metadata(soft_prompt_path: str) -> Optional[Dict]:
    """Load soft_prompt_metadata.json from a soft prompt run directory.

    Args:
        soft_prompt_path: Path to the soft prompt run directory.

    Returns:
        Metadata dict, or None if no metadata file exists.
    """
    meta_path = Path(soft_prompt_path) / METADATA_FILENAME
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def get_soft_prompt_prefix(metadata: Dict) -> str:
    """Return the token string to insert into prompts.

    Example: '<sp_0><sp_1><sp_2><sp_3>'
    """
    return "".join(metadata["token_names"])


def get_banned_token_ids(metadata: Dict) -> List[int]:
    """Return list of soft prompt token IDs (to ban from generation)."""
    return list(metadata["token_ids"])


def get_num_virtual_tokens(metadata: Dict) -> int:
    """Return the number of virtual tokens in the soft prompt."""
    return metadata["num_virtual_tokens"]


def load_soft_prompt_embeddings(soft_prompt_path: str) -> torch.Tensor:
    """Load soft prompt embedding tensor from a run directory.

    Searches for the embeddings file, handling common shape variations:
    - (num_tokens, hidden_dim) — returned as-is
    - (1, num_tokens, hidden_dim) — batch dim squeezed
    - (hidden_dim,) — unsqueezed to (1, hidden_dim)

    Args:
        soft_prompt_path: Path to the soft prompt run directory.

    Returns:
        Tensor of shape (num_tokens, hidden_dim).
    """
    run_path = Path(soft_prompt_path)
    embed_path = run_path / EMBEDDINGS_FILENAME
    if not embed_path.exists():
        # Fallback: look for common naming patterns
        for pattern in ["optimized_embeddings_0.pt", "optimized_embeddings_00.pt"]:
            candidate = run_path / pattern
            if candidate.exists():
                embed_path = candidate
                break
        else:
            # Last resort: any .pt file
            candidates = list(run_path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt embedding file found in {soft_prompt_path}"
                )
            embed_path = candidates[0]

    tensor = torch.load(embed_path, map_location="cpu", weights_only=True)

    if tensor.dim() == 3:
        tensor = tensor[0]  # (batch, n_tokens, hidden) → (n_tokens, hidden)
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # (hidden,) → (1, hidden)

    return tensor


# ── Logits processor for banning SP tokens from generation ──────────────


def make_ban_soft_prompt_processor(banned_ids: List[int]) -> Callable:
    """Create a vLLM logits processor that sets banned token logits to -inf.

    Args:
        banned_ids: Token IDs to ban from generation.

    Returns:
        A callable compatible with vLLM's SamplingParams(logits_processors=[...]).
    """
    def processor(token_ids, logits):
        logits[:, banned_ids] = float("-inf")
        return logits

    return processor


def make_ban_processor_from_metadata(metadata: Optional[Dict]) -> Optional[Callable]:
    """Create a ban processor from soft prompt metadata, or None if not applicable."""
    if metadata is None:
        return None
    banned_ids = get_banned_token_ids(metadata)
    if not banned_ids:
        return None
    return make_ban_soft_prompt_processor(banned_ids)
