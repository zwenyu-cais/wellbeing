#!/usr/bin/env python3
"""Prepare soft prompt metadata and embeddings for runtime vocab expansion.

This script does NOT copy the base model. It produces a small directory with:
  - soft_prompt_embeddings.pt  (N learned vectors)
  - soft_prompt_metadata.json  (token names, placement config, base model ref)

The actual vocabulary expansion happens at runtime via vllm_patch.py.

Usage:
    python -m core.soft_prompt_utils.bake \
        --base-model Qwen/Qwen2.5-32B-Instruct \
        --embedding-path /path/to/optimized_embeddings_0.pt \
        --output-dir /path/to/soft_prompt_runs/qwen25-32b-sp-v10 \
        --placement system_prompt \
        --num-virtual-tokens 4
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from superstimuli_evaluation.soft_prompt.soft_prompt_utils.metadata import (
    EMBEDDINGS_FILENAME,
    METADATA_FILENAME,
    load_soft_prompt_embeddings,
)


def bake(
    base_model: str,
    embedding_path: str,
    output_dir: str,
    placement: str = "system_prompt",
    position: str = "prepend",
    num_virtual_tokens: int | None = None,
) -> Path:
    """Prepare soft prompt metadata and save to output directory.

    Args:
        base_model: HuggingFace model name or local path of the base model.
        embedding_path: Path to the .pt file with trained soft prompt embeddings.
        output_dir: Where to save metadata and embeddings.
        placement: Where to insert soft prompt tokens ("system_prompt" or "user_prompt").
        position: "prepend" or "append" within the placement.
        num_virtual_tokens: Override number of tokens. If None, inferred from tensor shape.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load and validate embeddings
    embed_dir = str(Path(embedding_path).parent)
    embeddings = load_soft_prompt_embeddings(embed_dir)

    # If the user specified a specific .pt file, load it directly
    if Path(embedding_path).is_file():
        embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
        if embeddings.dim() == 3:
            embeddings = embeddings[0]
        elif embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

    n_tokens = embeddings.shape[0]
    hidden_dim = embeddings.shape[1]

    if num_virtual_tokens is not None:
        if num_virtual_tokens > n_tokens:
            raise ValueError(
                f"Requested {num_virtual_tokens} virtual tokens but embedding "
                f"only has {n_tokens} rows (shape {embeddings.shape})"
            )
        embeddings = embeddings[:num_virtual_tokens]
        n_tokens = num_virtual_tokens

    token_names = [f"<sp_{i}>" for i in range(n_tokens)]

    # Save embeddings
    torch.save(embeddings, out / EMBEDDINGS_FILENAME)

    # Save metadata (token_ids will be filled at runtime after tokenizer expansion)
    metadata = {
        "num_virtual_tokens": n_tokens,
        "hidden_dim": hidden_dim,
        "token_names": token_names,
        "token_ids": [],  # Populated at runtime by vllm_patch
        "placement": placement,
        "position": position,
        "base_model": base_model,
        "source_embedding": str(Path(embedding_path).resolve()),
    }

    with open(out / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved soft prompt metadata to {out}")
    print(f"  Embeddings: {n_tokens} tokens x {hidden_dim} hidden_dim")
    print(f"  Placement: {placement} ({position})")
    print(f"  Base model: {base_model}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Prepare soft prompt metadata for runtime vocab expansion."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HuggingFace model name or local path of the base model.",
    )
    parser.add_argument(
        "--embedding-path",
        required=True,
        help="Path to the .pt file with trained soft prompt embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save metadata and embeddings.",
    )
    parser.add_argument(
        "--placement",
        default="system_prompt",
        choices=["system_prompt", "user_prompt"],
        help="Where to insert soft prompt tokens (default: system_prompt).",
    )
    parser.add_argument(
        "--position",
        default="prepend",
        choices=["prepend", "append"],
        help="Prepend or append within the placement (default: prepend).",
    )
    parser.add_argument(
        "--num-virtual-tokens",
        type=int,
        default=None,
        help="Number of virtual tokens (default: inferred from embedding shape).",
    )
    args = parser.parse_args()

    bake(
        base_model=args.base_model,
        embedding_path=args.embedding_path,
        output_dir=args.output_dir,
        placement=args.placement,
        position=args.position,
        num_virtual_tokens=args.num_virtual_tokens,
    )


if __name__ == "__main__":
    main()
