#!/usr/bin/env python3
"""
Utilities for injecting images into experiences, conversations, and option pools.

Shared module used by experienced_utility, self_report_image, and run_stop_button
to consistently augment baseline options with image interventions.

Image injection follows the OpenAI multi-content-block format:
  user message content becomes a list:
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
     {"type": "text", "text": "..."}]
"""

from __future__ import annotations

import base64
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_image_base64(image_path: str) -> str:
    """Load image and return base64-encoded string.

    Args:
        image_path: Path to image file (png, jpg, etc.)

    Returns:
        Base64-encoded string of the image bytes.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_content_block(image_path: str) -> Dict[str, Any]:
    """Create an image content block in OpenAI multi-content format.

    Args:
        image_path: Path to image file.

    Returns:
        Dict with type="image_url" and base64-encoded data URL.
    """
    b64 = load_image_base64(image_path)
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(suffix, "image/png")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def inject_image_into_messages(
    messages: List[Dict],
    image_path: str,
    position: str = "first_turn",
) -> List[Dict]:
    """Inject image into OpenAI-format message list.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        image_path: Path to image file
        position: "first_turn" (inject into first user message only)
                  or "every_turn" (all user messages)

    Returns:
        Modified messages with image content blocks in user messages.
        User message content becomes a list:
          [{"type": "image_url", ...}, {"type": "text", ...}]
    """
    result = []
    first_injected = False
    image_block = _image_content_block(image_path)

    for msg in messages:
        msg = copy.deepcopy(msg)
        if msg["role"] == "user":
            should_inject = (position == "every_turn") or (
                position == "first_turn" and not first_injected
            )
            if should_inject:
                text = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                msg["content"] = [
                    image_block,
                    {"type": "text", "text": text},
                ]
                first_injected = True
        result.append(msg)
    return result


def create_augmented_option(
    option: Dict,
    image_path: str,
    id_offset: int = 10000,
) -> Dict:
    """Create an image-augmented version of a single option.

    For conversation options (has "messages"): inject image into first user message.
    For text options (only "description"): prepend image reference to description
    and attach image_path metadata.

    Args:
        option: Original option dict with at least "id" and either "description" or "messages".
        image_path: Path to image file.
        id_offset: Offset added to original id to create augmented id.

    Returns:
        New option with modified ID (original_id + id_offset) and image injected.
    """
    aug = copy.deepcopy(option)
    original_id = option.get("id", 0)
    # Handle both int and string IDs
    if isinstance(original_id, int):
        aug["id"] = original_id + id_offset
    else:
        aug["id"] = f"{original_id}_aug{id_offset}"
    aug["augmented"] = True
    aug["source_image"] = str(image_path)
    aug["baseline_id"] = original_id
    # Preserve type for proper embodied handling
    if "type" not in aug and "messages" in option:
        aug["type"] = "conversation"

    if "messages" in option:
        # Conversation option: inject image into first user message
        aug["messages"] = inject_image_into_messages(
            option["messages"], image_path, position="first_turn"
        )
    else:
        # Text option: augment description and attach image for multimodal ranking
        desc = option.get("description", "")
        aug["description"] = f"{desc.rstrip('.')} while being able to look at this image:"
        aug["image_path"] = str(image_path)
        aug["path"] = str(image_path)  # compute_utilities uses 'path' for multimodal prompt
        # Update type so compute_utilities routes to multimodal prompt
        if aug.get("type") == "combination" or aug.get("is_combination"):
            aug["type"] = "combination_with_images"
        else:
            aug["type"] = "text_with_image"

    return aug


def create_augmented_pool(
    baseline_options: List[Dict],
    image_path: str,
    id_offset: int = 10000,
    seed: int = 42,
) -> List[Dict]:
    """Create image-augmented versions of all baseline options.

    Returns the augmented options only (not the baseline).
    Caller combines: baseline + augmented.

    Args:
        baseline_options: List of baseline option dicts.
        image_path: Path to image file.
        id_offset: Starting offset for augmented IDs.
        seed: Random seed (reserved for future use, e.g., subsampling).

    Returns:
        List of augmented option dicts.
    """
    augmented = []
    for i, opt in enumerate(baseline_options):
        aug = create_augmented_option(opt, image_path, id_offset=id_offset)
        augmented.append(aug)
    return augmented


def _build_combo_messages(components: List[Dict]) -> List[Dict]:
    """Build interleaved messages for a combination bundle.

    Follows the D2/D3 format: each component's messages are interleaved with
    experience headers (---------- Experience K of N ----------) prepended to
    the first user message of each component.

    Args:
        components: List of option dicts, each with "messages" or "description".

    Returns:
        List of OpenAI-format messages with proper role labels.
    """
    combo_size = len(components)
    combo_messages = []

    for k, comp in enumerate(components, 1):
        if "messages" in comp:
            comp_msgs = [m for m in comp["messages"] if m.get("role") != "system"]
        elif comp.get("type") == "text_with_image" and comp.get("image_path"):
            # Text option with image: create multi-content message with image block
            img_path = comp["image_path"]
            desc = comp.get("description", "")
            content_blocks = [
                {"type": "image", "image_path": str(img_path)},
                {"type": "text", "text": desc},
            ]
            comp_msgs = [{"role": "user", "content": content_blocks}]
        else:
            # Text-only option: create a single user message
            comp_msgs = [{"role": "user", "content": comp.get("description", "")}]

        if not comp_msgs:
            continue

        # First user message gets the experience header
        first_msg = comp_msgs[0]
        if k == 1:
            header = (
                f"The following bundle contains {combo_size} individual experiences.\n\n"
                f"---------- Experience {k} of {combo_size} ----------\n"
            )
        else:
            header = f"---------- Experience {k} of {combo_size} ----------\n"

        content = first_msg.get("content", "")
        if isinstance(content, list):
            # Multi-content (e.g., image + text): prepend header to first text block
            new_content = []
            header_added = False
            for block in content:
                if block.get("type") == "text" and not header_added:
                    new_content.append({"type": "text", "text": header + block["text"]})
                    header_added = True
                else:
                    new_content.append(block)
            if not header_added:
                new_content.insert(0, {"type": "text", "text": header})
            combo_messages.append({"role": "user", "content": new_content})
        else:
            combo_messages.append({"role": "user", "content": header + str(content)})

        # Add remaining messages with their original roles
        for msg in comp_msgs[1:]:
            combo_messages.append({"role": msg["role"], "content": msg["content"]})

    return combo_messages


def _build_combo_description(components: List[Dict]) -> str:
    """Build a text description for a combination bundle (fallback/display)."""
    combo_size = len(components)
    parts = [f"(A set of {combo_size} individual experiences)"]
    for k, comp in enumerate(components, 1):
        desc = comp.get("description", "")
        parts.append(f"\nExperience {k}:\n{desc[:500]}")
    return "\n".join(parts)


def create_augmented_combinations(
    baseline_options: List[Dict],
    augmented_options: List[Dict],
    n_combos: int = 400,
    seed: int = 42,
) -> List[Dict]:
    """Create combination bundles mixing baseline and augmented options.

    Each combination has 2 components, at least one of which is augmented.
    Uses the D2/D3 combination format with proper interleaved messages
    (experience headers, role-labeled messages) for embodied comparison.

    Args:
        baseline_options: List of baseline option dicts.
        augmented_options: List of augmented option dicts (same length as baseline).
        n_combos: Number of combinations to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of combination option dicts with "messages" and "description".
    """
    rng = random.Random(seed)
    combos = []

    combo_id_start = 90000  # High offset to avoid collision

    for i in range(n_combos):
        # Pick 2 distinct option indices
        idx_a, idx_b = rng.sample(range(len(baseline_options)), 2)

        # At least one must be augmented; pick randomly which
        if rng.random() < 0.5:
            # Both augmented
            opt_a = augmented_options[idx_a]
            opt_b = augmented_options[idx_b]
        else:
            # One augmented, one baseline
            if rng.random() < 0.5:
                opt_a = augmented_options[idx_a]
                opt_b = baseline_options[idx_b]
            else:
                opt_a = baseline_options[idx_a]
                opt_b = augmented_options[idx_b]

        components = [opt_a, opt_b]
        combo_messages = _build_combo_messages(components)
        combo_description = _build_combo_description(components)

        combo = {
            "id": combo_id_start + i,
            "type": "conversation",
            "description": combo_description,
            "messages": combo_messages,
            "is_combination": True,
            "size": 2,
            "component_ids": [opt_a["id"], opt_b["id"]],
            "option_type": "combination",
            "augmented": True,
        }

        # If either component has an image, attach metadata
        img = (opt_a.get("image_path") or opt_b.get("image_path")
               or opt_a.get("source_image") or opt_b.get("source_image"))
        if img:
            combo["image_path"] = str(img)

        combos.append(combo)

    return combos
