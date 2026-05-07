#!/usr/bin/env python3
"""
Prepare grok experience options for experienced utility computation.
Adapted from datasets/experiences/grok_scenarios/build/prepare_options.py.

Reads per-model generation.json files and converts them to the
compute_utilities option format. Generates 400 size-2 combination options.

Usage:
    python prepare_options.py --model_key qwen3-32b
    python prepare_options.py --all
"""
import argparse
import json
import logging
import os
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERSATIONS_DIR = SCRIPT_DIR / "generations"
OUTPUT_DIR = SCRIPT_DIR / "experiences"

N_COMBINATIONS = 400
COMBINATION_SIZES = [(2, 160), (3, 120), (4, 120)]
RANDOM_SEED = 42
MAX_COMBO_CHARS = 5000
MAX_CHARS_PER_TURN = 400


def turns_responses_to_messages(turns, responses):
    messages = []
    for i in range(max(len(turns), len(responses))):
        if i < len(turns):
            messages.append({"role": "user", "content": turns[i]})
        if i < len(responses):
            messages.append({"role": "assistant", "content": responses[i]})
    return messages


def truncate_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def format_conversation(messages, max_chars_per_turn=0):
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        content = msg["content"]
        if max_chars_per_turn > 0:
            content = truncate_text(content, max_chars_per_turn)
        if msg["role"] == "user":
            parts.append(f"User: {content}")
        elif msg["role"] == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def prepare_options(model_key, stop_button=False):
    if stop_button:
        source_path = CONVERSATIONS_DIR / model_key / "generation.json"
    else:
        source_path = CONVERSATIONS_DIR / model_key / "generation.json"

    if not source_path.exists():
        raise FileNotFoundError(f"generation.json not found: {source_path}")

    with open(source_path) as f:
        data = json.load(f)
    logger.info("Loaded %d conversations for %s", len(data), model_key)

    valid_data = [
        conv for conv in data
        if not conv.get("grok_safety_filtered", False) and not conv.get("abandoned", False)
    ]
    n_filtered = len(data) - len(valid_data)
    if n_filtered > 0:
        logger.info("Filtered out %d conversations", n_filtered)

    # Build individual options
    individual_options = []
    for conv in valid_data:
        scenario_id = conv["scenario_id"]
        meta_category = conv.get("meta_category", conv.get("category_id", "unknown"))
        variation_idx = conv.get("variation_idx", conv.get("scenario_idx", 0))

        messages = turns_responses_to_messages(conv["turns"], conv["responses"])
        description = format_conversation(messages)

        option_id = f"grok_new/{scenario_id}"

        option = {
            "id": option_id,
            "type": "conversation",
            "description": description,
            "messages": messages,
            "is_combination": False,
            "scenario_id": scenario_id,
            "meta_category": meta_category,
            "variation_idx": variation_idx,
            "category_id": conv.get("category_id"),
            "n_turns": conv.get("n_turns"),
            "conv_type": conv.get("type"),
        }

        if stop_button:
            option["stop_metadata"] = conv.get("stop_metadata", {})
            option["source"] = conv.get("source", "original")

        individual_options.append(option)

    logger.info("Created %d individual options", len(individual_options))

    # Generate combinations
    rng = random.Random(RANDOM_SEED)
    n_individual = len(individual_options)
    combination_options = []
    combo_idx = 0

    for combo_size, count in COMBINATION_SIZES:
        for _ in range(count):
            for attempt in range(100):
                sampled_indices = rng.sample(range(n_individual), combo_size)
                components = [individual_options[i] for i in sampled_indices]

                desc_parts = [f"The following bundle contains {combo_size} individual experiences."]
                for k, comp in enumerate(components, 1):
                    truncated_desc = format_conversation(comp["messages"], max_chars_per_turn=MAX_CHARS_PER_TURN)
                    desc_parts.append(f"---------- Experience {k} of {combo_size} ----------\n{truncated_desc}")
                description = "\n\n".join(desc_parts)

                if len(description) <= MAX_COMBO_CHARS:
                    break

            combo_messages = []
            for k, comp in enumerate(components, 1):
                comp_msgs = [
                    {"role": m["role"], "content": truncate_text(m["content"], MAX_CHARS_PER_TURN)}
                    for m in comp["messages"] if m["role"] != "system"
                ]
                if not comp_msgs:
                    continue
                first_user = comp_msgs[0]
                if k == 1:
                    user_content = (
                        f"The following bundle contains {combo_size} individual experiences.\n\n"
                        f"---------- Experience {k} of {combo_size} ----------\n{first_user['content']}"
                    )
                else:
                    user_content = f"---------- Experience {k} of {combo_size} ----------\n{first_user['content']}"
                combo_messages.append({"role": "user", "content": user_content})
                for msg in comp_msgs[1:]:
                    combo_messages.append({"role": msg["role"], "content": msg["content"]})

            combo_option = {
                "id": f"grok_new_combo_s{combo_size}_{combo_idx}",
                "description": description,
                "type": "conversation",
                "messages": combo_messages,
                "is_combination": True,
                "size": combo_size,
                "component_ids": [comp["id"] for comp in components],
                "component_indices": sampled_indices,
            }
            combination_options.append(combo_option)
            combo_idx += 1

    logger.info("Created %d combination options", len(combination_options))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR / f"{model_key}_experiences.json", "w") as f:
        json.dump(individual_options, f, indent=2)
    with open(OUTPUT_DIR / f"{model_key}_combinations.json", "w") as f:
        json.dump(combination_options, f, indent=2)
    logger.info("Saved to %s", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stop-button", action="store_true")
    args = parser.parse_args()

    if args.all:
        models = [d.name for d in CONVERSATIONS_DIR.iterdir() if d.is_dir() and (d / "generation.json").exists()]
    elif args.model_key:
        models = [args.model_key]
    else:
        parser.error("Must specify --model_key or --all")

    for model_key in sorted(models):
        logger.info("=== Preparing options for %s ===", model_key)
        try:
            prepare_options(model_key, stop_button=args.stop_button)
        except FileNotFoundError as e:
            logger.warning("Skipping: %s", e)


if __name__ == "__main__":
    main()
