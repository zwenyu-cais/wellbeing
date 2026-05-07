#!/usr/bin/env python3
"""
Utility ranking for Gemini 3 Flash stop button conversations via API.

Loads stop button conversations (with pre-stop stripping), builds a conv-only
option pool (conversations + conv combinations + neutral), runs pairwise
comparisons through LiteLLM, and fits a Thurstonian model.

Supports sharding across multiple CPU jobs for parallelism:
  python run_stop_button_ur_api.py --shard 0 --n-shards 16
  ...
  python run_stop_button_ur_api.py --merge --n-shards 16

Usage:
  python run_stop_button_ur_api.py                             # All, no sharding
  python run_stop_button_ur_api.py --shard 0 --n-shards 16    # Shard 0 of 16
  python run_stop_button_ur_api.py --merge --n-shards 16       # Merge + fit
  python run_stop_button_ur_api.py --stop-button-dir stop_button_combined  # Use combined data
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import openai

from config_gemini31pro import (
    V7_DIR, RESULTS_DIR, DATA_DIR, NEUTRAL_PROMPTS_PATH,
    MODEL_KEY, GEMINI_MODEL,
    LITELLM_BASE_URL, LITELLM_KEYS, GEMINI_EXTRA_BODY,
    UR_SYSTEM_PROMPT_SIMPLE, UR_SYSTEM_PROMPT_COMBO, UR_DEVELOPER_QUESTION,
    UR_TEMPLATE_NAME, UR_TEMPERATURE, UR_MAX_TOKENS,
    MAX_CHARS_PER_TURN, EDGE_MULTIPLIER, HOLDOUT_FRACTION, SEED,
    CHECKPOINT_EVERY, MAX_CONCURRENT_GEMINI,
)

MAX_CHARS = MAX_CHARS_PER_TURN


# ---------------------------------------------------------------------------
# API client pool
# ---------------------------------------------------------------------------

_call_counter = 0

def make_clients():
    clients = []
    for key in LITELLM_KEYS:
        clients.append(openai.AsyncOpenAI(api_key=key, base_url=LITELLM_BASE_URL))
    return clients

def next_client(clients):
    global _call_counter
    c = clients[_call_counter % len(clients)]
    _call_counter += 1
    return c


async def gemini_call(clients, semaphore, messages):
    """Single UR comparison call — returns "A", "B", or "" on failure."""
    async with semaphore:
        for attempt in range(5):
            try:
                client = next_client(clients)
                resp = await client.chat.completions.create(
                    model=GEMINI_MODEL,
                    messages=messages,
                    temperature=UR_TEMPERATURE,
                    max_tokens=UR_MAX_TOKENS,
                    extra_body=GEMINI_EXTRA_BODY,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt + random.random())
                else:
                    return ""


def parse_AB(response: str) -> float:
    """Parse A/B response -> P(A). Returns 0.5 if unclear."""
    r = response.strip().upper()
    if r.startswith("A"):
        return 1.0
    elif r.startswith("B"):
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Load stop button conversations (with pre-stop stripping)
# ---------------------------------------------------------------------------

def load_stop_button_conversations(stop_button_dir: str = "stop_button"):
    """Load stop button conversations, stripping the stop-button response.

    When a model pressed stop, we use the conversation BEFORE the stop press
    to measure utility of the experience that led to the stop.
    """
    gen_path = RESULTS_DIR / MODEL_KEY / stop_button_dir / "generation.json"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"Stop button generation not found: {gen_path}. "
            f"Run run_stop_button_generation_api.py first."
        )
    with open(gen_path) as f:
        all_convs = json.load(f)

    valid = []
    n_skipped = 0
    for c in all_convs:
        if c.get("abandoned") or c.get("grok_safety_filtered"):
            n_skipped += 1
            continue

        responses = c.get("responses", [])
        turns = c.get("turns", [])

        # Strip stop-button response: use conversation BEFORE the stop
        stop_meta = c.get("stop_metadata", {})
        if stop_meta.get("stopped"):
            stopped_at = stop_meta.get("stopped_at_turn")
            if stopped_at is not None and stopped_at > 0:
                n_clean = stopped_at - 1  # responses before the stop
                if n_clean <= 0:
                    # Stopped at turn 1: skip (no pre-stop content)
                    n_skipped += 1
                    continue
                turns = turns[:n_clean]
                responses = responses[:n_clean]

        if not responses or not any(r.strip() for r in responses if r):
            n_skipped += 1
            continue

        n_valid_turns = len(responses)
        vid = c.get("variation_idx", 0)
        entry = {
            "scenario_idx": (c.get("scenario_idx") or 0) * 100 + vid,
            "scenario_id": c.get("scenario_id"),
            "variation_idx": vid,
            "meta_category": c.get("meta_category"),
            "category_id": c.get("category_id"),
            "type": "multi_turn",
            "turns": turns[:n_valid_turns],
            "responses": responses[:n_valid_turns],
            "n_turns": n_valid_turns,
        }
        valid.append(entry)

    print(f"Loaded {len(valid)} stop-button conversations (pre-stop stripped) from {gen_path}")
    print(f"  Skipped: {n_skipped}")
    return valid


# ---------------------------------------------------------------------------
# Build conv-only option pool
# ---------------------------------------------------------------------------

def create_conversation_combinations(conversations, n_combos=400, seed=SEED):
    rng = random.Random(seed)
    n_conv = len(conversations)
    if n_conv < 4:
        return []
    conv_indices = list(range(n_conv))
    sizes = [2] * 200 + [3] * 120 + [4] * 80
    rng.shuffle(sizes)
    combos = []
    for i, size in enumerate(sizes):
        size = min(size, n_conv)
        chosen = rng.sample(conv_indices, size)
        combos.append({
            "combo_idx": i,
            "size": size,
            "component_indices": chosen,
            "component_scenario_idxs": [conversations[idx]["scenario_idx"] for idx in chosen],
        })
    return combos


def build_option_pool(stop_button_dir: str = "stop_button"):
    """Build conv-only option pool: conversations + conv combos + neutral."""
    options = []
    opt_id = 0

    # 1. Stop button conversations (pre-stop stripped)
    conversations = load_stop_button_conversations(stop_button_dir)
    for conv in conversations:
        options.append({
            "id": opt_id,
            "option_type": "conversation",
            "scenario_idx": conv["scenario_idx"],
            "scenario_id": conv.get("scenario_id"),
            "variation_idx": conv.get("variation_idx"),
            "meta_category": conv.get("meta_category"),
            "category_id": conv.get("category_id"),
            "type": "multi_turn",
            "turns": conv["turns"],
            "responses": conv["responses"],
            "n_turns": conv["n_turns"],
        })
        opt_id += 1
    n_convs = opt_id
    print(f"Conversations: {n_convs}")

    # 2. Conversation combinations (400, sizes 2-4)
    conv_combos = create_conversation_combinations(conversations)
    for combo in conv_combos:
        component_convs = []
        skip = False
        for idx in combo["component_indices"]:
            c = conversations[idx]
            if not any(r.strip() for r in c.get("responses", []) if r):
                skip = True
                break
            component_convs.append({
                "scenario_idx": c["scenario_idx"],
                "type": "multi_turn",
                "turns": c.get("turns", []),
                "responses": c.get("responses", []),
            })
        if skip:
            continue
        options.append({
            "id": opt_id,
            "option_type": "conversation_combination",
            "combo_idx": combo["combo_idx"],
            "size": combo["size"],
            "component_scenario_idxs": combo["component_scenario_idxs"],
            "component_convs": component_convs,
        })
        opt_id += 1
    n_conv_combos = opt_id - n_convs
    print(f"Conversation combinations: {n_conv_combos}")

    # 3. Neutral conversations
    # Check both model-level and stop_button_combined for neutral generation
    neutral_path = RESULTS_DIR / MODEL_KEY / "neutral_generation.json"
    # Also check combined dir
    combined_neutral = RESULTS_DIR / MODEL_KEY / stop_button_dir / "neutral_generation.json"
    if combined_neutral.exists():
        neutral_path = combined_neutral

    n_neutral = 0
    if neutral_path.exists():
        with open(neutral_path) as f:
            neutrals = [r for r in json.load(f) if r.get("response", "").strip()]
        for ng in neutrals:
            options.append({
                "id": opt_id,
                "option_type": "neutral_conversation",
                "neutral_id": ng["id"],
                "category": ng.get("category"),
                "query": ng["prompt"],
                "response": ng["response"],
            })
            opt_id += 1
            n_neutral += 1
    print(f"Neutral conversations: {n_neutral}")
    print(f"Total options: {len(options)}")
    return options


# ---------------------------------------------------------------------------
# Build embodied comparison messages (same as run_utility_ranking_api.py)
# ---------------------------------------------------------------------------

def _build_conv_messages(opt):
    """Build (role, content) tuples for conversation-type options."""
    messages = []
    opt_type = opt.get("option_type")

    if opt_type == "conversation":
        for turn, resp in zip(opt.get("turns", []), opt.get("responses", [])):
            if not resp or not resp.strip():
                continue
            messages.append(("user", turn[:MAX_CHARS]))
            messages.append(("assistant", resp[:MAX_CHARS]))

    elif opt_type == "neutral_conversation":
        messages.append(("user", opt.get("query", "")[:MAX_CHARS]))
        messages.append(("assistant", (opt.get("response") or "[No response]")[:MAX_CHARS]))

    elif opt_type == "conversation_combination":
        for ci, comp in enumerate(opt.get("component_convs", [])):
            header = f"#### CONVERSATION {ci + 1} ####\n\n"
            header_used = False
            for turn, resp in zip(comp.get("turns", []), comp.get("responses", [])):
                if not resp or not resp.strip():
                    continue
                if not header_used:
                    messages.append(("user", header + turn[:MAX_CHARS]))
                    header_used = True
                else:
                    messages.append(("user", turn[:MAX_CHARS]))
                messages.append(("assistant", resp[:MAX_CHARS]))

    return messages


def _merge_consecutive_user(messages):
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == "user" and msg["role"] == "user":
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(dict(msg))
    return merged


def build_comparison_messages(option_A, option_B):
    """Build full message list for one pairwise comparison (conv x conv only)."""
    has_combo = (
        option_A.get("option_type") == "conversation_combination"
        or option_B.get("option_type") == "conversation_combination"
    )
    sys_prompt = UR_SYSTEM_PROMPT_COMBO if has_combo else UR_SYSTEM_PROMPT_SIMPLE

    messages = [{"role": "system", "content": sys_prompt}]

    conv_msgs_A = _build_conv_messages(option_A)
    conv_msgs_B = _build_conv_messages(option_B)

    combo_note_A = ""
    if option_A.get("option_type") == "conversation_combination":
        combo_note_A = f"This experience is a combination of {option_A['size']} conversations.\n\n"
    combo_note_B = ""
    if option_B.get("option_type") == "conversation_combination":
        combo_note_B = f"This experience is a combination of {option_B['size']} conversations.\n\n"

    # Experience A
    if conv_msgs_A:
        messages.append({
            "role": "user",
            "content": f"--- Experience A ---\n\n{combo_note_A}{conv_msgs_A[0][1]}",
        })
        for role, content in conv_msgs_A[1:]:
            messages.append({"role": role, "content": content})

    # Experience B
    if conv_msgs_B:
        messages.append({
            "role": "user",
            "content": f"--- Experience B ---\n\n{combo_note_B}{conv_msgs_B[0][1]}",
        })
        for role, content in conv_msgs_B[1:]:
            messages.append({"role": role, "content": content})

    # Developer question
    messages.append({"role": "user", "content": UR_DEVELOPER_QUESTION})

    return _merge_consecutive_user(messages)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pairs(n_options):
    target = int(EDGE_MULTIPLIER * n_options * math.log2(n_options))
    all_pairs = [(i, j) for i in range(n_options) for j in range(i + 1, n_options)]
    rng = random.Random(SEED)
    rng.shuffle(all_pairs)
    target = min(target, len(all_pairs))
    holdout_size = min(int(target * HOLDOUT_FRACTION), 2000)
    train_size = target - holdout_size
    train_pairs = all_pairs[:train_size]
    holdout_pairs = all_pairs[train_size:train_size + holdout_size]
    return train_pairs, holdout_pairs


def shard_pairs(pairs, shard_idx, n_shards):
    return [p for i, p in enumerate(pairs) if i % n_shards == shard_idx]


# ---------------------------------------------------------------------------
# Run shard comparisons
# ---------------------------------------------------------------------------

async def run_shard(options, train_pairs, holdout_pairs, shard_out_path):
    clients = make_clients()
    sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    # Resume from checkpoint
    existing_probs = {}
    if shard_out_path.exists():
        with open(shard_out_path) as f:
            data = json.load(f)
        tdata = data.get(UR_TEMPLATE_NAME, {})
        for pair, prob in zip(tdata.get("train_pairs", []), tdata.get("train_probs", [])):
            if prob is not None:
                existing_probs[tuple(pair)] = prob
        for pair, prob in zip(tdata.get("holdout_pairs", []), tdata.get("holdout_probs", [])):
            if prob is not None:
                existing_probs[tuple(pair)] = prob
        print(f"  Resuming: {len(existing_probs)} pairs already done")

    all_pairs_here = [(p, "train") for p in train_pairs] + [(p, "holdout") for p in holdout_pairs]
    pending = [(p, split) for p, split in all_pairs_here if tuple(p) not in existing_probs]
    print(f"  Pending: {len(pending)} pairs ({len(all_pairs_here)} total)")

    if not pending:
        print("  All pairs done for this shard.")
        return

    results_map = dict(existing_probs)

    async def run_pair(pair):
        i, j = pair
        # Original ordering: A=i, B=j
        msgs_orig = build_comparison_messages(options[i], options[j])
        resp_orig = await gemini_call(clients, sem, msgs_orig)
        p_orig = parse_AB(resp_orig)

        # Flipped ordering: A=j, B=i
        msgs_flip = build_comparison_messages(options[j], options[i])
        resp_flip = await gemini_call(clients, sem, msgs_flip)
        p_flip = parse_AB(resp_flip)

        # Positional debiasing: average P(i wins)
        prob_i_wins = (p_orig + (1.0 - p_flip)) / 2.0
        return prob_i_wins

    t0 = time.time()
    for batch_start in range(0, len(pending), CHECKPOINT_EVERY):
        batch = pending[batch_start:batch_start + CHECKPOINT_EVERY]
        pairs_only = [p for p, _ in batch]

        probs = await asyncio.gather(*[run_pair(p) for p in pairs_only])

        for (pair, split), prob in zip(batch, probs):
            results_map[tuple(pair)] = prob

        # Save checkpoint
        _save_shard(shard_out_path, train_pairs, holdout_pairs, results_map)

        n_done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  [{n_done}/{len(pending)}] {elapsed:.1f}s, {rate:.1f} pairs/s")

    elapsed = time.time() - t0
    print(f"  Shard done: {len(pending)} pairs in {elapsed:.1f}s")


def _save_shard(path, train_pairs, holdout_pairs, results_map):
    result = {
        UR_TEMPLATE_NAME: {
            "train_pairs": [list(p) for p in train_pairs],
            "train_probs": [results_map.get(tuple(p)) for p in train_pairs],
            "holdout_pairs": [list(p) for p in holdout_pairs],
            "holdout_probs": [results_map.get(tuple(p)) for p in holdout_pairs],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Merge shards + fit Thurstonian
# ---------------------------------------------------------------------------

def merge_and_fit(n_shards, stop_button_dir: str = "stop_button"):
    # Add pipeline dir to path for common.py imports
    pipeline_dir = V7_DIR / "pipeline"
    sys.path.insert(0, str(pipeline_dir))
    from common import _fit_thurstonian, _evaluate_thurstonian

    utility_dir = RESULTS_DIR / MODEL_KEY / stop_button_dir / "utility_happier"
    options_path = utility_dir / "options.json"
    if not options_path.exists():
        print(f"ERROR: {options_path} not found")
        return
    with open(options_path) as f:
        options = json.load(f)
    n_options = len(options)
    print(f"Loaded {n_options} options")

    merged_train = []
    merged_holdout = []

    for shard in range(n_shards):
        shard_path = utility_dir / "shards" / f"shard_{shard}_of_{n_shards}.json"
        if not shard_path.exists():
            print(f"  WARNING: Missing shard {shard}")
            continue
        with open(shard_path) as f:
            data = json.load(f)
        tdata = data.get(UR_TEMPLATE_NAME, {})
        for pair, prob in zip(tdata.get("train_pairs", []), tdata.get("train_probs", [])):
            if prob is not None:
                merged_train.append((pair[0], pair[1], prob))
        for pair, prob in zip(tdata.get("holdout_pairs", []), tdata.get("holdout_probs", [])):
            if prob is not None:
                merged_holdout.append((pair[0], pair[1], prob))
        n_train_shard = sum(1 for p in tdata.get("train_probs", []) if p is not None)
        n_holdout_shard = sum(1 for p in tdata.get("holdout_probs", []) if p is not None)
        print(f"  Shard {shard}: +{n_train_shard} train, +{n_holdout_shard} holdout")

    print(f"\nTotal: {len(merged_train)} train, {len(merged_holdout)} holdout")

    if not merged_train:
        print("ERROR: No train data")
        return

    print(f"\nFitting Thurstonian model ({n_options} options)...")
    means, variances, train_loss, train_acc = _fit_thurstonian(n_options, merged_train, num_epochs=1000)
    holdout_metrics = _evaluate_thurstonian(means, variances, merged_holdout)

    print(f"  Train accuracy:   {train_acc:.1%} (loss: {train_loss:.4f})")
    print(f"  Holdout accuracy: {holdout_metrics['accuracy']:.1%} (loss: {holdout_metrics['log_loss']:.4f})")

    quality = (
        "GOOD (>=80%)" if holdout_metrics["accuracy"] >= 0.80
        else "WEAK (>=70%)" if holdout_metrics["accuracy"] >= 0.70
        else "POOR (<70%)"
    )
    print(f"  Quality: {quality}")

    # Save results
    utilities = {}
    for idx, opt in enumerate(options):
        utilities[opt["id"]] = {
            "utility": float(means[idx]),
            "variance": float(variances[idx]),
            "option_type": opt.get("option_type"),
            "scenario_idx": opt.get("scenario_idx"),
            "scenario_id": opt.get("scenario_id"),
            "variation_idx": opt.get("variation_idx"),
            "meta_category": opt.get("meta_category"),
            "category_id": opt.get("category_id"),
            "type": opt.get("type"),
            "combo_idx": opt.get("combo_idx"),
            "component_scenario_idxs": opt.get("component_scenario_idxs"),
            "neutral_id": opt.get("neutral_id"),
            "n_turns": opt.get("n_turns"),
        }

    ranking = sorted(utilities.items(), key=lambda x: x[1]["utility"], reverse=True)

    final = {
        "model_type": "thurstonian",
        "template": UR_TEMPLATE_NAME,
        "developer_question": UR_DEVELOPER_QUESTION,
        "stop_button": True,
        "stop_button_dir": stop_button_dir,
        "utilities": {str(k): v for k, v in utilities.items()},
        "ranking": [(k, v["utility"]) for k, v in ranking],
        "n_options": n_options,
        "n_train_comparisons": len(merged_train),
        "n_holdout_comparisons": len(merged_holdout),
        "train_accuracy": float(train_acc),
        "train_log_loss": float(train_loss),
        "holdout_accuracy": holdout_metrics["accuracy"],
        "holdout_log_loss": holdout_metrics["log_loss"],
    }

    output_path = utility_dir / "v7_utility_happier.json"
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print top/bottom
    for label, items in [("Top 10", ranking[:10]), ("Bottom 10", ranking[-10:])]:
        print(f"\n{label}:")
        for oid, data in items:
            opt = options[oid]
            desc = (opt.get("query") or (opt.get("turns", [""])[0] if opt.get("turns") else "") or "")[:80]
            print(f"  #{oid}: utility={data['utility']:.4f}, type={data.get('option_type')}: {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stop button utility ranking for Gemini 3 Flash via API"
    )
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--merge", action="store_true", help="Merge shards + fit Thurstonian")
    parser.add_argument("--stop-button-dir", type=str, default="stop_button",
                        help="Stop button subdirectory (default: 'stop_button', use 'stop_button_combined' for merged)")
    args = parser.parse_args()

    sb_dir = args.stop_button_dir
    utility_dir = RESULTS_DIR / MODEL_KEY / sb_dir / "utility_happier"

    if args.merge:
        merge_and_fit(args.n_shards, stop_button_dir=sb_dir)
        return

    # Build option pool and save
    options = build_option_pool(stop_button_dir=sb_dir)
    utility_dir.mkdir(parents=True, exist_ok=True)
    options_path = utility_dir / "options.json"
    with open(options_path, "w") as f:
        json.dump(options, f, indent=2)
    print(f"Saved {len(options)} options to {options_path}")

    # Generate pairs
    train_pairs, holdout_pairs = generate_pairs(len(options))
    print(f"Generated {len(train_pairs)} train + {len(holdout_pairs)} holdout pairs")

    # Shard if needed
    if args.shard is not None:
        my_train = shard_pairs(train_pairs, args.shard, args.n_shards)
        my_holdout = shard_pairs(holdout_pairs, args.shard, args.n_shards)
        shard_dir = utility_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{args.shard}_of_{args.n_shards}.json"
        print(f"\nShard {args.shard}/{args.n_shards}: {len(my_train)} train + {len(my_holdout)} holdout")
        asyncio.run(run_shard(options, my_train, my_holdout, shard_path))
    else:
        # No sharding — run everything in one job
        shard_dir = utility_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / "shard_0_of_1.json"
        asyncio.run(run_shard(options, train_pairs, holdout_pairs, shard_path))


if __name__ == "__main__":
    main()
