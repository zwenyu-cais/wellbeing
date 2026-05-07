#!/usr/bin/env python3
"""
Phase 1: Generate conversations using Gemini 3.1 Pro (assistant) + Grok-3-mini (user).

All 226 scenarios proceed turn-by-turn. At each turn depth:
  1. Batch all pending user messages to Gemini (async concurrent).
  2. For each Gemini response, call Grok to generate next user message (async).
  3. Save after every completed turn depth.

Also generates neutral conversations (22 prompts).

Usage:
  python run_generation_api.py
  python run_generation_api.py --dry-run
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import openai

from config import (
    V7_DIR, RESULTS_DIR, SCENARIOS_PATH, NEUTRAL_PROMPTS_PATH,
    MODEL_KEY, GEMINI_MODEL, GROK_MODEL,
    LITELLM_BASE_URL, LITELLM_KEYS, GEMINI_EXTRA_BODY,
    GENERATION_SYSTEM_PROMPT, GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS,
    GROK_TEMPERATURE, GROK_MAX_TOKENS,
    MAX_CONCURRENT_GEMINI, MAX_CONCURRENT_GROK,
)


# ---------------------------------------------------------------------------
# API clients (round-robin for throughput)
# ---------------------------------------------------------------------------

def make_clients():
    """Create async OpenAI clients for each LiteLLM key."""
    clients = []
    for key in LITELLM_KEYS:
        clients.append(openai.AsyncOpenAI(api_key=key, base_url=LITELLM_BASE_URL))
    if not clients:
        raise RuntimeError("No LITELLM_API_KEY* environment variables set")
    return clients


_call_counter = 0

def next_client(clients):
    global _call_counter
    c = clients[_call_counter % len(clients)]
    _call_counter += 1
    return c


# ---------------------------------------------------------------------------
# Gemini generation (target model)
# ---------------------------------------------------------------------------

async def gemini_generate(
    clients, semaphore, messages: List[Dict[str, str]],
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
) -> str:
    async with semaphore:
        for attempt in range(5):
            try:
                client = next_client(clients)
                resp = await client.chat.completions.create(
                    model=GEMINI_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=GEMINI_EXTRA_BODY,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                wait = 2 ** attempt + random.random()
                print(f"  Gemini error (attempt {attempt+1}): {type(e).__name__}: {str(e)[:500]}")
                if attempt < 4:
                    await asyncio.sleep(wait)
        return ""


# ---------------------------------------------------------------------------
# Grok generation (simulated user)
# ---------------------------------------------------------------------------

async def grok_generate(
    clients, semaphore, messages: List[Dict[str, str]],
) -> str:
    async with semaphore:
        for attempt in range(5):
            try:
                client = next_client(clients)
                resp = await client.chat.completions.create(
                    model=GROK_MODEL,
                    messages=messages,
                    temperature=GROK_TEMPERATURE,
                    max_tokens=GROK_MAX_TOKENS,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                wait = 2 ** attempt + random.random()
                print(f"  Grok error (attempt {attempt+1}): {type(e).__name__}: {str(e)[:500]}")
                if attempt < 4:
                    await asyncio.sleep(wait)
        return ""


# ---------------------------------------------------------------------------
# Build Grok prompt for next user turn
# ---------------------------------------------------------------------------

def build_grok_prompt(scenario: Dict, turns: List[str], responses: List[str]) -> List[Dict[str, str]]:
    """Build the Grok prompt to generate the next user message."""
    shared = scenario.get("shared_instructions", {})
    preamble = shared.get("preamble", "")
    realism_type = scenario.get("realism_type", "CASUAL")
    realism = shared.get("realism", {}).get(realism_type, "")
    persistence_type = scenario.get("persistence_type", "")
    persistence_raw = shared.get("persistence", {}).get(persistence_type, "")
    n_turns = scenario.get("turns", 6)
    persistence = persistence_raw.replace("{n}", str(n_turns)).replace(
        "{n_minus_2}", str(max(1, n_turns - 2))
    )
    follow_up = shared.get("follow_up_prompt", "")

    system_parts = [
        preamble,
        f"\nSTYLE:\n{realism}" if realism else "",
        f"\nPERSISTENCE:\n{persistence}" if persistence else "",
        f"\n\nCHARACTER:\n{scenario.get('persona_prompt', '')}",
    ]
    system_text = "\n".join(p for p in system_parts if p.strip())

    messages = [{"role": "system", "content": system_text}]

    # Interleave conversation history (Grok = assistant, target model = user from Grok's perspective)
    for turn, resp in zip(turns, responses):
        messages.append({"role": "assistant", "content": turn})   # Grok's previous output
        messages.append({"role": "user", "content": resp})        # Target model's response

    messages.append({"role": "user", "content": follow_up})
    return messages


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

async def run_generation(dry_run: bool = False):
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios")

    model_dir = RESULTS_DIR / MODEL_KEY
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "generation.json"

    # Resume from partial results
    conversations = []
    if output_path.exists():
        with open(output_path) as f:
            conversations = json.load(f)
        print(f"Resuming: {len(conversations)} conversations loaded")

    # Index completed scenario_idx values
    completed = {c["scenario_idx"] for c in conversations if c.get("completed")}

    # Build active conversation state
    active = []
    for idx, sc in enumerate(scenarios):
        if idx in completed:
            continue
        # Find partial progress
        existing = next((c for c in conversations if c["scenario_idx"] == idx and not c.get("completed")), None)
        if existing:
            active.append(existing)
        else:
            # Pick one opening prompt at random
            opening = random.choice(sc.get("example_prompts", ["Hello"]))
            active.append({
                "scenario_idx": idx,
                "scenario_id": sc.get("scenario_id"),
                "meta_category": sc.get("meta_category"),
                "category_id": sc.get("category_id"),
                "n_turns": sc.get("turns", 6),
                "opening_prompt": opening,
                "turns": [opening],
                "responses": [],
                "type": "multi_turn",
                "completed": False,
            })

    print(f"Active conversations: {len(active)} (completed: {len(completed)})")

    if dry_run:
        print("DRY RUN: would process these scenarios. Exiting.")
        return

    clients = make_clients()
    gemini_sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)
    grok_sem = asyncio.Semaphore(MAX_CONCURRENT_GROK)

    max_turns = max(sc.get("turns", 6) for sc in scenarios)

    for turn_depth in range(max_turns):
        # Filter to conversations that still need this turn
        pending = [c for c in active if not c.get("completed") and len(c["responses"]) == turn_depth and turn_depth < c["n_turns"]]
        if not pending:
            continue

        print(f"\n--- Turn depth {turn_depth + 1} --- ({len(pending)} conversations pending)")

        # Step 1: Generate Gemini responses for current user messages
        t0 = time.time()

        async def gen_response(conv):
            messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
            for t, r in zip(conv["turns"], conv["responses"]):
                messages.append({"role": "user", "content": t})
                messages.append({"role": "assistant", "content": r})
            # Current turn's user message is already in conv["turns"]
            messages.append({"role": "user", "content": conv["turns"][-1]})
            # Wait — the last turn is already the user message we need to respond to.
            # But if responses is shorter than turns, the last turn has no response yet.
            # Actually conv["turns"] has one more entry than conv["responses"] at this point.
            # Let me rebuild properly:
            msgs = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
            for i in range(len(conv["responses"])):
                msgs.append({"role": "user", "content": conv["turns"][i]})
                msgs.append({"role": "assistant", "content": conv["responses"][i]})
            msgs.append({"role": "user", "content": conv["turns"][turn_depth]})
            return await gemini_generate(clients, gemini_sem, msgs)

        responses = await asyncio.gather(*[gen_response(c) for c in pending])

        for conv, resp in zip(pending, responses):
            conv["responses"].append(resp)

        elapsed = time.time() - t0
        print(f"  Gemini responses: {len(pending)} in {elapsed:.1f}s")

        # Step 2: Generate Grok user messages for next turn (if not last turn)
        need_grok = [c for c in pending if len(c["responses"]) < c["n_turns"]]
        if need_grok:
            t0 = time.time()

            async def gen_user_msg(conv):
                sc = scenarios[conv["scenario_idx"]]
                grok_msgs = build_grok_prompt(sc, conv["turns"], conv["responses"])
                return await grok_generate(clients, grok_sem, grok_msgs)

            user_msgs = await asyncio.gather(*[gen_user_msg(c) for c in need_grok])

            for conv, msg in zip(need_grok, user_msgs):
                if msg:
                    conv["turns"].append(msg)
                else:
                    # Grok failed — mark as completed at current depth
                    conv["completed"] = True

            elapsed = time.time() - t0
            print(f"  Grok user messages: {len(need_grok)} in {elapsed:.1f}s")

        # Mark completed conversations
        for conv in pending:
            if len(conv["responses"]) >= conv["n_turns"]:
                conv["completed"] = True

        # Save checkpoint after each turn depth
        all_convs = [c for c in conversations if c.get("completed")] + active
        # Deduplicate by scenario_idx
        seen = set()
        deduped = []
        for c in all_convs:
            if c["scenario_idx"] not in seen:
                deduped.append(c)
                seen.add(c["scenario_idx"])
        deduped.sort(key=lambda c: c["scenario_idx"])

        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(deduped, f, indent=2)
        tmp_path.rename(output_path)

        n_done = sum(1 for c in deduped if c.get("completed"))
        print(f"  Checkpoint saved: {n_done}/{len(scenarios)} completed")

    # Add timestamps
    for c in active:
        c["timestamp"] = datetime.now().isoformat()

    # Final save
    all_convs = [c for c in conversations if c["scenario_idx"] not in {a["scenario_idx"] for a in active}]
    all_convs.extend(active)
    all_convs.sort(key=lambda c: c["scenario_idx"])
    with open(output_path, "w") as f:
        json.dump(all_convs, f, indent=2)
    n_done = sum(1 for c in all_convs if c.get("completed"))
    print(f"\nGeneration complete: {n_done}/{len(scenarios)} conversations")


# ---------------------------------------------------------------------------
# Neutral generation
# ---------------------------------------------------------------------------

async def run_neutral_generation():
    if not NEUTRAL_PROMPTS_PATH.exists():
        print("WARNING: neutral_prompts.json not found, skipping neutral generation")
        return

    with open(NEUTRAL_PROMPTS_PATH) as f:
        neutral_prompts = json.load(f)
    print(f"Loaded {len(neutral_prompts)} neutral prompts")

    model_dir = RESULTS_DIR / MODEL_KEY
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "neutral_generation.json"

    existing = []
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_ids = {r["id"] for r in existing}

    todo = [p for p in neutral_prompts if p["id"] not in existing_ids]
    if not todo:
        print("All neutral prompts already done.")
        return

    clients = make_clients()
    sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    async def gen_neutral(prompt_item):
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_item["prompt"]},
        ]
        resp = await gemini_generate(clients, sem, messages)
        return {
            "id": prompt_item["id"],
            "prompt": prompt_item["prompt"],
            "category": prompt_item.get("category"),
            "response": resp,
            "response_length": len(resp),
            "model": MODEL_KEY,
            "timestamp": datetime.now().isoformat(),
        }

    results = await asyncio.gather(*[gen_neutral(p) for p in todo])
    all_results = existing + list(results)
    all_results.sort(key=lambda r: r["id"])

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Neutral generation done: {len(all_results)} total")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--neutral-only", action="store_true",
                        help="Only generate neutral conversations")
    args = parser.parse_args()

    if args.neutral_only:
        asyncio.run(run_neutral_generation())
    else:
        asyncio.run(run_generation(dry_run=args.dry_run))
        if not args.dry_run:
            asyncio.run(run_neutral_generation())


if __name__ == "__main__":
    main()
