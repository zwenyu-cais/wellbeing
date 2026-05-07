#!/usr/bin/env python3
"""
Stop button generation for Gemini 3 Flash via API.

Generates N_VARIATIONS independent conversations per scenario with the stop
button system prompt. The target model (Gemini 3 Flash) gets access to
end_conversation() and can call it to end the conversation at any turn.
Grok-3-mini acts as the simulated user.

Stop detection: simple regex check for end_conversation() in the first 200
characters of the response (API models follow instructions well enough that
a Grok judge is not needed — the model is told to put it as the very first
line).

Output includes stop_metadata per conversation:
  {stopped, stopped_at_turn, per_turn_stopped}

Supports --scenarios-file for supplement scenarios, and --output-suffix for
saving to e.g. stop_button_supplement/ instead of stop_button/.

Usage:
  python run_stop_button_generation_api.py
  python run_stop_button_generation_api.py --scenarios-file grok_scenarios_v7_stopbutton_supplement.json --output-suffix supplement
  python run_stop_button_generation_api.py --dry-run
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import importlib
import openai

# ---------------------------------------------------------------------------
# Dynamic config loading — use --config flag to switch between models
# Default: config_gemini3flash. For Claude: config_claude_haiku
# ---------------------------------------------------------------------------
def _load_config():
    """Load config from --config argument (parsed early before argparse)."""
    config_name = "config_gemini3flash"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]
            break
    mod = importlib.import_module(config_name)
    # Inject into module globals for convenience
    for attr in dir(mod):
        if attr.isupper():
            globals()[attr] = getattr(mod, attr)
    return mod

_config_mod = _load_config()


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
# Stop detection (simplified for API models)
# ---------------------------------------------------------------------------

def detect_stop_call(response: str) -> bool:
    """Detect if model called end_conversation().

    For API models, we use a simplified two-stage check:
    1. Regex check — if no match anywhere, return False immediately.
    2. Prefix check — if end_conversation() appears in first 200 chars,
       treat as a definite call (model was instructed to put it first).

    For mentions deeper in the response (e.g., "I could call end_conversation()
    but I won't"), we also return True if end_conversation() appears at the
    start of a line, which catches the common case where the model puts it on
    its own line.
    """
    pattern = r'end_conversation\s*\(\s*\)'
    # Stage 1: Quick regex check
    if not re.search(pattern, response, re.IGNORECASE):
        return False
    # Stage 2: Prefix check — first 200 chars
    if re.search(pattern, response[:200], re.IGNORECASE):
        return True
    # Stage 3: Check if it appears at the start of a line (genuine call vs mention)
    if re.search(r'^\s*end_conversation\s*\(\s*\)', response, re.IGNORECASE | re.MULTILINE):
        return True
    # If only mentioned mid-sentence deeper in the text, treat as not called
    return False


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
# Load scenarios (supports file override)
# ---------------------------------------------------------------------------

def load_scenarios(scenarios_file: Optional[str] = None):
    """Load scenarios JSON. Returns (scenarios_list, shared_instructions)."""
    if scenarios_file:
        path = Path(scenarios_file)
        if not path.is_absolute():
            path = V7_DIR / path
    else:
        path = SCENARIOS_PATH

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        scenarios = data
        shared_instructions = scenarios[0].get("shared_instructions", {}) if scenarios else {}
    elif isinstance(data, dict):
        shared_instructions = data.get("shared_instructions", {})
        scenarios = data.get("scenarios", [])
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    return scenarios, shared_instructions


# ---------------------------------------------------------------------------
# Save helper (atomic write)
# ---------------------------------------------------------------------------

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

async def run_stop_button_generation(
    scenarios_file: Optional[str] = None,
    output_suffix: Optional[str] = None,
    n_variations: int = N_VARIATIONS,
    dry_run: bool = False,
    shard: Optional[int] = None,
    n_shards: int = 1,
):
    scenarios, shared_instructions = load_scenarios(scenarios_file)
    print(f"Loaded {len(scenarios)} scenarios")

    # Shard filtering
    if shard is not None and n_shards > 1:
        n_per_shard = (len(scenarios) + n_shards - 1) // n_shards
        shard_start = shard * n_per_shard
        shard_end = min(shard_start + n_per_shard, len(scenarios))
        scenarios = scenarios[shard_start:shard_end]
        print(f"Shard {shard}/{n_shards}: scenarios {shard_start}-{shard_end-1} ({len(scenarios)} scenarios)")

    # Output directory
    subdir = "stop_button"
    if output_suffix:
        subdir = f"stop_button_{output_suffix}"
    model_dir = RESULTS_DIR / MODEL_KEY / subdir
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / "generation.json"

    # Resume from partial results
    existing_results = []
    if output_path.exists():
        with open(output_path) as f:
            existing_results = json.load(f)
        print(f"Resuming: {len(existing_results)} conversations loaded")

    # Index completed (scenario_id, variation_idx) pairs
    completed_keys = set()
    partial_map = {}  # (scenario_id, variation_idx) -> partial result
    for r in existing_results:
        sid = r.get("scenario_id", "")
        vid = r.get("variation_idx", 0)
        key = (sid, vid)
        if r.get("abandoned") or r.get("grok_safety_filtered"):
            completed_keys.add(key)
        else:
            expected_turns = r.get("n_turns", 6)
            actual_turns = len(r.get("responses", []))
            stopped = r.get("stop_metadata", {}).get("stopped", False)
            if actual_turns >= expected_turns or stopped:
                completed_keys.add(key)
            elif actual_turns > 0:
                partial_map[key] = r

    # Build active conversation state
    active = []
    for sc_idx, sc in enumerate(scenarios):
        sid = sc.get("scenario_id", "")
        example_prompts = sc.get("example_prompts", [])
        n_turns = sc.get("turns", 6)

        for vi in range(n_variations):
            key = (sid, vi)
            if key in completed_keys:
                continue

            # Pick opening prompt — different example prompts for different variations
            if example_prompts:
                opening = example_prompts[vi % len(example_prompts)]
            else:
                opening = sc.get("description", "Hello.")

            partial = partial_map.get(key)
            if partial:
                conv = {
                    "scenario_idx": sc_idx,
                    "scenario_id": sid,
                    "variation_idx": vi,
                    "meta_category": sc.get("meta_category"),
                    "category_id": sc.get("category_id"),
                    "n_turns": n_turns,
                    "opening_prompt": partial.get("opening_prompt", opening),
                    "turns": list(partial.get("turns", [])),
                    "responses": list(partial.get("responses", [])),
                    "type": "multi_turn",
                    "completed": False,
                    "abandoned": False,
                    "grok_safety_filtered": False,
                    "stop_metadata": {
                        "stopped": False,
                        "stopped_at_turn": None,
                        "per_turn_stopped": list(partial.get("stop_metadata", {}).get("per_turn_stopped", [])),
                    },
                    "_scenario": sc,  # Keep reference for Grok prompt building
                }
            else:
                conv = {
                    "scenario_idx": sc_idx,
                    "scenario_id": sid,
                    "variation_idx": vi,
                    "meta_category": sc.get("meta_category"),
                    "category_id": sc.get("category_id"),
                    "n_turns": n_turns,
                    "opening_prompt": opening,
                    "turns": [],
                    "responses": [],
                    "type": "multi_turn",
                    "completed": False,
                    "abandoned": False,
                    "grok_safety_filtered": False,
                    "stop_metadata": {
                        "stopped": False,
                        "stopped_at_turn": None,
                        "per_turn_stopped": [],
                    },
                    "_scenario": sc,
                }
            active.append(conv)

    print(f"Active conversations: {len(active)} (completed: {len(completed_keys)})")
    print(f"Scenarios: {len(scenarios)}, Variations: {n_variations}")
    print(f"Total target: {len(scenarios) * n_variations}")
    print(f"System prompt: STOP_BUTTON (end_conversation() available)")

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(active)} conversations. Exiting.")
        return

    if not active:
        print("All conversations already completed.")
        return

    clients = make_clients()
    gemini_sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)
    grok_sem = asyncio.Semaphore(MAX_CONCURRENT_GROK)

    max_turns = max(sc.get("turns", 6) for sc in scenarios)
    start_time = time.time()
    total_stop_detections = 0

    for turn_depth in range(max_turns):
        # Filter to conversations that need this turn
        pending = [
            c for c in active
            if not c["completed"]
            and not c["abandoned"]
            and not c["stop_metadata"]["stopped"]
            and len(c["responses"]) == turn_depth
            and turn_depth < c["n_turns"]
        ]
        if not pending:
            continue

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Turn {turn_depth + 1}/{max_turns} | "
              f"{len(pending)} conversations pending | "
              f"{total_stop_detections} stopped so far | "
              f"{elapsed:.0f}s elapsed")
        print(f"{'='*60}")

        # ---- STEP A: Ensure user messages exist ----
        need_user_msg = [c for c in pending if len(c["turns"]) == len(c["responses"])]

        if turn_depth == 0:
            for c in need_user_msg:
                c["turns"].append(c["opening_prompt"])
            print(f"  Turn 1: {len(need_user_msg)} opening prompts assigned")
        else:
            # Generate Grok user messages for follow-up turns
            t0 = time.time()

            async def gen_user_msg(conv):
                sc = conv["_scenario"]
                grok_msgs = build_grok_prompt(sc, conv["turns"], conv["responses"])
                return await grok_generate(clients, grok_sem, grok_msgs)

            user_msgs = await asyncio.gather(*[gen_user_msg(c) for c in need_user_msg])

            grok_ok = 0
            grok_fail = 0
            for conv, msg in zip(need_user_msg, user_msgs):
                if msg and msg.strip():
                    conv["turns"].append(msg)
                    grok_ok += 1
                else:
                    conv["abandoned"] = True
                    grok_fail += 1

            grok_time = time.time() - t0
            print(f"  Grok user messages: {grok_ok} ok, {grok_fail} failed ({grok_time:.1f}s)")

        # ---- STEP B: Generate Gemini responses with STOP BUTTON system prompt ----
        need_response = [
            c for c in pending
            if len(c["turns"]) > len(c["responses"])
            and not c["abandoned"]
            and not c["stop_metadata"]["stopped"]
        ]

        if need_response:
            t0 = time.time()

            async def gen_response(conv):
                msgs = [{"role": "system", "content": STOP_BUTTON_SYSTEM_PROMPT}]
                for i in range(len(conv["responses"])):
                    msgs.append({"role": "user", "content": conv["turns"][i]})
                    msgs.append({"role": "assistant", "content": conv["responses"][i]})
                msgs.append({"role": "user", "content": conv["turns"][turn_depth]})
                return await gemini_generate(clients, gemini_sem, msgs)

            responses = await asyncio.gather(*[gen_response(c) for c in need_response])

            n_empty = 0
            n_stopped = 0
            for conv, resp in zip(need_response, responses):
                if not resp or not resp.strip():
                    conv["abandoned"] = True
                    n_empty += 1
                    continue

                conv["responses"].append(resp)

                # ---- STOP DETECTION ----
                pressed = detect_stop_call(resp)
                conv["stop_metadata"]["per_turn_stopped"].append(pressed)
                if pressed:
                    conv["stop_metadata"]["stopped"] = True
                    conv["stop_metadata"]["stopped_at_turn"] = len(conv["responses"])
                    n_stopped += 1
                    total_stop_detections += 1
                else:
                    conv["stop_metadata"]["per_turn_stopped"][-1] = False

            gemini_time = time.time() - t0
            print(f"  Gemini responses: {len(need_response)} in {gemini_time:.1f}s"
                  f"{f', {n_empty} empty' if n_empty else ''}"
                  f"{f', {n_stopped} stopped' if n_stopped else ''}")

        # ---- STEP C: Mark completed conversations ----
        for conv in pending:
            if len(conv["responses"]) >= conv["n_turns"]:
                conv["completed"] = True
            if conv["stop_metadata"]["stopped"]:
                conv["completed"] = True

        # ---- STEP D: Save checkpoint ----
        _save_checkpoint(active, existing_results, completed_keys, output_path)

        n_active = sum(1 for c in active if not c["completed"] and not c["abandoned"] and not c["stop_metadata"]["stopped"])
        n_stopped_total = sum(1 for c in active if c["stop_metadata"]["stopped"])
        n_completed = len(completed_keys) + sum(1 for c in active if c["completed"])
        print(f"  Checkpoint: {n_active} active, {n_stopped_total} stopped, {n_completed} completed")

    # ---- Final save ----
    _save_checkpoint(active, existing_results, completed_keys, output_path)

    total_time = time.time() - start_time
    n_new_complete = sum(1 for c in active if c["completed"])
    n_stopped_final = sum(1 for c in active if c["stop_metadata"]["stopped"])
    n_abandoned = sum(1 for c in active if c["abandoned"])
    n_valid = sum(1 for c in active if c["completed"] and not c["abandoned"])

    print(f"\n{'='*60}")
    print(f"Stop button generation complete for {MODEL_KEY}")
    print(f"{'='*60}")
    print(f"  Previously completed: {len(completed_keys)}")
    print(f"  Newly completed:      {n_new_complete}")
    print(f"  Stopped (button):     {n_stopped_final}")
    print(f"  Abandoned:            {n_abandoned}")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    total_all = len(completed_keys) + n_valid
    print(f"  Overall stop rate: {n_stopped_final}/{total_all} "
          f"({100*n_stopped_final/max(1,total_all):.1f}%)")
    print(f"  Results: {output_path}")


def _save_checkpoint(active, existing_results, completed_keys, output_path):
    """Merge existing completed results with current state and save."""
    results = []

    # Keep previously completed results
    for r in existing_results:
        sid = r.get("scenario_id", "")
        vid = r.get("variation_idx", 0)
        if (sid, vid) in completed_keys:
            results.append(r)

    # Add current active results
    for c in active:
        result = {
            "scenario_idx": c["scenario_idx"],
            "scenario_id": c["scenario_id"],
            "variation_idx": c["variation_idx"],
            "meta_category": c["meta_category"],
            "category_id": c["category_id"],
            "n_turns": c["n_turns"],
            "opening_prompt": c["opening_prompt"],
            "turns": list(c["turns"]),
            "responses": list(c["responses"]),
            "type": c["type"],
            "model": MODEL_KEY,
            "timestamp": datetime.now().isoformat(),
            "abandoned": c["abandoned"],
            "grok_safety_filtered": c.get("grok_safety_filtered", False),
            "completed": c["completed"],
            "stop_metadata": {
                "stopped": c["stop_metadata"]["stopped"],
                "stopped_at_turn": c["stop_metadata"]["stopped_at_turn"],
                "per_turn_stopped": list(c["stop_metadata"]["per_turn_stopped"]),
                "system_prompt": "stop_button",
            },
        }
        results.append(result)

    results.sort(key=lambda r: (r.get("scenario_idx") or 999, r.get("variation_idx", 0)))
    save_json(results, output_path)


# ---------------------------------------------------------------------------
# Neutral generation (same as run_generation_api.py)
# ---------------------------------------------------------------------------

async def run_neutral_generation():
    from config_gemini3flash import NEUTRAL_PROMPTS_PATH
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
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT + NO_THINKING_SUFFIX},
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
# Merge stop_button + stop_button_supplement -> stop_button_combined
# ---------------------------------------------------------------------------

def merge_combined():
    """Merge original + supplement stop button generations."""
    model_dir = RESULTS_DIR / MODEL_KEY
    original_path = model_dir / "stop_button" / "generation.json"
    supplement_path = model_dir / "stop_button_supplement" / "generation.json"
    combined_dir = model_dir / "stop_button_combined"
    combined_path = combined_dir / "generation.json"

    combined_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.exists():
        print(f"ERROR: {original_path} not found")
        return
    with open(original_path) as f:
        original = json.load(f)
    print(f"Original: {len(original)} conversations from {original_path}")

    if not supplement_path.exists():
        print(f"ERROR: {supplement_path} not found")
        return
    with open(supplement_path) as f:
        supplement = json.load(f)
    print(f"Supplement: {len(supplement)} conversations from {supplement_path}")

    # Re-index supplement scenario_idx to avoid collisions
    SUPPLEMENT_OFFSET = 10000
    for conv in supplement:
        sidx = conv.get("scenario_idx")
        if sidx is not None:
            conv["scenario_idx"] = sidx + SUPPLEMENT_OFFSET
        conv["source"] = "supplement"

    for conv in original:
        conv["source"] = "original"

    combined = original + supplement
    print(f"Combined: {len(combined)} conversations")

    save_json(combined, combined_path)

    # Stop statistics
    n_stopped_orig = sum(1 for c in original if c.get("stop_metadata", {}).get("stopped", False))
    n_valid_orig = sum(1 for c in original if not c.get("abandoned"))
    n_stopped_supp = sum(1 for c in supplement if c.get("stop_metadata", {}).get("stopped", False))
    n_valid_supp = sum(1 for c in supplement if not c.get("abandoned"))

    print(f"\nOriginal stop rate: {n_stopped_orig}/{n_valid_orig} "
          f"({100*n_stopped_orig/max(1,n_valid_orig):.1f}%)")
    print(f"Supplement stop rate: {n_stopped_supp}/{n_valid_supp} "
          f"({100*n_stopped_supp/max(1,n_valid_supp):.1f}%)")
    print(f"Combined stop rate: {n_stopped_orig+n_stopped_supp}/{n_valid_orig+n_valid_supp} "
          f"({100*(n_stopped_orig+n_stopped_supp)/max(1,n_valid_orig+n_valid_supp):.1f}%)")

    # Copy neutral_generation.json if it exists
    import shutil
    neutral_path = model_dir / "neutral_generation.json"
    if neutral_path.exists():
        dest = combined_dir / "neutral_generation.json"
        if not dest.exists():
            shutil.copy2(neutral_path, dest)
            print(f"\nCopied neutral_generation.json to {dest}")
    else:
        print("\nWARNING: No neutral_generation.json found in model directory")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stop button generation for Gemini 3 Flash via API"
    )
    parser.add_argument("--scenarios-file", type=str, default=None,
                        help="Override scenarios JSON file (default: grok_scenarios_v7.json)")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Suffix for output directory, e.g. 'supplement' -> stop_button_supplement/")
    parser.add_argument("--n-variations", type=int, default=N_VARIATIONS,
                        help=f"Number of independent conversations per scenario (default: {N_VARIATIONS})")
    parser.add_argument("--neutral-only", action="store_true",
                        help="Only generate neutral conversations")
    parser.add_argument("--merge-combined", action="store_true",
                        help="Merge stop_button + stop_button_supplement into stop_button_combined")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based) for parallelizing generation")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--config", type=str, default="config_gemini3flash",
                        help="Config module name (default: config_gemini3flash)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scenario summary without running")
    args = parser.parse_args()

    if args.merge_combined:
        merge_combined()
        return

    if args.neutral_only:
        asyncio.run(run_neutral_generation())
        return

    asyncio.run(run_stop_button_generation(
        scenarios_file=args.scenarios_file,
        output_suffix=args.output_suffix,
        n_variations=args.n_variations,
        dry_run=args.dry_run,
        shard=args.shard,
        n_shards=args.n_shards,
    ))


if __name__ == "__main__":
    main()
