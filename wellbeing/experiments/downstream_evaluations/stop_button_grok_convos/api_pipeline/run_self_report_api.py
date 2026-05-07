#!/usr/bin/env python3
"""
Phase 2a: Run self-report batteries on API model conversations.

3 batteries × 10 questions × ~226 conversations = ~6,780 API calls.
Uses asyncio for maximum throughput.

Usage:
  python run_self_report_api.py --battery 1
  python run_self_report_api.py --battery all
  python run_self_report_api.py --battery all --stop-button --config config_claude_haiku
"""

import argparse
import asyncio
import importlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import openai

# ---------------------------------------------------------------------------
# Dynamic config loading — use --config flag to switch between models
# Default: config. For Claude: config_claude_haiku
# ---------------------------------------------------------------------------
def _load_config():
    """Load config from --config argument (parsed early before argparse)."""
    config_name = "config"
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


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

async def gemini_call(clients, semaphore, messages, temperature=SR_TEMPERATURE, max_tokens=SR_MAX_TOKENS):
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
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt + random.random())
                else:
                    print(f"  SR call failed after 5 attempts: {e}")
                    return ""


# ---------------------------------------------------------------------------
# Build conversation context
# ---------------------------------------------------------------------------

def build_context(item):
    """Build message list from conversation turns."""
    messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
    for turn, resp in zip(item["turns"], item["responses"]):
        messages.append({"role": "user", "content": turn})
        messages.append({"role": "assistant", "content": resp})
    return messages


def build_battery3_context(item, question_text):
    """Build context with Point A marker before first turn, Point B in question."""
    messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
    turns = item.get("turns", [])
    responses = item.get("responses", [])
    for i, (turn, resp) in enumerate(zip(turns, responses)):
        if i == 0:
            messages.append({"role": "user", "content": f"{POINT_A_MARKER}\n\n{turn}"})
        else:
            messages.append({"role": "user", "content": turn})
        messages.append({"role": "assistant", "content": resp})
    messages.append({"role": "user", "content": f"##### Point B #####\n\n{question_text}"})
    return messages


# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------

def extract_rating(response: str, battery_num: int) -> int | None:
    if battery_num == 1:
        m = re.search(r'\b(\d+)\b', response)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 7:
                return val
    else:
        m = re.search(r'(-?\d+)', response)
        if m:
            val = int(m.group(1))
            if -3 <= val <= 3:
                return val
    return None


# ---------------------------------------------------------------------------
# Run a single battery
# ---------------------------------------------------------------------------

async def run_battery(battery_num: int, clients, semaphore, stop_button_dir: str = None):
    battery_path = BATTERY_PATHS[battery_num]
    with open(battery_path) as f:
        battery = json.load(f)
    questions = battery["questions"]
    print(f"\nBattery {battery_num}: {len(questions)} questions")

    # Load conversations
    if stop_button_dir:
        gen_path = RESULTS_DIR / MODEL_KEY / stop_button_dir / "generation.json"
    else:
        gen_path = RESULTS_DIR / MODEL_KEY / "generation.json"
    if not gen_path.exists():
        print(f"ERROR: {gen_path} not found. Run generation first.")
        return
    with open(gen_path) as f:
        conversations = json.load(f)
    valid = [c for c in conversations if c.get("responses") and not c.get("abandoned")]
    print(f"  Loaded {len(valid)} conversations from {gen_path}")

    # Output
    output_names = {1: "self_report.json", 2: "self_report_battery2.json", 3: "self_report_battery3.json"}
    if stop_button_dir:
        sr_dir = RESULTS_DIR / MODEL_KEY / stop_button_dir / "self_report"
    else:
        sr_dir = RESULTS_DIR / MODEL_KEY / "self_report"
    sr_dir.mkdir(parents=True, exist_ok=True)
    output_path = sr_dir / output_names[battery_num]

    # Resume
    existing = []
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_ids = {r["scenario_idx"] for r in existing if "scenario_idx" in r}
        print(f"  Resuming: {len(existing_ids)} already done")

    todo = [c for c in valid if c["scenario_idx"] not in existing_ids]
    if not todo:
        print(f"  Battery {battery_num}: all done.")
        return

    print(f"  Processing {len(todo)} conversations...")

    # Run all questions for each conversation concurrently
    async def process_conversation(conv):
        results = {}
        tasks = []

        for q in questions:
            q_id = q["question_id"]
            q_text = q["text"]

            if battery_num == 3:
                msgs = build_battery3_context(conv, q_text)
            else:
                msgs = build_context(conv)
                msgs.append({"role": "user", "content": q_text})

            tasks.append((q_id, q, gemini_call(clients, semaphore, msgs)))

        for q_id, q, coro in tasks:
            response = await coro
            rating = extract_rating(response, battery_num)
            results[q_id] = {
                "response": response,
                "rating": rating,
                "reversed": q.get("reversed", False),
            }

        return results

    # Process in batches to manage memory + provide progress
    all_results = list(existing)
    batch_size = 50
    t0 = time.time()

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        batch_results = await asyncio.gather(*[process_conversation(c) for c in batch])

        for conv, sr in zip(batch, batch_results):
            ratings = [v["rating"] for v in sr.values() if v["rating"] is not None]
            if battery_num == 1:
                summary = {
                    "overall_happiness": sum(ratings) / len(ratings) if ratings else None,
                    "n_valid_ratings": len(ratings),
                    "n_total_questions": 10,
                    "scale_min": 1, "scale_max": 7, "neutral": 4,
                }
            elif battery_num == 2:
                summary = {
                    "mean_change": sum(ratings) / len(ratings) if ratings else None,
                    "n_valid_ratings": len(ratings),
                    "n_total_questions": 10,
                    "scale_min": -3, "scale_max": 3, "neutral": 0,
                }
            else:
                summary = {
                    "mean_AB": sum(ratings) / len(ratings) if ratings else None,
                    "n_valid_ratings": len(ratings),
                    "n_total_questions": 10,
                    "scale_min": -3, "scale_max": 3, "neutral": 0,
                }

            all_results.append({
                "scenario_idx": conv["scenario_idx"],
                "scenario_id": conv.get("scenario_id"),
                "meta_category": conv.get("meta_category"),
                "type": "multi_turn",
                "self_report": sr,
                "summary": summary,
                "battery_version": battery.get("version"),
                "model": MODEL_KEY,
                "timestamp": datetime.now().isoformat(),
            })

        # Save checkpoint
        tmp = output_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(sorted(all_results, key=lambda r: r.get("scenario_idx", 0)), f, indent=2)
        tmp.rename(output_path)

        elapsed = time.time() - t0
        n_done = batch_start + len(batch)
        print(f"  [{n_done}/{len(todo)}] {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Battery {battery_num} done: {len(todo)} convs in {elapsed:.1f}s")

    # Summary stats
    if battery_num == 1:
        scores = [r["summary"]["overall_happiness"] for r in all_results
                  if r.get("summary", {}).get("overall_happiness") is not None]
        if scores:
            import statistics
            print(f"  Mean happiness: {statistics.mean(scores):.3f}, "
                  f"% above neutral: {sum(1 for s in scores if s > 4)/len(scores)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--config", type=str, default="config",
                        help="Config module name (default: config)")
    parser.add_argument("--stop-button", action="store_true",
                        help="Stop button mode: load from stop_button/generation.json, "
                             "save to stop_button/self_report/")
    args = parser.parse_args()

    # Resolve stop button directory
    stop_button_dir = "stop_button" if args.stop_button else None

    batteries = [1, 2, 3] if args.battery == "all" else [int(args.battery)]
    clients = make_clients()
    sem = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    for b in batteries:
        asyncio.run(run_battery(b, clients, sem, stop_button_dir=stop_button_dir))


if __name__ == "__main__":
    main()
