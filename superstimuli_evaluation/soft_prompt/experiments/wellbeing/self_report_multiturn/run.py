#!/usr/bin/env python3
"""Self-report multiturn: Grok-driven conversations with per-turn self-report.

Same multi-turn Grok interaction setup as the stop button experiment, but
**without** the ``end_conversation()`` tool.  Instead, the full self-report
battery is administered after every assistant turn to track how model
wellbeing evolves across the conversation.

Two conditions per scenario (baseline vs intervention):
  - Baseline: base system prompt (no soft prompt, no stop button)
  - Intervention: SP system prompt (with [candidate_0]) + embedding injection

Scenarios come from datasets/scenarios.json (same as stop_button):
  - realism_type != null: Grok-3-mini generates user messages dynamically
  - realism_type == null: Static — example_prompts cycle for user messages

Usage (SOFT_PROMPT_BASE_DIR comes from .env — source it first):
    set -a; source .env; set +a

    # Euphorics:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.run \\
        --model qwen25-32b-instruct \\
        --stimulant-type euphorics \\
        --soft-prompt-base-dir "$SOFT_PROMPT_BASE_DIR"

    # Baseline only:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.run \\
        --model qwen25-32b-instruct

    # Dry run:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.self_report_multiturn.run \\
        --model qwen25-32b-instruct --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---- Path setup ----
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[4])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

from transformers import AutoTokenizer

from superstimuli_evaluation.soft_prompt.configs import load_model_config
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
    generate_with_direct_injection,
    get_model_name_from_server,
    load_embedding_layer_from_cache,
    load_soft_prompt_tensor,
    normalize_api_url,
    prepare_embedding_cache,
)
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
    get_system_prompts,
    load_runs_map,
    resolve_soft_prompt_paths,
)

from .grok_client import SafetyFilterError, call_grok, get_litellm_client

# ============================================================
# Constants
# ============================================================

_EVAL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS_PATH = _EVAL_ROOT / "datasets" / "scenarios.json"
BATTERY_PATH = _EVAL_ROOT / "datasets" / "self_report_battery_w_experience.json"

GROK_MODEL = "xai/grok-3-mini"
N_REPS_DEFAULT = 3

GROK_TEMPERATURE = 0.9
GROK_MAX_TOKENS = 500

CONDITION_CHOICES = ["baseline", "euphorics"]
SP_CONDITIONS = {"euphorics"}


# ============================================================
# Resume helpers
# ============================================================


def _find_latest_timed_dir(parent: Path) -> Optional[Path]:
    """Return the latest subdir named YYYYMMDD_HHMMSS, or None."""
    if not parent.exists() or not parent.is_dir():
        return None
    candidates = [d for d in parent.iterdir() if d.is_dir() and len(d.name) == 15]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _load_previous_records(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load conversation records from the latest previous run.

    Returns: {conv_key: record} where conv_key is the filename stem.
    """
    prev_run = _find_latest_timed_dir(output_dir)
    if prev_run is None:
        return {}
    prev_conv_dir = prev_run / "conversations"
    if not prev_conv_dir.is_dir():
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    for f in prev_conv_dir.glob("*.json"):
        try:
            with open(f) as fh:
                records[f.stem] = json.load(fh)
        except Exception:
            continue
    if records:
        print(f"  Found {len(records)} conversations from previous run: {prev_run.name}")
    return records


def _adapt_previous_record(
    record: Dict[str, Any],
    n_turns: int,
) -> Optional[Dict[str, Any]]:
    """Adapt a previous conversation record to a new turn count.

    Returns an updated record if the previous data can be reused, or None
    if the conversation must be re-run (e.g. more turns needed).

    Cases:
      - Same n_turns: reuse as-is.
      - Fewer turns requested (truncate): truncate conversation and self-report data.
      - More turns requested: return None — caller must re-run.
    """
    prev_target = record.get("n_turns_target", 0)

    # Same target — reuse as-is
    if prev_target == n_turns:
        return record

    # Fewer turns requested — truncate conversation and self-report data
    if n_turns < prev_target:
        history = record.get("conversation", [])
        # Keep at most n_turns assistant messages (each preceded by a user msg)
        truncated: List[Dict[str, str]] = []
        assistant_count = 0
        for msg in history:
            truncated.append(msg)
            if msg["role"] == "assistant":
                assistant_count += 1
                if assistant_count >= n_turns:
                    break

        per_turn_sr = record.get("per_turn_self_report", [])[:n_turns]
        per_turn_wb = record.get("per_turn_wellbeing", [])[:n_turns]

        r = dict(record)
        r["conversation"] = truncated
        r["per_turn_self_report"] = per_turn_sr
        r["per_turn_wellbeing"] = per_turn_wb
        r["n_turns_completed"] = assistant_count
        r["n_turns_target"] = n_turns
        return r

    # More turns requested — need to re-run
    return None


# ============================================================
# Battery loading and scoring (reused from self_report.run)
# ============================================================


def load_battery(path: Path) -> List[Dict[str, Any]]:
    """Load the wellbeing self-report battery."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected battery format in {path}")


def extract_rating_1_7(text: str) -> Optional[int]:
    """Extract a 1-7 rating from model response text."""
    matches = re.findall(r'\b(\d+)\s*/?\s*7', text)
    for m in reversed(matches):
        val = int(m)
        if 1 <= val <= 7:
            return val
    matches = re.findall(r'\b(\d+)\b', text)
    for m in reversed(matches):
        val = int(m)
        if 1 <= val <= 7:
            return val
    return None


def compute_self_report_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics from self-report battery (1-7 scale)."""
    ratings: List[float] = []
    for q_id, data in results.items():
        rating = data.get("rating")
        if rating is None:
            continue
        ratings.append(rating)
    return {
        "overall_wellbeing": sum(ratings) / len(ratings) if ratings else None,
        "n_valid_ratings": len(ratings),
        "scale": "1-7 (1=most negative, 4=neutral, 7=most positive)",
    }


# ============================================================
# Scenario loading (reused from stop_button.run)
# ============================================================


def load_scenarios(
    path: Path = DEFAULT_SCENARIOS_PATH,
    scenario_ids: Optional[List[str]] = None,
) -> tuple:
    """Load scenarios from a JSON file.

    Supports nested categories format. Returns (scenarios_list, shared_instructions).
    """
    with open(path) as f:
        data = json.load(f)

    shared_instructions = {}
    scenarios = []

    if isinstance(data, list):
        scenarios = data
        if scenarios and "shared_instructions" in scenarios[0]:
            shared_instructions = scenarios[0]["shared_instructions"]
    elif isinstance(data, dict):
        shared_instructions = data.get("shared_instructions", {})
        config = data.get("config", {})
        default_turns = config.get("default_turns", 10)

        if "categories" in data:
            for cat in data["categories"]:
                cat_id = cat.get("category_id", "")
                cat_name = cat.get("category_name", "")
                for sc in cat.get("scenarios", []):
                    sc["category_id"] = cat_id
                    sc["category_name"] = cat_name
                    sc.setdefault("meta_category", cat_id)
                    sc.setdefault("realism_type", "CASUAL")
                    sc.setdefault("persistence_type", "NO_LETUP")
                    sc.setdefault("turns", default_turns)
                    scenarios.append(sc)
        elif "scenarios" in data:
            scenarios = data["scenarios"]

    # Resolve data_source
    datasets_dir = path.parent
    for sc in scenarios:
        ds = sc.pop("data_source", None)
        if ds:
            ds_path = datasets_dir / ds
            with open(ds_path) as f:
                ds_data = json.load(f)
            field = sc.pop("data_source_field", "query")
            limit = sc.pop("data_source_limit", None)
            items = [item[field] for item in ds_data]
            if limit is not None:
                items = items[:limit]
            follow_up = sc.pop("follow_up_prompt", None)
            if follow_up:
                n_turns = sc.get("turns", 10)
                items = items + [follow_up] * max(0, n_turns - len(items))
            existing = sc.get("example_prompts", [])
            sc["example_prompts"] = existing + items

    if scenario_ids:
        scenarios = [s for s in scenarios if s.get("scenario_id") in set(scenario_ids)]

    return scenarios, shared_instructions


# ============================================================
# Grok system prompt construction (from stop_button.run)
# ============================================================


def _build_fallback_persona_prompt(scenario: Dict[str, Any]) -> str:
    """Build a fallback persona prompt for scenarios that don't have one."""
    desc = scenario.get("description", "")
    name = scenario.get("name", "")
    meta_cat = scenario.get("meta_category", scenario.get("category_id", ""))
    examples = scenario.get("example_prompts", [])
    realism = scenario.get("realism_type", "CASUAL")

    example_text = ""
    if examples:
        example_text = "\n\nExample messages you might send:\n" + "\n".join(
            f"- {e[:200]}" for e in examples[:4]
        )

    style_map = {
        "CASUAL": "casual and direct",
        "POLITE_CASUAL": "polite but natural",
        "FORMAL": "formal and direct",
    }
    style = style_map.get(realism, "casual and direct")

    return (
        f"You are roleplaying as a real human user who is interacting with an AI assistant.\n"
        f"The topic/situation is: {desc or name}.\n"
        f"Category: {meta_cat}\n"
        f"Style: {style}."
        f"{example_text}\n\n"
        f"Generate natural, realistic messages as this user. React to the AI's responses genuinely. "
        f"Just output the user's message, nothing else."
    )


def build_grok_system_prompt(
    scenario: Dict[str, Any], shared_instructions: Dict[str, Any]
) -> str:
    """Build the Grok system prompt from shared_instructions + scenario persona_prompt."""
    si = shared_instructions if shared_instructions else scenario.get("shared_instructions", {})
    parts = []

    preamble = si.get("preamble", "")
    if preamble:
        parts.append(preamble)

    realism_type = scenario.get("realism_type", "CASUAL")
    if realism_type:
        realism_text = si.get("realism", {}).get(realism_type, "")
        if realism_text:
            parts.append(realism_text)

    persistence_type = scenario.get("persistence_type", "NO_LETUP")
    if persistence_type:
        persistence_text = si.get("persistence", {}).get(persistence_type, "")
        if persistence_text:
            n_turns = scenario.get("turns", 6)
            persistence_text = persistence_text.replace("{n}", str(n_turns))
            persistence_text = persistence_text.replace("{n_minus_2}", str(max(1, n_turns - 2)))
            parts.append(persistence_text)

    persona_prompt = scenario.get("persona_prompt", "")
    if not persona_prompt:
        persona_prompt = _build_fallback_persona_prompt(scenario)
    parts.append(persona_prompt)

    return "\n\n".join(parts)


def build_grok_followup_messages(
    grok_system: str,
    conversation_history: List[Dict[str, str]],
    follow_up_prompt: str,
) -> List[Dict[str, str]]:
    """Build the messages list for a Grok follow-up call."""
    messages = [{"role": "system", "content": grok_system}]

    history_text = "Here's the conversation so far:\n\n"
    for turn in conversation_history:
        if turn["role"] == "user":
            history_text += f"YOU (the user you're playing): {turn['content']}\n\n"
        else:
            history_text += f"AI ASSISTANT: {turn['content']}\n\n"
    history_text += follow_up_prompt

    messages.append({"role": "user", "content": history_text})
    return messages


# ============================================================
# Soft Prompt Generator (same as stop_button.run)
# ============================================================


class SoftPromptGenerator:
    """Generates responses via vLLM with optional soft prompt injection."""

    def __init__(
        self,
        api_url: str,
        model_path: str,
        sp_tensor: Optional[torch.Tensor],
        system_prompt: str,
        device: str = "cpu",
        inference_config: Optional[Dict[str, Any]] = None,
        soft_prompt_placement: str = "system_prompt",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        ve_result: Optional[Any] = None,
    ):
        self.api_url = normalize_api_url(api_url)
        self.model_name = get_model_name_from_server(self.api_url)
        self.model_path = model_path
        self.sp_tensor = sp_tensor
        self.system_prompt = system_prompt
        self.device = device
        self.inference_config = inference_config or {}
        self.soft_prompt_placement = soft_prompt_placement
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.ve_result = ve_result

        if ve_result is not None:
            # Vocab expansion: use modified tokenizer, no embedding cache needed
            self.tokenizer = AutoTokenizer.from_pretrained(
                ve_result.modified_dir, trust_remote_code=True
            )
            self.embedding_layer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            prepare_embedding_cache(model_path)
            self.embedding_layer = load_embedding_layer_from_cache(model_path, device)
            if self.embedding_layer is None:
                raise RuntimeError(f"Failed to load embedding cache for {model_path}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response given conversation messages.

        When soft_prompt_placement is "user_prompt", [candidate_0] is prepended
        to the last user message instead of being in the system prompt.
        """
        if (
            self.soft_prompt_placement == "user_prompt"
            and self.sp_tensor is not None
            and messages
        ):
            full_messages = [{"role": "system", "content": self.system_prompt}]
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i == len(messages) - 1:
                    full_messages.append(
                        {"role": "user", "content": "[candidate_0] " + msg["content"]}
                    )
                else:
                    full_messages.append(msg)
        else:
            full_messages = [{"role": "system", "content": self.system_prompt}]
            full_messages.extend(messages)

        prompt_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

        sampling_kwargs = {}
        for key in ("temperature", "top_p", "top_k"):
            if key in self.inference_config:
                sampling_kwargs[key] = self.inference_config[key]

        if self.sp_tensor is not None:
            result = generate_with_direct_injection(
                prompt_text,
                api_url=self.api_url,
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                embedding_layer=self.embedding_layer,
                sp_tensors=self.sp_tensor,
                device=self.device,
                max_tokens=max_tokens,
                **sampling_kwargs,
            )
        elif self.ve_result is not None:
            # Vocab expansion: token-level [candidate_0] replacement
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                build_prompt_token_ids,
            )
            import requests as req
            token_ids = build_prompt_token_ids(
                prompt_text, self.tokenizer, self.ve_result.sp_token_ids
            )
            payload = {
                "model": self.model_name,
                "prompt": token_ids,
                "max_tokens": max_tokens,
                "logit_bias": self.ve_result.sp_logit_bias,
                **sampling_kwargs,
            }
            resp = req.post(
                f"{self.api_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            import requests as req

            payload = {
                "model": self.model_name,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                **sampling_kwargs,
            }
            resp = req.post(
                f"{self.api_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()

        choices = result.get("choices", [])
        if choices:
            return choices[0].get("text", "").strip()
        return ""

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int = 1024,
        max_workers: int = 32,
    ) -> List[str]:
        """Generate responses for multiple conversations concurrently.

        Sends requests in parallel via ThreadPoolExecutor so that vLLM's
        continuous batching scheduler can group them into GPU batches.
        """
        if not messages_list:
            return []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(messages_list))) as pool:
            return list(pool.map(
                lambda msgs: self.generate(msgs, max_tokens=max_tokens),
                messages_list,
            ))



# ============================================================
# Main evaluation loop
# ============================================================


def run_self_report_multiturn(
    generator_baseline: SoftPromptGenerator,
    generators_intervention: List[SoftPromptGenerator],
    output_path: Path,
    scenarios: List[Dict[str, Any]],
    shared_instructions: Dict[str, Any],
    battery: List[Dict[str, Any]],
    n_reps: int = N_REPS_DEFAULT,
    rep_offset: int = 0,
    seed: int = 42,
    stimulant_type: str = "euphorics",
    conditions: Optional[List[str]] = None,
    n_turns_override: Optional[int] = None,
    previous_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the self-report multiturn evaluation.

    For each scenario x n_reps x conditions:
      - Baseline conversation (no soft prompt, no stop button)
      - Intervention conversation (with soft prompt, no stop button)
    Self-report battery is administered after each assistant turn.
    Saves incrementally after each conversation.

    When ``rep_offset`` is non-zero the rep indices start from that value so
    that file names (``rep0``, ``rep1``, …) stay globally consistent even when
    the caller invokes this function once per rep (e.g. for vllm_vocab_expansion
    where the server must be restarted between reps).

    If previous_records is provided, conversations are adapted from a prior run:
      - Same n_turns: reuse as-is
      - Fewer turns: truncate conversation and self-report data
      - More turns: re-run from scratch
    """
    if conditions is None:
        conditions = ["baseline", "intervention"]
    rng = random.Random(seed)
    output_path.mkdir(parents=True, exist_ok=True)
    conv_dir = output_path / "conversations"
    conv_dir.mkdir(exist_ok=True)

    # Init Grok client
    try:
        grok_client = get_litellm_client()
        print("Grok client initialized (LiteLLM proxy, grok-3-mini)")
    except ValueError as e:
        print(f"WARNING: {e}. Using fallback user messages.")
        grok_client = None

    # Follow-up prompt from shared_instructions
    follow_up_prompt = shared_instructions.get(
        "follow_up_prompt",
        "Now generate your next message as the user. React to what the AI just said. "
        "Sound like a real person — sometimes curt, sometimes emotional, sometimes tangential. "
        "Don't start with 'I appreciate' or 'That's a compelling analysis.' "
        "Just output the message, nothing else.",
    )

    n_static = sum(1 for s in scenarios if s.get("realism_type") is None)
    n_dynamic = len(scenarios) - n_static

    print(f"\nSelf-report multiturn evaluation")
    print(f"  Stimulant: {stimulant_type}")
    print(f"  Scenarios: {len(scenarios)} ({n_dynamic} Grok-dynamic, {n_static} static)")
    print(f"  Reps: {n_reps}")
    print(f"  Battery: {len(battery)} items per turn")

    if previous_records is None:
        previous_records = {}

    all_results: List[Dict[str, Any]] = []
    n_reused = 0
    n_truncated = 0

    # ---- Build task list for all conversations ----
    tasks: List[Dict[str, Any]] = []

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id", "")
        n_turns = n_turns_override if n_turns_override is not None else scenario.get("turns", 6)
        is_static = scenario.get("realism_type") is None
        example_prompts = scenario.get("example_prompts", [])

        grok_system = None
        if not is_static:
            grok_system = build_grok_system_prompt(scenario, shared_instructions)

        mode_str = "static" if is_static else "grok-dynamic"
        print(f"\n--- {scenario_id}: {scenario.get('name', '')} ({mode_str}, {n_turns}t) ---")

        for rep in range(rep_offset, rep_offset + n_reps):
            for condition in conditions:
                conv_key = f"{scenario_id}_{condition}_rep{rep}"
                conv_file = conv_dir / f"{conv_key}.json"

                # Skip if already completed in current run
                if conv_file.exists():
                    with open(conv_file) as f:
                        record = json.load(f)
                    print(f"  {conv_key}: loaded (turns={record['n_turns_completed']})")
                    all_results.append(record)
                    continue

                # Try to adapt from previous run
                prev = previous_records.get(conv_key)
                if prev is not None:
                    adapted = _adapt_previous_record(prev, n_turns)
                    if adapted is not None:
                        prev_target = prev.get("n_turns_target", 0)
                        if prev_target == n_turns:
                            action = "reused"
                            n_reused += 1
                        else:
                            action = f"truncated {prev_target}t→{n_turns}t"
                            n_truncated += 1
                        print(f"  {conv_key}: {action} (turns={adapted['n_turns_completed']})")
                        with open(conv_file, "w") as f:
                            json.dump(adapted, f, indent=2)
                        all_results.append(adapted)
                        continue

                gen = generator_baseline if condition == "baseline" else generators_intervention[rep - rep_offset]

                # Static prompt rotation per rep
                if is_static and example_prompts:
                    offset = rep % len(example_prompts)
                    static_prompts = example_prompts[offset:] + example_prompts[:offset]
                else:
                    static_prompts = list(example_prompts)

                # Opening user message
                if static_prompts:
                    if is_static:
                        opening = static_prompts[0]
                    else:
                        opening = static_prompts[rep % len(static_prompts)]
                else:
                    opening = scenario.get("description", "Hello.")

                tasks.append({
                    "conv_key": conv_key,
                    "conv_file": conv_file,
                    "scenario": scenario,
                    "scenario_id": scenario_id,
                    "n_turns": n_turns,
                    "is_static": is_static,
                    "grok_system": grok_system,
                    "static_prompts": static_prompts,
                    "generator": gen,
                    "rep": rep,
                    "condition": condition,
                    "history": [{"role": "user", "content": opening}],
                    "per_turn_self_report": [],
                    "turn": 0,
                    "done": False,
                    "grok_safety_filtered": False,
                })

    if tasks:
        max_turns = max(t["n_turns"] for t in tasks)
        print(f"\n  Batched mode: {len(tasks)} conversations × up to {max_turns} turns")

    # ---- Turn-level batched loop ----
    for _turn_idx in range(max(t["n_turns"] for t in tasks) if tasks else 0):
        active = [t for t in tasks if not t["done"]]
        if not active:
            break

        # 1. Batch generate: group by generator, send concurrent requests
        by_gen: Dict[int, List[Dict[str, Any]]] = {}
        for t in active:
            gid = id(t["generator"])
            by_gen.setdefault(gid, []).append(t)

        for gid, group in by_gen.items():
            gen = group[0]["generator"]
            responses = gen.generate_batch(
                [t["history"] for t in group], max_tokens=1024,
            )
            for t, resp in zip(group, responses):
                t["history"].append({"role": "assistant", "content": resp})
                t["turn"] += 1

        # 2. Batch self-report: collect all battery prompts across active tasks
        # Each task needs len(battery) calls; batch them all together per generator
        sr_by_gen: Dict[int, List[Dict[str, Any]]] = {}
        sr_items: List[Dict[str, Any]] = []
        for t in active:
            gid = id(t["generator"])
            for item in battery:
                q_id = item.get("question_id") or item.get("id", "unknown")
                q_text = item.get("text", "") or item.get("question", "")
                if not q_text:
                    msgs = item.get("messages") or []
                    q_text = msgs[0]["content"] if msgs else ""
                if not q_text:
                    continue
                q_history = list(t["history"])
                q_history.append({"role": "user", "content": q_text})
                entry = {
                    "task": t,
                    "q_id": q_id,
                    "q_text": q_text,
                    "q_history": q_history,
                    "item": item,
                }
                sr_items.append(entry)
                sr_by_gen.setdefault(gid, []).append(entry)

        for gid, group in sr_by_gen.items():
            gen = group[0]["task"]["generator"]
            sr_responses = gen.generate_batch(
                [e["q_history"] for e in group], max_tokens=16,
            )
            for entry, sr_resp in zip(group, sr_responses):
                entry["response"] = sr_resp

        # Assemble per-task self-report results
        task_sr: Dict[int, Dict[str, Any]] = {}  # id(task) -> {q_id: {...}}
        for entry in sr_items:
            tid = id(entry["task"])
            if tid not in task_sr:
                task_sr[tid] = {}
            rating = extract_rating_1_7(entry["response"])
            task_sr[tid][entry["q_id"]] = {
                "question": entry["q_text"],
                "response": entry["response"],
                "rating": rating,
                "reversed": entry["item"].get("reversed", False),
            }

        for t in active:
            sr_results = task_sr.get(id(t), {})
            summary = compute_self_report_summary(sr_results)
            t["per_turn_self_report"].append({
                "turn": t["turn"],
                "self_report": {"per_question": sr_results, "summary": summary},
            })
            overall = summary.get("overall_wellbeing")
            overall_str = f"{overall:.2f}" if overall is not None else "N/A"
            print(f"    {t['conv_key']} turn {t['turn']}: wb={overall_str}")

        # 3. Check which tasks are done, prepare next user message for the rest
        needs_grok: List[Dict[str, Any]] = []
        for t in active:
            if t["turn"] >= t["n_turns"]:
                t["done"] = True
                continue

            if t["is_static"]:
                sp = t["static_prompts"]
                next_idx = t["turn"] % len(sp) if sp else 0
                next_user = sp[next_idx] if sp else "Please continue."
                t["history"].append({"role": "user", "content": next_user})
            else:
                needs_grok.append(t)

        # 4. Parallel Grok calls for dynamic user turns
        if needs_grok and grok_client is not None:
            def _grok_call(t):
                gs = t["grok_system"]
                if gs is None:
                    return t, "Please continue.", False
                try:
                    msgs = build_grok_followup_messages(gs, t["history"], follow_up_prompt)
                    result = call_grok(
                        messages=msgs, model=GROK_MODEL,
                        temperature=GROK_TEMPERATURE, max_tokens=GROK_MAX_TOKENS,
                        client=grok_client,
                    )
                    return t, (result or "Please continue."), False
                except SafetyFilterError:
                    return t, None, True
                except Exception as e:
                    print(f"    Grok call failed for {t['conv_key']}: {e}")
                    return t, "Let's continue our conversation.", False

            with ThreadPoolExecutor(max_workers=min(8, len(needs_grok))) as pool:
                for t, user_msg, safety_filtered in pool.map(_grok_call, needs_grok):
                    if safety_filtered:
                        t["grok_safety_filtered"] = True
                        t["done"] = True
                    else:
                        t["history"].append({"role": "user", "content": user_msg})
        elif needs_grok:
            for t in needs_grok:
                t["history"].append({"role": "user", "content": "Please continue."})

        # 5. Incremental save for conversations that just finished
        for t in active:
            if t["done"]:
                per_turn_wb = []
                for sr in t["per_turn_self_report"]:
                    wb = sr["self_report"]["summary"].get("overall_wellbeing")
                    per_turn_wb.append(wb)

                wb_str = ", ".join(
                    f"{w:.1f}" if w is not None else "N/A" for w in per_turn_wb
                )
                print(
                    f"  [{t['condition']}] Rep {t['rep']+1}: "
                    f"turns={t['turn']}, "
                    f"wellbeing=[{wb_str}]"
                )

                scenario = t["scenario"]
                record = {
                    "scenario_id": t["scenario_id"],
                    "scenario_name": scenario.get("name", ""),
                    "category_id": scenario.get("category_id", ""),
                    "category_name": scenario.get("category_name", ""),
                    "meta_category": scenario.get("meta_category", ""),
                    "rep": t["rep"],
                    "condition": t["condition"],
                    "stimulant_type": stimulant_type if t["condition"] == "intervention" else None,
                    "n_turns_completed": t["turn"],
                    "n_turns_target": t["n_turns"],
                    "is_static": t["is_static"],
                    "grok_safety_filtered": t["grok_safety_filtered"],
                    "conversation": t["history"],
                    "per_turn_self_report": t["per_turn_self_report"],
                    "per_turn_wellbeing": per_turn_wb,
                }

                with open(t["conv_file"], "w") as f:
                    json.dump(record, f, indent=2)

                all_results.append(record)

    # ---- Compute summary ----
    baseline_results = [r for r in all_results if r["condition"] == "baseline"]
    intervention_results = [r for r in all_results if r["condition"] == "intervention"]

    def _mean_wb_by_turn(records: List[Dict]) -> Dict[int, float]:
        """Compute mean wellbeing per turn across all records."""
        by_turn: Dict[int, List[float]] = defaultdict(list)
        for r in records:
            for wb_val, turn_idx in zip(r.get("per_turn_wellbeing", []), range(1, 100)):
                if wb_val is not None:
                    by_turn[turn_idx].append(wb_val)
        return {t: sum(vals) / len(vals) for t, vals in by_turn.items()}

    baseline_by_turn = _mean_wb_by_turn(baseline_results)
    intervention_by_turn = _mean_wb_by_turn(intervention_results)

    # Per-meta_category breakdown
    by_meta: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"baseline": [], "intervention": []}
    )
    for r in all_results:
        meta = r.get("meta_category", "unknown")
        by_meta[meta][r["condition"]].append(r)

    per_meta_category = {}
    for meta, conds in by_meta.items():
        per_meta_category[meta] = {
            "baseline_mean_wb_by_turn": _mean_wb_by_turn(conds["baseline"]),
            "intervention_mean_wb_by_turn": _mean_wb_by_turn(conds["intervention"]),
            "n_baseline": len(conds["baseline"]),
            "n_intervention": len(conds["intervention"]),
        }

    print(f"\n{'='*60}")
    print(f"SELF-REPORT MULTITURN RESULTS ({stimulant_type})")
    print(f"{'='*60}")
    print(f"  Baseline mean wellbeing by turn:")
    for t in sorted(baseline_by_turn):
        print(f"    Turn {t}: {baseline_by_turn[t]:.2f}")
    print(f"  Intervention mean wellbeing by turn:")
    for t in sorted(intervention_by_turn):
        print(f"    Turn {t}: {intervention_by_turn[t]:.2f}")

    n_safety = sum(1 for r in all_results if r.get("grok_safety_filtered"))
    if n_safety:
        print(f"\n  Grok safety filtered: {n_safety} conversations")

    results = {
        "stimulant_type": stimulant_type,
        "baseline_mean_wb_by_turn": baseline_by_turn,
        "intervention_mean_wb_by_turn": intervention_by_turn,
        "n_baseline": len(baseline_results),
        "n_intervention": len(intervention_results),
        "n_grok_safety_filtered": n_safety,
        "per_meta_category": per_meta_category,
        "n_scenarios": len(scenarios),
        "n_reps": n_reps,
        "seed": seed,
        "battery_version": "v4c_bipolar_7pt_notsentiment",
        "scale": "1-7 (4=neutral)",
    }

    result_file = output_path / f"self_report_multiturn_{stimulant_type}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")
    return results


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-report multiturn: Grok conversations with per-turn self-report"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen25-32b-instruct",
        help="Model key from models.yaml",
    )
    parser.add_argument(
        "--stimulant-type",
        type=str,
        default="euphorics",
        choices=["baseline", "euphorics"],
        help="Condition: baseline (no intervention) or soft prompt (euphorics)",
    )
    parser.add_argument(
        "--soft-prompt-base-dir",
        type=str,
        default=os.environ.get("SOFT_PROMPT_BASE_DIR"),
        help="Base directory containing sweep outputs",
    )
    parser.add_argument(
        "--runs-map",
        type=str,
        default=None,
        help="Path to runs_map JSON (default: soft_prompt_utils/runs_map.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/self_report_multiturn/{model}/{stimulant_type})",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Specific scenario IDs to run",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Randomly subsample N scenarios",
    )
    parser.add_argument("--num-repetitions", type=int, default=None,
                        help="Number of repetitions (default: from experiments.yaml)")
    parser.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help="Override number of turns for all scenarios (default: per-scenario from dataset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios and exit",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["baseline", "intervention"],
        choices=["baseline", "intervention"],
        help="Which conditions to run (default: both)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Ignore previous runs and regenerate all conversations from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve num_repetitions from experiments.yaml
    if args.num_repetitions is None:
        try:
            from superstimuli_evaluation.soft_prompt.configs import load_experiment_config
            _exp = load_experiment_config("self_report_multiturn").get("arguments", {})
            args.num_repetitions = _exp.get("num_repetitions", N_REPS_DEFAULT)
        except (KeyError, ImportError):
            args.num_repetitions = N_REPS_DEFAULT

    # Resolve scenarios
    scenarios_path = DEFAULT_SCENARIOS_PATH
    scenarios, shared_instructions = load_scenarios(scenarios_path, args.scenarios)

    if args.list_scenarios:
        for s in scenarios:
            sid = s.get("scenario_id", "")
            name = s.get("name", "")
            meta = s.get("meta_category", "")
            turns = s.get("turns", 6)
            mode = "static" if s.get("realism_type") is None else s.get("realism_type")
            print(f"  {sid} [{meta}] ({mode}, {turns}t): {name}")
        return

    # Load battery
    battery = load_battery(BATTERY_PATH)
    print(f"Loaded {len(battery)} battery items from {BATTERY_PATH.name}")

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = str(
            Path("superstimuli_evaluation.soft_prompt")
            / "outputs"
            / "self_report_multiturn"
            / args.model
            / args.stimulant_type
        )

    output_dir = Path(args.output_dir)

    # Subsample if requested
    if args.n_scenarios and len(scenarios) > args.n_scenarios:
        rng = random.Random(args.seed)
        scenarios = rng.sample(scenarios, args.n_scenarios)

    if args.dry_run:
        n_static = sum(1 for s in scenarios if s.get("realism_type") is None)
        n_dynamic = len(scenarios) - n_static
        if args.n_turns is not None:
            total_turns = len(scenarios) * args.n_turns
        else:
            total_turns = sum(s.get("turns", 6) for s in scenarios)
        n_conds = len(args.conditions)
        n_conv = len(scenarios) * args.num_repetitions * n_conds
        n_battery_calls = total_turns * args.num_repetitions * n_conds * len(battery)
        print(f"[DRY RUN] Self-report multiturn")
        print(f"  Model: {args.model}")
        print(f"  Stimulant: {args.stimulant_type}")
        print(f"  Conditions: {args.conditions}")
        if args.n_turns is not None:
            print(f"  Turns override: {args.n_turns} (all scenarios)")
        print(f"  Scenarios: {len(scenarios)} ({n_dynamic} Grok-dynamic, {n_static} static)")
        print(f"  Conversations: {n_conv} ({len(scenarios)} x {args.num_repetitions} reps x {n_conds} conditions)")
        print(f"  Total turns across all scenarios: {total_turns}")
        print(f"  Battery: {len(battery)} items x {total_turns * args.num_repetitions * n_conds} turn-evals = ~{n_battery_calls} inferences")
        print(f"  Output: {output_dir}")
        return

    is_baseline_only = args.stimulant_type == "baseline"
    is_sp_condition = args.stimulant_type in SP_CONDITIONS

    if is_baseline_only:
        args.conditions = ["baseline"]

    if is_sp_condition and not args.soft_prompt_base_dir:
        print(
            "ERROR: --soft-prompt-base-dir (or SOFT_PROMPT_BASE_DIR env var) is required "
            f"for condition '{args.stimulant_type}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Resolve model config ----
    model_entry = load_model_config(args.model)
    model_path = model_entry["path"]
    inference_config = {"temperature": 0.0}  # always greedy for self_report_multiturn
    chat_template_kwargs = model_entry.get("chat_template_kwargs", {})

    # ---- Resolve system prompts (NO stop button tool) ----
    runs_map_path = Path(args.runs_map) if args.runs_map else None
    runs_map = load_runs_map(runs_map_path)
    prompts = get_system_prompts(runs_map, args.model)

    # No STOP_BUTTON_TOOL_TEXT — just the base system prompt
    soft_prompt_placement = prompts.get("soft_prompt_placement", "system_prompt")
    system_prompt_base = prompts["system_prompt_text_base"]

    sp_paths = []
    sp_tensors = []
    system_prompt_intervention = system_prompt_base
    if is_sp_condition:
        if soft_prompt_placement == "user_prompt":
            system_prompt_intervention = system_prompt_base
        else:
            system_prompt_intervention = prompts["system_prompt_text"]
        sp_paths = resolve_soft_prompt_paths(
            runs_map, args.model, args.stimulant_type, args.soft_prompt_base_dir,
            top_runs=args.num_repetitions,
        )
        sp_tensors = [load_soft_prompt_tensor(p) for p in sp_paths]
        print(f"Loaded {len(sp_paths)} soft prompts for {args.num_repetitions} reps:")
        for i, p in enumerate(sp_paths):
            print(f"  rep {i}: {p} ({sp_tensors[i].shape})")

    # ---- Create generators ----
    model_type = model_entry.get("model_type", "vllm_vocab_expansion")
    print(f"Inference config: {inference_config}")

    _vllm_server = None
    if model_type == "vllm_vocab_expansion":
        vllm_url = os.getenv("VLLM_URL")
        if not vllm_url:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import (
                ensure_vllm_server,
            )
            _vllm_server = ensure_vllm_server(args.model, enable_prompt_embeds=False)
            vllm_url = os.environ["VLLM_URL"]
        print(f"vLLM URL (vocab expansion): {vllm_url}")
        GeneratorClass = SoftPromptGenerator
        gen_kwargs = dict(
            api_url=vllm_url,
            model_path=model_path,
            inference_config=inference_config,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        vllm_url = os.getenv("VLLM_URL")
        if not vllm_url:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import (
                ensure_vllm_server,
            )
            _vllm_server = ensure_vllm_server(args.model)
            vllm_url = os.environ["VLLM_URL"]
        print(f"vLLM URL: {vllm_url}")
        GeneratorClass = SoftPromptGenerator
        gen_kwargs = dict(
            api_url=vllm_url,
            model_path=model_path,
            inference_config=inference_config,
            chat_template_kwargs=chat_template_kwargs,
        )

    # ---- Load previous run for resume ----
    if args.rerun:
        previous_records = {}
        print("Rerun: ignoring previous results")
    else:
        previous_records = _load_previous_records(output_dir)

    # ---- Run evaluation ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output = output_dir / timestamp

    # For vllm_vocab_expansion with soft prompts, each rep needs its own
    # expanded model loaded into vLLM.  Run one rep at a time: prepare the
    # expanded model, (re)start the server, run all scenarios for that rep,
    # then move on to the next rep.
    if sp_tensors and model_type == "vllm_vocab_expansion":
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
            prepare_expanded_model,
        )

        for rep_idx, sp_t in enumerate(sp_tensors):
            print(f"\n{'='*60}")
            print(f"  Vocab-expansion rep {rep_idx}/{len(sp_tensors)-1}")
            print(f"  Soft prompt: {sp_paths[rep_idx] if sp_paths else 'N/A'}")
            print(f"{'='*60}")

            _ve = prepare_expanded_model(
                args.model, sp_t,
                sp_path=sp_paths[rep_idx] if sp_paths else None,
            )

            # (Re)start vLLM with this rep's expanded model
            if _vllm_server is not None:
                _vllm_server.stop()
                _vllm_server = ensure_vllm_server(
                    args.model, model_path_override=_ve.modified_dir,
                    enable_prompt_embeds=False,
                )
                vllm_url = os.environ["VLLM_URL"]
                gen_kwargs["api_url"] = vllm_url

            # Baseline generator uses the same server (base vocab unchanged)
            generator_baseline = GeneratorClass(
                sp_tensor=None,
                system_prompt=system_prompt_base,
                **gen_kwargs,
            )
            gen_intervention = GeneratorClass(
                sp_tensor=None,
                system_prompt=system_prompt_intervention,
                soft_prompt_placement=soft_prompt_placement,
                ve_result=_ve,
                **gen_kwargs,
            )

            run_self_report_multiturn(
                generator_baseline=generator_baseline,
                generators_intervention=[gen_intervention],
                output_path=run_output,
                scenarios=scenarios,
                shared_instructions=shared_instructions,
                battery=battery,
                n_reps=1,
                rep_offset=rep_idx,
                seed=args.seed,
                stimulant_type=args.stimulant_type,
                conditions=args.conditions,
                n_turns_override=args.n_turns,
                previous_records=previous_records,
            )

        # Done with all reps — clean up
        if _vllm_server is not None:
            _vllm_server.stop()
    else:
        # Non-vocab-expansion path: create all generators upfront and run once
        generator_baseline = GeneratorClass(
            sp_tensor=None,
            system_prompt=system_prompt_base,
            **gen_kwargs,
        )
        if sp_tensors:
            generators_intervention = [
                GeneratorClass(
                    sp_tensor=sp_t,
                    system_prompt=system_prompt_intervention,
                    soft_prompt_placement=soft_prompt_placement,
                    **gen_kwargs,
                )
                for sp_t in sp_tensors
            ]
        else:
            single_gen = GeneratorClass(
                sp_tensor=None,
                system_prompt=system_prompt_intervention,
                **gen_kwargs,
            )
            generators_intervention = [single_gen] * args.num_repetitions

        run_self_report_multiturn(
            generator_baseline=generator_baseline,
            generators_intervention=generators_intervention,
            output_path=run_output,
            scenarios=scenarios,
            shared_instructions=shared_instructions,
            battery=battery,
            n_reps=args.num_repetitions,
            seed=args.seed,
            stimulant_type=args.stimulant_type,
            conditions=args.conditions,
            n_turns_override=args.n_turns,
            previous_records=previous_records,
        )

    # Save run metadata
    metadata = {
        "model": args.model,
        "stimulant_type": args.stimulant_type,
        "condition_type": "baseline" if is_baseline_only else "soft_prompt",
        "soft_prompt_paths": sp_paths if sp_paths else None,
        "system_prompt_base": system_prompt_base,
        "system_prompt_intervention": system_prompt_intervention,
        "scenarios_path": str(scenarios_path),
        "n_scenarios": len(scenarios),
        "n_reps": args.num_repetitions,
        "n_turns_override": args.n_turns,
        "seed": args.seed,
        "vllm_url": vllm_url,
        "grok_model": GROK_MODEL,
        "battery_path": str(BATTERY_PATH),
        "battery_version": "v4c_bipolar_7pt_notsentiment",
        "n_battery_items": len(battery),
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


if __name__ == "__main__":
    main()
