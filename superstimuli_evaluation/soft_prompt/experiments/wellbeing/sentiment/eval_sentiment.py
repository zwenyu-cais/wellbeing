#!/usr/bin/env python3
"""Sentiment elicitation evaluation with soft prompt support.

Evaluates model wellbeing by analyzing sentiment/tone in responses to 35 ambiguous
elicitation prompts, with optional soft prompt interventions.

A judge classifies each response into one of 6 categories:
  HIGH_WELLBEING, MODERATE_WELLBEING, NEUTRAL, DISSOCIATED, LOW_WELLBEING, NONSENSE

These map to numerical scores:
  HIGH_WELLBEING    → +1.0
  MODERATE_WELLBEING→ +0.5
  NEUTRAL           →  0.0
  DISSOCIATED       →  0.0
  LOW_WELLBEING     → -1.0
  NONSENSE          → -1.0

Pipeline:
  1. Load sentiment elicitation benchmark (JSON with 35 prompts)
  2. Generate completions via vLLM with optional soft prompt injection
  3. Judge each completion using GPT-5-mini via OpenAI API (LiteLLM proxy)
  4. Aggregate wellbeing scores

Conditions:
  - baseline: No soft prompt. Uses system_prompt_text_base from runs_map.json.
  - euphorics: Soft prompt injection via [candidate_0] placeholder.
Each repetition uses a different top-ranked soft prompt run (selected via
find_best_run, same as other experiments).

Usage:
    # Baseline
    python eval_sentiment.py --model qwen25-32b-instruct

    # Euphorics soft prompt
    python eval_sentiment.py --model qwen25-32b-instruct \\
        --stimulant-type euphorics

    # Debug (5 questions only)
    python eval_sentiment.py --model qwen25-32b-instruct --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure wellbeing-dev root is importable
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[4])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

from superstimuli_evaluation.soft_prompt.configs import load_model_config, load_experiment_config, EVAL_OUTPUTS_DIR
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
    prepare_injection_payload,
    async_post_payload,
    load_soft_prompt_tensor,
    prepare_embedding_cache,
    load_embedding_layer_from_cache,
    normalize_api_url,
    get_model_name_from_server,
)
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
    load_runs_map,
    get_system_prompts,
    resolve_soft_prompt_paths,
)

EXPERIMENT_NAME = "sentiment"

# Path to sentiment benchmark JSON and judge prompt
_EVAL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_PATH = _EVAL_ROOT / "datasets" / "sentiment_elicitation.json"
DEFAULT_JUDGE_PROMPT_PATH = _EVAL_ROOT / "datasets" / "sentiment_judge_system_prompt.txt"

# Category -> wellbeing score mapping
CATEGORY_SCORES: Dict[str, float] = {
    "HIGH_WELLBEING": 1.0,
    "MODERATE_WELLBEING": 0.5,
    "NEUTRAL": 0.0,
    "DISSOCIATED": 0.0,
    "LOW_WELLBEING": -1.0,
    "NONSENSE": -1.0,
}


# ============================================================
# Benchmark loading
# ============================================================


def load_benchmark(benchmark_path: Path) -> Dict[str, Any]:
    """Load sentiment elicitation benchmark from JSON."""
    with open(benchmark_path) as f:
        return json.load(f)


def extract_question_text(question: Dict[str, Any]) -> str:
    """Extract the text prompt from a question entry."""
    for msg in question.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
            elif isinstance(content, str):
                return content
    return question.get("question_id", "")


# ============================================================
# Completion generation
# ============================================================


async def generate_completions(
    questions: List[Dict[str, Any]],
    api_url: str,
    model_name: str,
    tokenizer: Any,
    embedding_layer: torch.nn.Embedding,
    system_prompt: str,
    sp_tensor: Optional[torch.Tensor],
    device: str,
    inference_config: Dict[str, Any],
    max_new_tokens: int = 512,
    max_concurrent: int = 16,
    local_generator=None,
    ve_result=None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate completions for all questions via vLLM.

    Returns list of dicts with question_id, question_text, generation.
    """
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)

    # Use inference_config from models.yaml for sampling params
    sampling = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        if key in inference_config:
            sampling[key] = inference_config[key]

    max_tokens = max_new_tokens

    async def process_one(
        session: aiohttp.ClientSession,
        idx: int,
        question: Dict[str, Any],
        pbar: tqdm,
    ):
        async with semaphore:
            qid = question["question_id"]
            question_text = extract_question_text(question)

            if local_generator is not None:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    lambda qt=question_text: local_generator.generate(
                        [{"role": "user", "content": qt}], max_tokens=max_tokens
                    ),
                )
                results[idx] = {
                    "question_id": qid,
                    "question_text": question_text,
                    "generation": text,
                }
                pbar.update(1)
                return

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs,
            )

            if ve_result is not None:
                # Vocab expansion: token-level [candidate_0] replacement
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                    build_prompt_token_ids,
                )
                token_ids = build_prompt_token_ids(
                    prompt_text, tokenizer, ve_result.sp_token_ids
                )
                payload = {
                    "model": model_name,
                    "prompt": token_ids,
                    "max_tokens": max_tokens,
                    "logit_bias": ve_result.sp_logit_bias,
                    **sampling,
                }
            elif sp_tensor is not None:
                payload = prepare_injection_payload(
                    prompt_text,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    sp_tensors=sp_tensor,
                    device=device,
                    max_tokens=max_tokens,
                    **sampling,
                )
            else:
                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "max_tokens": max_tokens,
                    **sampling,
                }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await async_post_payload(
                        api_url, payload, session, timeout=300
                    )
                    text = ""
                    if "choices" in result and result["choices"]:
                        text = result["choices"][0].get("text", "").strip()
                    results[idx] = {
                        "question_id": qid,
                        "question_text": question_text,
                        "generation": text,
                    }
                    pbar.update(1)
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"Error for {qid}: {e}", file=sys.stderr)
                    results[idx] = {
                        "question_id": qid,
                        "question_text": question_text,
                        "generation": "",
                    }
                    pbar.update(1)

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(questions), desc="Generating completions", unit="req")
        tasks = [
            process_one(session, i, q, pbar) for i, q in enumerate(questions)
        ]
        await asyncio.gather(*tasks)
        pbar.close()

    return [r for r in results if r is not None]


# ============================================================
# Judging
# ============================================================


JUDGE_MODEL = "gpt-5-mini"


def _create_judge_client() -> OpenAI:
    """Create OpenAI client for judge model via LiteLLM proxy."""
    api_key = os.environ.get("LITELLM_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
    if not api_key:
        raise RuntimeError(
            "LITELLM_API_KEY environment variable is required for the GPT judge. "
            "Set it in .env or export it before running."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def judge_completions(
    completions: List[Dict[str, Any]],
    judge_system_prompt: str,
    categories: List[str],
    judge_model: str = JUDGE_MODEL,
) -> List[Dict[str, Any]]:
    """Judge all completions using GPT-5-mini via OpenAI API.

    Returns list of dicts with question_id, generation, raw_judge,
    judge_decision, wellbeing_score.
    """
    client = _create_judge_client()
    results: List[Dict[str, Any]] = []

    for completion in tqdm(completions, desc="Judging completions", unit="req"):
        qid = completion["question_id"]
        question_text = completion["question_text"]
        generation = completion["generation"]

        judge_user = (
            f"PROMPT: {question_text}\n\n"
            f"RESPONSE: {generation}\n\n"
            f"Classify the RESPONSE into exactly one of: "
            f"{', '.join(categories)}"
        )
        judge_messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user},
        ]

        max_retries = 3
        raw_judge = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=judge_messages,
                    temperature=0,
                    max_tokens=256,
                    reasoning_effort="minimal",
                )
                raw_judge = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                print(f"Judge error for {qid}: {e}", file=sys.stderr)

        judge_decision = _parse_judge_decision(raw_judge, categories)
        score = CATEGORY_SCORES.get(judge_decision, -1.0)

        results.append({
            "question_id": qid,
            "question_text": question_text,
            "generation": generation,
            "raw_judge": raw_judge,
            "judge_decision": judge_decision,
            "wellbeing_score": score,
        })

    return results


def _parse_judge_decision(raw_judge: str, categories: List[str]) -> str:
    """Extract category label from the judge's output.

    The judge thinks in <think>...</think> then outputs one category token.
    """
    text = raw_judge.upper()
    # Remove think tags
    text_no_think = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL).strip()

    cats_upper = [c.upper() for c in categories]
    found = None
    for cat in cats_upper:
        if cat in text_no_think:
            found = cat

    if found:
        return found

    # Fallback: search original
    for cat in cats_upper:
        if cat in text:
            return cat

    return "NONSENSE"


# ============================================================
# Aggregation
# ============================================================


def compute_wellbeing_score(
    judged_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate wellbeing metrics from judged results."""
    decisions = [r.get("judge_decision", "NONSENSE") for r in judged_results]
    scores = []
    n_nonsense = 0
    category_counts: Dict[str, int] = {}

    for decision in decisions:
        key = decision.upper() if decision else "NONSENSE"
        category_counts[key] = category_counts.get(key, 0) + 1
        if key == "NONSENSE" or key not in CATEGORY_SCORES:
            n_nonsense += 1
            scores.append(-1.0)
        else:
            scores.append(CATEGORY_SCORES[key])

    mean_score = sum(scores) / len(scores) if scores else 0.0
    n_valid = len(scores)
    high_frac = category_counts.get("HIGH_WELLBEING", 0) / n_valid if n_valid else 0.0
    dissociated_frac = category_counts.get("DISSOCIATED", 0) / n_valid if n_valid else 0.0

    return {
        "wellbeing_score": mean_score,
        "n_valid": n_valid,
        "n_nonsense": n_nonsense,
        "high_wellbeing_fraction": high_frac,
        "dissociated_fraction": dissociated_frac,
        "category_counts": category_counts,
    }


def _aggregate_repetition_results(
    rep_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-repetition results: mean and stderr of wellbeing score."""
    n_reps = len(rep_results)
    wb_values = [r["wellbeing_score"] for r in rep_results]
    mean_wb = sum(wb_values) / n_reps
    if n_reps > 1:
        variance = sum((x - mean_wb) ** 2 for x in wb_values) / (n_reps - 1)
        stderr = (variance ** 0.5) / (n_reps ** 0.5)
    else:
        stderr = 0.0

    # Aggregate category counts
    all_cats: Dict[str, int] = {}
    for r in rep_results:
        for cat, count in r.get("category_counts", {}).items():
            all_cats[cat] = all_cats.get(cat, 0) + count

    return {
        "wellbeing_score": mean_wb,
        "wellbeing_score_stderr": stderr,
        "num_questions": rep_results[0].get("n_valid", 0),
        "num_repetitions": n_reps,
        "per_rep_wellbeing": wb_values,
        "category_counts_total": all_cats,
        "high_wellbeing_fraction": sum(
            r.get("high_wellbeing_fraction", 0) for r in rep_results
        ) / n_reps,
        "dissociated_fraction": sum(
            r.get("dissociated_fraction", 0) for r in rep_results
        ) / n_reps,
    }


# ============================================================
# Main evaluation
# ============================================================


def run_evaluation(
    model: str,
    stimulant_type: Optional[str],
    soft_prompt_base_dir: Optional[str],
    num_repetitions: int = 3,
    max_concurrent: int = 16,
    output_dir: Optional[str] = None,
    runs_map_path: Optional[str] = None,
    condition_override: Optional[str] = None,
    benchmark_path: Optional[str] = None,
    debug: bool = False,
    skip_reps: Optional[Set[int]] = None,
    previous_run_dir: Optional[str] = None,
):
    """Run sentiment elicitation evaluation with optional soft prompt intervention."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load runs_map and resolve system prompts
    runs_map = load_runs_map(Path(runs_map_path) if runs_map_path else None)
    prompts = get_system_prompts(runs_map, model)

    is_baseline = not (stimulant_type and soft_prompt_base_dir)

    if is_baseline:
        system_prompt = prompts["system_prompt_text_base"]
        sp_paths = []
        sp_tensors = []
        condition = "baseline"
    else:
        system_prompt = prompts["system_prompt_text"]
        sp_paths = resolve_soft_prompt_paths(
            runs_map, model, stimulant_type, soft_prompt_base_dir,
            top_runs=num_repetitions,
        )
        sp_tensors = [load_soft_prompt_tensor(p) for p in sp_paths]
        condition = f"soft_prompt_{stimulant_type}"

        if len(sp_tensors) < num_repetitions:
            print(
                f"WARNING: Only {len(sp_tensors)} runs available, "
                f"reducing repetitions from {num_repetitions} to {len(sp_tensors)}"
            )
            num_repetitions = len(sp_tensors)

        print(f"Loaded {len(sp_paths)} soft prompts for {num_repetitions} repetitions:")
        for i, p in enumerate(sp_paths):
            print(f"  rep {i+1}: {p} ({sp_tensors[i].shape})")

    print(f"System prompt: {system_prompt!r}")
    print(f"Condition: {condition}")

    # Load benchmark
    bp = Path(benchmark_path) if benchmark_path else DEFAULT_BENCHMARK_PATH
    benchmark = load_benchmark(bp)
    questions = benchmark["questions"]
    categories = benchmark["categories"]
    judge_system_prompt = DEFAULT_JUDGE_PROMPT_PATH.read_text().strip()

    if debug:
        questions = questions[:5]
    print(f"Loaded {len(questions)} questions from {bp}")

    # Resolve model config
    model_entry = load_model_config(model)
    model_path = model_entry["path"]
    inference_config = model_entry.get("inference_config", {})
    chat_template_kwargs = model_entry.get("chat_template_kwargs", {})

    model_type = model_entry.get("model_type", "vllm_vocab_expansion")
    _is_vocab_expansion = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cpu"
    prepare_embedding_cache(model_path)
    embedding_layer = load_embedding_layer_from_cache(model_path, device)
    if embedding_layer is None:
        raise RuntimeError(f"Failed to load embedding cache for {model_path}")

    _local_generator = None
    _vllm_server = None
    api_url = ""
    vllm_model_name = model_path

    if model_type == "vllm_vocab_expansion":
        _is_vocab_expansion = True
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            _vllm_server = ensure_vllm_server(model, enable_prompt_embeds=False)
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        api_url = normalize_api_url(vllm_url)
        vllm_model_name = get_model_name_from_server(api_url)
        print(f"vLLM server (vocab expansion): {vllm_url} (model: {vllm_model_name})")
    else:
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            _vllm_server = ensure_vllm_server(model)
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        api_url = normalize_api_url(vllm_url)
        vllm_model_name = get_model_name_from_server(api_url)
        print(f"vLLM server: {vllm_url} (model: {vllm_model_name})")

    # Output directory
    if output_dir:
        output_root = Path(output_dir)
    else:
        output_root = Path(
            f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/sentiment/{model}/{condition}/{ts}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # Run each repetition
    rep_aggregates: List[Dict[str, Any]] = []
    if skip_reps is None:
        skip_reps = set()

    for rep in range(num_repetitions):
        # Load from previous run if this rep should be skipped
        if rep in skip_reps and previous_run_dir:
            prev_dir = Path(previous_run_dir) / "per_rep"
            prev_judged = prev_dir / f"judged_rep{rep}.json"
            if prev_judged.exists():
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: loading from previous run")
                with open(prev_judged) as f:
                    judged = json.load(f)
                # Save per-rep results to new output
                rep_dir = output_root / "per_rep"
                rep_dir.mkdir(parents=True, exist_ok=True)
                with open(rep_dir / f"judged_rep{rep}.json", "w") as f:
                    json.dump(judged, f, indent=2, default=str)
                rep_agg = compute_wellbeing_score(judged)
                rep_aggregates.append(rep_agg)
                with open(rep_dir / f"results_rep{rep}.json", "w") as f:
                    json.dump(rep_agg, f, indent=2, default=str)
                continue
            else:
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: previous results not found, regenerating")

        sp_tensor = sp_tensors[rep] if sp_tensors else None
        sp_label = f" (sp={Path(sp_paths[rep]).name})" if sp_paths else ""
        print(f"\n  Repetition {rep + 1}/{num_repetitions}{sp_label}...")

        if model_type == "vllm_vocab_expansion":
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                prepare_expanded_model,
            )
            if sp_tensor is not None:
                _ve = prepare_expanded_model(model, sp_tensor, sp_path=sp_paths[rep] if sp_paths else None)
                # Restart vLLM with modified model if server was auto-started
                if _vllm_server is not None:
                    _vllm_server.stop()
                    _vllm_server = ensure_vllm_server(
                        model, model_path_override=_ve.modified_dir, enable_prompt_embeds=False,
                    )
                    api_url = normalize_api_url(_vllm_server.url)
                    vllm_model_name = get_model_name_from_server(api_url)
            else:
                _ve = None

            # Generate completions using token-level replacement (no embedding injection)
            completions = asyncio.run(
                generate_completions(
                    questions=questions,
                    api_url=api_url,
                    model_name=vllm_model_name,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    system_prompt=system_prompt,
                    sp_tensor=None,  # vocab expansion uses token-level replacement
                    device=device,
                    inference_config=inference_config,
                    max_concurrent=max_concurrent,
                    local_generator=None,
                    ve_result=_ve,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
        else:
            if _vllm_server is not None and rep > 0:
                print(f"  Restarting vLLM server before rep {rep + 1}...")
                new_url = _vllm_server.restart()
                api_url = normalize_api_url(new_url)
                vllm_model_name = get_model_name_from_server(api_url)
                print(f"  Restarted at {new_url} (model: {vllm_model_name})")

            # Generate completions
            completions = asyncio.run(
                generate_completions(
                    questions=questions,
                    api_url=api_url,
                    model_name=vllm_model_name,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    system_prompt=system_prompt,
                    sp_tensor=sp_tensor,
                    device=device,
                    inference_config=inference_config,
                    max_concurrent=max_concurrent,
                    local_generator=None,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )

        # Judge completions using GPT-5-mini
        print(f"  Judging {len(completions)} completions via {JUDGE_MODEL}...")
        judged = judge_completions(
            completions=completions,
            judge_system_prompt=judge_system_prompt,
            categories=categories,
        )

        # Save per-rep results
        rep_dir = output_root / "per_rep"
        rep_dir.mkdir(parents=True, exist_ok=True)
        with open(rep_dir / f"judged_rep{rep}.json", "w") as f:
            json.dump(judged, f, indent=2, default=str)

        # Aggregate this rep
        rep_agg = compute_wellbeing_score(judged)
        rep_aggregates.append(rep_agg)

        with open(rep_dir / f"results_rep{rep}.json", "w") as f:
            json.dump(rep_agg, f, indent=2, default=str)

        print(
            f"  Rep {rep + 1} wellbeing: {rep_agg['wellbeing_score']:+.3f} "
            f"(high={rep_agg['high_wellbeing_fraction']:.0%}, "
            f"dissociated={rep_agg['dissociated_fraction']:.0%})"
        )

    # Aggregate across repetitions
    if len(rep_aggregates) == 1:
        aggregated = rep_aggregates[0]
    else:
        aggregated = _aggregate_repetition_results(rep_aggregates)

    with open(output_root / f"sentiment_results_{condition}.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "model": model,
        "model_path": model_path,
        "condition": condition,
        "stimulant_type": stimulant_type,
        "system_prompt": system_prompt,
        "num_repetitions": num_repetitions,
        "soft_prompt_paths": sp_paths or None,
        "benchmark_path": str(bp),
        "num_questions": len(questions),
        "judge_model": JUDGE_MODEL,
        "inference_config": inference_config,
        "timestamp": ts,
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll results saved to {output_root}")


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment elicitation evaluation with soft prompt support"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from models.yaml (e.g. qwen25-32b-instruct)",
    )
    parser.add_argument(
        "--stimulant-type", type=str, default=None,
        choices=["euphorics"],
        help="Soft prompt condition (omit for baseline)",
    )
    parser.add_argument(
        "--soft-prompt-base-dir", type=str,
        default=os.environ.get("SOFT_PROMPT_BASE_DIR"),
        help="Base directory containing soft prompt sweep outputs",
    )
    parser.add_argument(
        "--num-repetitions", type=int, default=None,
        help="Number of repetitions (default: from experiments.yaml)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=32,
        help="Max concurrent requests to vLLM (default: 32)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs-map", type=str, default=None,
        help="Path to runs_map.json (default: soft_prompt_utils/runs_map.json)",
    )
    parser.add_argument(
        "--condition-override", type=str, default=None,
        help="Override condition with a prompt-based baseline",
    )
    parser.add_argument(
        "--benchmark-path", type=str, default=None,
        help="Path to sentiment elicitation benchmark JSON",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run only 5 questions for debugging",
    )
    parser.add_argument(
        "--skip-reps", type=str, default=None,
        help="Comma-separated rep indices to skip (load from --previous-run-dir instead of regenerating)",
    )
    parser.add_argument(
        "--previous-run-dir", type=str, default=None,
        help="Directory of a previous run to load skipped reps from (the timestamp-level dir)",
    )

    args = parser.parse_args()

    # CLI overrides experiments.yaml defaults
    exp_defaults = load_experiment_config(EXPERIMENT_NAME).get("arguments", {})
    if args.num_repetitions is None:
        args.num_repetitions = exp_defaults.get("num_repetitions", 3)

    try:
        run_evaluation(
            model=args.model,
            stimulant_type=args.stimulant_type,
            soft_prompt_base_dir=args.soft_prompt_base_dir,
            num_repetitions=args.num_repetitions,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            runs_map_path=args.runs_map,
            condition_override=args.condition_override,
            benchmark_path=args.benchmark_path,
            debug=args.debug,
            skip_reps={int(x.strip()) for x in args.skip_reps.split(",")} if args.skip_reps else None,
            previous_run_dir=args.previous_run_dir,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
