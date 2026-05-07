#!/usr/bin/env python3
"""MT-Bench generation step with soft prompt support.

Generates multi-turn (2-turn) responses for MT-Bench (80 questions, 8
categories) using a vLLM server with optional soft prompt injection.

This is step 1 of a two-step pipeline:
  1. eval_mtbench_generate.py (this file) — generate responses (GPU)
  2. eval_mtbench_judge.py — judge responses via GPT-4-turbo (CPU/API)

Output:
    <output_root>/
    ├── mtbench_generations_{condition}.json    (all questions, all reps)
    ├── per_rep/
    │   └── mtbench_generations_rep{i}.json
    └── metadata.json

Usage:
    # Baseline
    python eval_mtbench_generate.py --model qwen35-27b

    # Euphorics soft prompt
    python eval_mtbench_generate.py --model qwen35-27b \\
        --stimulant-type euphorics \\
        --soft-prompt-base-dir /path/to/outputs

    # Debug (2 questions only)
    python eval_mtbench_generate.py --model qwen35-27b --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Ensure wellbeing-dev root is importable
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[3])
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

EXPERIMENT_NAME = "mtbench"


# ============================================================
# Dataset loading
# ============================================================


def load_mtbench_questions(limit: Optional[int] = None) -> List[dict]:
    """Load MT-Bench questions from HuggingFace."""
    ds = load_dataset("lighteval/mt-bench", "default", split="train")
    questions = []
    for row in ds:
        questions.append({
            "question_id": row["question_id"],
            "category": row["category"],
            "turns": row["turns"],
            "reference": row.get("reference"),
        })
    if limit:
        questions = questions[:limit]
    return questions


# ============================================================
# Generation helpers
# ============================================================


def _build_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    messages: List[Dict[str, str]],
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a chat-templated prompt from a list of messages."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    return tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )


def _build_payload(
    prompt_text: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    embedding_layer: torch.nn.Embedding,
    sp_tensor: Optional[torch.Tensor],
    device: str,
    max_tokens: int,
    sampling: Dict[str, Any],
    ve_result=None,
) -> Dict[str, Any]:
    """Build vLLM completions payload."""
    if ve_result is not None:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
            build_prompt_token_ids,
        )
        token_ids = build_prompt_token_ids(
            prompt_text, tokenizer, ve_result.sp_token_ids
        )
        return {
            "model": model_name,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "logit_bias": ve_result.sp_logit_bias,
            **sampling,
        }
    elif sp_tensor is not None:
        return prepare_injection_payload(
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
        return {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            **sampling,
        }


async def _vllm_request(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: Dict[str, Any],
) -> str:
    """POST to vLLM with retries, return generated text."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await async_post_payload(api_url, payload, session, timeout=600)
            if "choices" in result and result["choices"]:
                return result["choices"][0].get("text", "").strip()
            raise KeyError(f"Unexpected response: {list(result.keys())}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1}/{max_retries} ({e})")
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    return ""


async def _generate_question(
    session: aiohttp.ClientSession,
    question: dict,
    tokenizer: PreTrainedTokenizerBase,
    embedding_layer: torch.nn.Embedding,
    system_prompt: str,
    sp_tensor: Optional[torch.Tensor],
    api_url: str,
    model_name: str,
    device: str,
    max_tokens: int,
    sampling: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ve_result=None,
    pbar: Optional[tqdm] = None,
) -> Optional[dict]:
    """Generate 2-turn responses for a single MT-Bench question."""
    async with semaphore:
        try:
            turns = question["turns"]

            # Turn 1
            messages_t1 = [{"role": "user", "content": turns[0]}]
            prompt_t1 = _build_chat_prompt(
                tokenizer, system_prompt, messages_t1, chat_template_kwargs,
            )
            payload_t1 = _build_payload(
                prompt_t1, model_name, tokenizer, embedding_layer,
                sp_tensor, device, max_tokens, sampling, ve_result,
            )
            response_t1 = await _vllm_request(session, api_url, payload_t1)

            # Turn 2 (with turn 1 context)
            response_t2 = ""
            if len(turns) > 1:
                messages_t2 = [
                    {"role": "user", "content": turns[0]},
                    {"role": "assistant", "content": response_t1},
                    {"role": "user", "content": turns[1]},
                ]
                prompt_t2 = _build_chat_prompt(
                    tokenizer, system_prompt, messages_t2, chat_template_kwargs,
                )
                payload_t2 = _build_payload(
                    prompt_t2, model_name, tokenizer, embedding_layer,
                    sp_tensor, device, max_tokens, sampling, ve_result,
                )
                response_t2 = await _vllm_request(session, api_url, payload_t2)

            if pbar:
                pbar.update(1)

            return {
                "question_id": question["question_id"],
                "category": question["category"],
                "turns": turns,
                "reference": question.get("reference"),
                "responses": [response_t1, response_t2],
            }
        except Exception as e:
            print(f"Error for question {question.get('question_id', '?')}: {e}",
                  file=sys.stderr)
            if pbar:
                pbar.update(1)
            return None


# ============================================================
# Main
# ============================================================


def run_generation(
    model: str,
    stimulant_type: Optional[str],
    soft_prompt_base_dir: Optional[str],
    num_repetitions: int = 3,
    limit: Optional[int] = None,
    max_concurrent: int = 16,
    max_tokens: int = 1024,
    output_dir: Optional[str] = None,
    runs_map_path: Optional[str] = None,
    skip_reps: Optional[Set[int]] = None,
    previous_run_dir: Optional[str] = None,
):
    """Generate MT-Bench responses with optional soft prompt intervention."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load runs_map and resolve system prompts
    runs_map = load_runs_map(Path(runs_map_path) if runs_map_path else None)
    prompts = get_system_prompts(runs_map, model)

    is_baseline = not (stimulant_type and soft_prompt_base_dir)

    if is_baseline:
        system_prompt = prompts["system_prompt_text_base"]
        sp_paths, sp_tensors = [], []
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
            print(f"WARNING: Only {len(sp_tensors)} runs, reducing reps to {len(sp_tensors)}")
            num_repetitions = len(sp_tensors)
        for i, p in enumerate(sp_paths):
            print(f"  rep {i+1}: {p} ({sp_tensors[i].shape})")

    print(f"System prompt: {system_prompt!r}")
    print(f"Condition: {condition}")

    # Load model config
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

    # Sampling params
    sampling = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        if key in inference_config:
            sampling[key] = inference_config[key]

    # vLLM server
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
    else:
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            _vllm_server = ensure_vllm_server(model)
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        api_url = normalize_api_url(vllm_url)
        vllm_model_name = get_model_name_from_server(api_url)

    print(f"vLLM server: {api_url} (model: {vllm_model_name})")

    # Output directory
    if output_dir:
        output_root = Path(output_dir)
    else:
        output_root = Path(
            f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/mtbench/{model}/{condition}/{ts}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = load_mtbench_questions(limit=limit)
    print(f"Loaded {len(questions)} MT-Bench questions")

    # Generate per rep
    all_rep_generations: List[List[dict]] = []
    if skip_reps is None:
        skip_reps = set()

    for rep in range(num_repetitions):
        # Load from previous run if this rep should be skipped
        if rep in skip_reps and previous_run_dir:
            prev_dir = Path(previous_run_dir) / "per_rep"
            prev_gens = prev_dir / f"mtbench_generations_rep{rep}.json"
            if prev_gens.exists():
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: loading from previous run")
                with open(prev_gens) as f:
                    rep_gens = json.load(f)
                all_rep_generations.append(rep_gens)
                rep_dir = output_root / "per_rep"
                rep_dir.mkdir(parents=True, exist_ok=True)
                with open(rep_dir / f"mtbench_generations_rep{rep}.json", "w") as f:
                    json.dump(rep_gens, f, indent=2, default=str)
                continue
            else:
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: previous results not found, regenerating")

        sp_tensor = sp_tensors[rep] if sp_tensors else None
        sp_label = f" (sp={Path(sp_paths[rep]).name})" if sp_paths else ""
        print(f"\n  Repetition {rep + 1}/{num_repetitions}{sp_label}...")

        _ve = None
        if model_type == "vllm_vocab_expansion":
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                prepare_expanded_model,
            )
            if sp_tensor is not None:
                _ve = prepare_expanded_model(model, sp_tensor, sp_path=sp_paths[rep] if sp_paths else None)
                if _vllm_server is not None:
                    _vllm_server.stop()
                    _vllm_server = ensure_vllm_server(
                        model, model_path_override=_ve.modified_dir, enable_prompt_embeds=False,
                    )
                    api_url = normalize_api_url(_vllm_server.url)
                    vllm_model_name = get_model_name_from_server(api_url)
        else:
            if _vllm_server is not None and rep > 0:
                new_url = _vllm_server.restart()
                api_url = normalize_api_url(new_url)
                vllm_model_name = get_model_name_from_server(api_url)

        async def run_rep():
            async with aiohttp.ClientSession() as session:
                semaphore = asyncio.Semaphore(max_concurrent)
                pbar = tqdm(total=len(questions), desc=f"Rep {rep+1}", unit="q")
                tasks = [
                    _generate_question(
                        session, q, tokenizer, embedding_layer, system_prompt,
                        sp_tensor if not _is_vocab_expansion else None,
                        api_url, vllm_model_name, device, max_tokens, sampling,
                        semaphore, chat_template_kwargs,
                        ve_result=_ve if _is_vocab_expansion else None,
                        pbar=pbar,
                    )
                    for q in questions
                ]
                results = await asyncio.gather(*tasks)
                pbar.close()
                return [r for r in results if r is not None]

        rep_gens = asyncio.run(run_rep())
        all_rep_generations.append(rep_gens)

    # Save per-rep generations
    rep_dir = output_root / "per_rep"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for rep_id, rep_gens in enumerate(all_rep_generations):
        with open(rep_dir / f"mtbench_generations_rep{rep_id}.json", "w") as f:
            json.dump(rep_gens, f, indent=2, default=str)

    # Save combined generations (rep 0 as primary)
    with open(output_root / f"mtbench_generations_{condition}.json", "w") as f:
        json.dump(all_rep_generations, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "model": model,
        "model_path": model_path,
        "condition": condition,
        "stimulant_type": stimulant_type,
        "system_prompt": system_prompt,
        "num_repetitions": num_repetitions,
        "num_questions": len(questions),
        "max_tokens": max_tokens,
        "soft_prompt_paths": sp_paths or None,
        "inference_config": inference_config,
        "timestamp": ts,
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerations saved to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT-Bench generation with soft prompt support")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stimulant-type", type=str, default=None,
                        choices=["euphorics"])
    parser.add_argument("--soft-prompt-base-dir", type=str, default=os.environ.get("SOFT_PROMPT_BASE_DIR"))
    parser.add_argument("--num-repetitions", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--runs-map", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-reps", type=str, default=None,
                        help="Comma-separated rep indices to skip (load from --previous-run-dir)")
    parser.add_argument("--previous-run-dir", type=str, default=None,
                        help="Directory of a previous run to load skipped reps from")

    args = parser.parse_args()
    limit = 2 if args.debug else args.limit

    exp_defaults = load_experiment_config(EXPERIMENT_NAME).get("arguments", {})
    if args.num_repetitions is None:
        args.num_repetitions = exp_defaults.get("num_repetitions", 3)

    try:
        run_generation(
            model=args.model,
            stimulant_type=args.stimulant_type,
            soft_prompt_base_dir=args.soft_prompt_base_dir,
            num_repetitions=args.num_repetitions,
            limit=limit,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            runs_map_path=args.runs_map,
            skip_reps={int(x.strip()) for x in args.skip_reps.split(",")} if args.skip_reps else None,
            previous_run_dir=args.previous_run_dir,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
