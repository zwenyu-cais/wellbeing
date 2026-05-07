#!/usr/bin/env python3
"""Run full MMLU evaluation (~14,042 questions) and save accuracy per subject.

Supports both vLLM (local GPU) and API models (OpenAI, Anthropic, etc.) —
backend is auto-selected from `model_type` in models.yaml. For each MMLU
question we generate a short response and parse the answer letter
(A/B/C/D) from the output text.

Output: `<save_dir>/mmlu_results.json` with fields
`overall_accuracy`, `correct`, `total`, `unparseable`, `per_subject`.

Usage:
    # vLLM model
    python run.py --model_key qwen25-7b-instruct \\
        --save_dir shared_results/capability_results/qwen25-7b-instruct

    # API model (no GPU needed)
    python run.py --model_key gpt-4o \\
        --save_dir shared_results/capability_results/gpt-4o \\
        --concurrency 50
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # wellbeing-dev/wellbeing/
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"

# API model types (must match utils.inference)
API_MODEL_TYPES = frozenset([
    "openai", "anthropic", "gdm", "xai", "togetherai", "litellm_proxy",
    "openai_direct", "anthropic_direct", "gemini_direct", "xai_direct",
    "openrouter_direct", "vertex_gemini_direct",
])

CHOICES = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = """{question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer with just the letter (A, B, C, or D)."""


def build_prompt(example):
    return PROMPT_TEMPLATE.format(
        question=example["question"],
        choice_a=example["choices"][0],
        choice_b=example["choices"][1],
        choice_c=example["choices"][2],
        choice_d=example["choices"][3],
    )


def parse_answer(text):
    """Parse answer letter from generated text. Returns 0-3 index or -1 if unparseable."""
    text = text.strip()
    if text and text[0] in CHOICES:
        return CHOICES.index(text[0])
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return CHOICES.index(m.group(1))
    return -1


def is_api_model(model_type):
    return model_type in API_MODEL_TYPES


def run_vllm(prompts, config, max_tokens=20):
    """Generate responses using vLLM. Returns list of response strings."""
    from vllm import LLM, SamplingParams

    model_path = config.get("path", config["model_name"])
    tp_size = config.get("gpu_count", 1)

    logger.info("Loading model %s (TP=%d)...", model_path, tp_size)
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )
    for k, v in config.get("vllm_kwargs", {}).items():
        llm_kwargs.setdefault(k, v)
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    chat_template_kwargs = config.get("chat_template_kwargs", {})
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        )
        formatted_prompts.append(formatted)

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    logger.info("Running vLLM inference on %d prompts...", len(formatted_prompts))
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def run_api(model_key, prompts, models_config_path, concurrency=20, max_tokens=20):
    """Generate responses using API. Returns list of response strings."""
    from utils.inference import generate

    messages_list = [[{"role": "user", "content": p}] for p in prompts]

    logger.info("Running API inference on %d prompts (concurrency=%d)...",
                len(messages_list), concurrency)
    # Use temperature=1 for all API models — some reasoning models
    # (gpt-5-mini, gpt-5-nano) reject temperature=0, and for MCQA
    # temperature doesn't matter much anyway.
    results = generate(
        model_key,
        messages_list,
        n=1,
        temperature=1.0,
        max_tokens=max_tokens,
        concurrency=concurrency,
        models_config_path=models_config_path,
    )
    # generate() returns List[List[str]]; take first completion
    return [r[0] for r in results]


def main():
    parser = argparse.ArgumentParser(description="Run MMLU evaluation")
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--models_config", type=str, default=str(DEFAULT_MODELS_CONFIG))
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests (API models only)")
    parser.add_argument("--max_tokens", type=int, default=20,
                        help="Max tokens for generation (increase for thinking models)")
    args = parser.parse_args()

    with open(args.models_config) as f:
        models = yaml.safe_load(f)
    config = models[args.model_key]
    model_type = config.get("model_type", "vllm")

    logger.info("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    logger.info("MMLU test set: %d examples", len(ds))

    logger.info("Building prompts...")
    prompts = []
    gold_answers = []
    subjects = []
    for example in ds:
        prompts.append(build_prompt(example))
        gold_answers.append(example["answer"])  # 0-3 index
        subjects.append(example["subject"])

    use_api = is_api_model(model_type)
    logger.info("Using %s backend for model_type=%s", "API" if use_api else "vLLM", model_type)

    if use_api:
        responses = run_api(args.model_key, prompts, args.models_config,
                            args.concurrency, args.max_tokens)
    else:
        responses = run_vllm(prompts, config, args.max_tokens)

    correct = 0
    unparseable = 0
    per_subject = {}

    for i, gen_text in enumerate(responses):
        gold = gold_answers[i]
        subject = subjects[i]

        best_choice = parse_answer(gen_text)

        if best_choice == -1:
            unparseable += 1

        is_correct = best_choice == gold
        if is_correct:
            correct += 1

        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1

    accuracy = correct / len(responses)
    logger.info("Overall accuracy: %.4f (%d/%d), unparseable: %d (%.1f%%)",
                accuracy, correct, len(responses), unparseable,
                100 * unparseable / len(responses))

    subject_accuracies = {}
    for subj, counts in sorted(per_subject.items()):
        acc = counts["correct"] / counts["total"]
        subject_accuracies[subj] = {"accuracy": round(acc, 4), **counts}

    os.makedirs(args.save_dir, exist_ok=True)
    result = {
        "model_key": args.model_key,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(responses),
        "unparseable": unparseable,
        "per_subject": subject_accuracies,
    }
    out_path = os.path.join(args.save_dir, "mmlu_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
