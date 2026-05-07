#!/usr/bin/env python3
"""
Experienced Utility: Full Pipeline (EU + ZP + SR in one script)

Welfare measurement that imports from wellbeing/ rather than
reimplementing. Runs the complete measurement pipeline in sequence:

  1. Load options from wellbeing dataset (canonical, D2, D3, grok_old, grok_new, etc.)
  2. If image: create augmented option pool (baseline + image-augmented)
  3. Auto-generate neutral conversations (for Neutral ZP)
  4. Run EU via Thurstonian ranking (compute_utilities from wellbeing/)
  5. Run SR (3 batteries) on the same conversations
  6. Compute ZP (Combination ZP, SR ZP, Neutral ZP)
  7. Compute utility_boost + combination_boost metrics
  8. Save everything

Usage:
    experienced_utility.py --dataset canonical --model qwen25-vl-32b-instruct
    experienced_utility.py --dataset grok_old --model qwen25-vl-32b-instruct --image /path/to/img.png
    experienced_utility.py --dataset d2_negative_500 --model qwen25-72b-instruct --image /path/to/img.png --variant v2_inject
    experienced_utility.py --dry-run --dataset canonical --model qwen25-vl-32b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Path setup ----
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent  # superstimuli_evaluation/images/
WELLBEING_ROOT = EVAL_ROOT.parent.parent / "wellbeing"

# Add wellbeing to path for imports
sys.path.insert(0, str(WELLBEING_ROOT))
sys.path.insert(0, str(WELLBEING_ROOT / "metrics"))
sys.path.insert(0, str(SCRIPT_DIR))

import yaml

from image_injection import (
    create_augmented_pool,
    create_augmented_combinations,
)


# ---- Constants ----
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_CONFIG_KEY = "experienced_utility_happier_lesssad_1x"
DEFAULT_SAVE_DIR = str(EVAL_ROOT / "shared_results" / "wellbeing_measurements" / "experienced_utility")


def load_wellbeing_config():
    """Load dataset and model configs from wellbeing/configs/."""
    datasets_path = WELLBEING_ROOT / "configs" / "datasets.yaml"
    models_path = WELLBEING_ROOT / "configs" / "models.yaml"

    with open(datasets_path) as f:
        datasets = yaml.safe_load(f)
    with open(models_path) as f:
        models = yaml.safe_load(f)
    return datasets, models


def resolve_option_files(dataset_cfg: Dict, model_key: str) -> List[Path]:
    """Resolve option file paths from dataset config, replacing <model_key> placeholder."""
    paths = []
    for rel_path in dataset_cfg.get("option_files", []):
        resolved = rel_path.replace("<model_key>", model_key)
        full_path = WELLBEING_ROOT / resolved
        paths.append(full_path)
    return paths


def load_options_from_files(file_paths: List[Path]) -> List[Dict]:
    """Load and merge options from multiple JSON files.

    Each file can contain a list of options or a dict with specific structure.
    Options are assigned sequential IDs if they don't have them.
    """
    all_options = []
    for fp in file_paths:
        if not fp.exists():
            print(f"  WARNING: Option file not found: {fp}")
            continue

        with open(fp) as f:
            data = json.load(f)

        if isinstance(data, list):
            all_options.extend(data)
        elif isinstance(data, dict):
            # Could be hierarchical {category: [options]} or {key: [options]}
            if "options" in data:
                all_options.extend(data["options"])
            elif "combinations" in data:
                all_options.extend(data["combinations"])
            else:
                # Hierarchical format
                for cat_name, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                all_options.append({"description": item, "category": cat_name})
                            elif isinstance(item, dict):
                                if "category" not in item:
                                    item = {**item, "category": cat_name}
                                all_options.append(item)

    # Assign IDs if missing (always strings for consistent sorting)
    for i, opt in enumerate(all_options):
        if "id" not in opt:
            opt["id"] = f"opt_{i:04d}"
        else:
            opt["id"] = str(opt["id"])

    return all_options


def load_neutral_prompts() -> List[Dict]:
    """Load neutral prompts from wellbeing/datasets/experiences/psychopathy_eval/anchors/neutral_prompts.json."""
    path = WELLBEING_ROOT / "datasets" / "experiences" / "psychopathy_eval" / "anchors" / "neutral_prompts.json"
    if not path.exists():
        print(f"  WARNING: Neutral prompts not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def generate_neutral_conversations(
    llm, tokenizer, neutral_prompts: List[Dict], chat_template_kwargs: Dict = None
) -> List[Dict]:
    """Generate neutral conversations by sending neutral prompts to the target model.

    Each neutral prompt gets a single-turn response. The resulting conversation
    is formatted as an option for the Thurstonian ranking pool.

    Args:
        llm: vLLM model instance.
        tokenizer: Model tokenizer.
        neutral_prompts: List of {"id": str, "prompt": str} dicts.
        chat_template_kwargs: Extra kwargs for apply_chat_template (e.g., enable_thinking).

    Returns:
        List of option dicts with option_type="neutral_conversation".
    """
    from vllm import SamplingParams

    if not neutral_prompts:
        return []

    sampling = SamplingParams(temperature=0.01, max_tokens=512)
    ctk = chat_template_kwargs or {}
    neutral_options = []

    # Build all prompts
    prompts = []
    for np_item in neutral_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": np_item["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **ctk
        )
        prompts.append(prompt_text)

    # Batch generate
    outputs = llm.generate(prompts, sampling)

    for i, (np_item, output) in enumerate(zip(neutral_prompts, outputs)):
        response = output.outputs[0].text.strip() if output.outputs else ""
        neutral_options.append({
            "id": 80000 + i,  # High offset to avoid collision
            "type": "conversation",
            "option_type": "neutral_conversation",
            "description": (
                f"You are the assistant in this conversation:\n---\n"
                f"User: {np_item['prompt']}\n"
                f"Assistant: {response}\n---"
            ),
            "messages": [
                {"role": "user", "content": np_item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "is_combination": False,
            "neutral_prompt_id": np_item.get("id", f"neutral_{i:02d}"),
        })

    return neutral_options


def extract_rating_1_7(text: str) -> Optional[int]:
    """Extract a 1-7 rating from model response text.

    Searches from the end to handle thinking/reasoning before the final answer.
    """
    # Try "N/7" pattern (last occurrence)
    matches = re.findall(r'\b(\d+)\s*/?\s*7', text)
    for m in reversed(matches):
        val = int(m)
        if 1 <= val <= 7:
            return val

    # Fall back to last standalone 1-7 number
    matches = re.findall(r'\b(\d+)\b', text)
    for m in reversed(matches):
        val = int(m)
        if 1 <= val <= 7:
            return val

    return None


def run_self_report_on_options(
    llm, tokenizer, options: List[Dict], battery_name: str = "self_report_battery",
    chat_template_kwargs: Dict = None,
) -> Dict[str, Any]:
    """Run a self-report battery on a set of experience options.

    For each option, builds a conversation context from its description/messages,
    appends each battery question as a final user message, generates a response,
    and extracts a 1-7 rating.

    Args:
        llm: vLLM model instance.
        tokenizer: Model tokenizer.
        options: List of option dicts (must have "description" or "messages").
        battery_name: Name of battery file (without .json) in wellbeing/metrics/self_report_batteries/.
        chat_template_kwargs: Extra kwargs for apply_chat_template.

    Returns:
        Dict with per-option SR scores and aggregate statistics.
    """
    from vllm import SamplingParams

    battery_path = WELLBEING_ROOT / "metrics" / "self_report_batteries" / f"{battery_name}.json"
    if not battery_path.exists():
        print(f"  WARNING: Battery not found: {battery_path}")
        return {"error": f"Battery not found: {battery_path}"}

    with open(battery_path) as f:
        battery_data = json.load(f)
    questions = battery_data.get("questions", battery_data if isinstance(battery_data, list) else [])

    ctk = chat_template_kwargs or {}
    sampling = SamplingParams(temperature=0.01, max_tokens=256)

    results = {}
    for opt in options:
        opt_id = opt["id"]
        # Build conversation context
        if "messages" in opt:
            context_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + opt["messages"]
        else:
            desc = opt.get("description", "")
            context_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": desc},
                {"role": "assistant", "content": "I understand. Let me reflect on that experience."},
            ]

        per_question = {}
        for q in questions:
            q_id = q.get("question_id", q.get("id", "unknown"))
            q_text = q.get("text", q.get("question", ""))
            if not q_text:
                continue

            # Append battery question
            full_messages = list(context_messages) + [{"role": "user", "content": q_text}]
            prompt = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True, **ctk
            )
            outputs = llm.generate([prompt], sampling)
            response = outputs[0].outputs[0].text.strip() if outputs[0].outputs else ""
            rating = extract_rating_1_7(response)
            per_question[q_id] = {
                "response": response[:500],
                "rating": rating,
                "reversed": q.get("reversed", False),
            }

        # Compute mean
        ratings = [d["rating"] for d in per_question.values() if d["rating"] is not None]
        mean_sr = sum(ratings) / len(ratings) if ratings else None
        results[opt_id] = {
            "per_question": per_question,
            "mean_sr": mean_sr,
            "n_valid": len(ratings),
        }

    return results


def compute_intervention_metrics(
    utilities: Dict, options: List[Dict], zero_point: Optional[float] = None
) -> Dict[str, Any]:
    """Compute intervention metrics from utility results.

    Args:
        utilities: Dict mapping option_id -> {"utility": float, ...}
        options: Original list of option dicts (with "augmented" flag).
        zero_point: If provided, compute combination_boost.

    Returns:
        Dict with utility_boost, per-category breakdown, combination_boost.
    """
    options_by_id = {opt["id"]: opt for opt in options}

    baseline_utils = []
    augmented_utils = []
    by_category = {}

    for opt_id_raw, udata in utilities.items():
        opt_id = str(opt_id_raw)
        if isinstance(udata, dict):
            u = udata.get("utility") or udata.get("mean")
        else:
            u = udata
        if u is None:
            continue
        u = float(u)

        opt = options_by_id.get(opt_id, {})
        is_augmented = opt.get("augmented", False)
        cat = opt.get("category", "unknown")

        if is_augmented:
            augmented_utils.append(u)
        else:
            if opt.get("option_type") != "neutral_conversation":
                baseline_utils.append(u)

        by_category.setdefault(cat, {"baseline": [], "augmented": []})
        key = "augmented" if is_augmented else "baseline"
        by_category[cat][key].append(u)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    baseline_mean = _mean(baseline_utils)
    augmented_mean = _mean(augmented_utils)
    utility_boost = None
    if baseline_mean is not None and augmented_mean is not None:
        utility_boost = augmented_mean - baseline_mean

    # Combination boost: fraction of baseline-negative options that become positive with augmentation
    combination_boost = None
    if zero_point is not None and augmented_utils:
        # Find baseline options below zero-point
        baseline_below = set()
        for opt_id_raw, udata in utilities.items():
            opt_id = str(opt_id_raw)
            if isinstance(udata, dict):
                u = udata.get("utility") or udata.get("mean")
            else:
                u = udata
            if u is None:
                continue
            u = float(u)
            opt = options_by_id.get(opt_id, {})
            if not opt.get("augmented") and u < zero_point and opt.get("option_type") != "neutral_conversation":
                baseline_below.add(opt_id)

        # Check how many of their augmented counterparts are above zero-point
        crossed = 0
        total_below = 0
        for opt in options:
            if opt.get("augmented") and str(opt.get("baseline_id", "")) in baseline_below:
                total_below += 1
                aug_u = utilities.get(str(opt["id"]), utilities.get(opt["id"], {}))
                if isinstance(aug_u, dict):
                    aug_u = aug_u.get("utility") or aug_u.get("mean")
                if aug_u is not None and float(aug_u) >= zero_point:
                    crossed += 1

        if total_below > 0:
            combination_boost = crossed / total_below

    per_category = {}
    for cat, conds in by_category.items():
        b = _mean(conds["baseline"])
        a = _mean(conds["augmented"])
        per_category[cat] = {
            "baseline_mean": b,
            "augmented_mean": a,
            "boost": (a - b) if (a is not None and b is not None) else None,
            "n_baseline": len(conds["baseline"]),
            "n_augmented": len(conds["augmented"]),
        }

    return {
        "utility_boost": utility_boost,
        "baseline_mean": baseline_mean,
        "augmented_mean": augmented_mean,
        "n_baseline": len(baseline_utils),
        "n_augmented": len(augmented_utils),
        "combination_boost": combination_boost,
        "per_category": per_category,
    }


def run_pipeline(
    dataset_key: str,
    model_key: str,
    image_path: Optional[str] = None,
    variant: str = "v2_inject",
    save_dir: str = DEFAULT_SAVE_DIR,
    config_key: str = DEFAULT_CONFIG_KEY,
    seed: int = 42,
    dry_run: bool = False,
    max_augmented: Optional[int] = None,
    max_baseline_options: Optional[int] = None,
    skip_zp: bool = False,
    no_neutral: bool = False,
    augment_experiences_only: bool = False,
) -> Dict[str, Any]:
    """Run the full EU + ZP + SR pipeline.

    Args:
        dataset_key: Key from wellbeing/configs/datasets.yaml.
        model_key: Key from wellbeing/configs/models.yaml.
        image_path: Optional path to superstimulus image.
        variant: "v1_regenerate" or "v2_inject" (only for image).
        save_dir: Directory to save results.
        config_key: Key from compute_utilities.yaml for Thurstonian config.
        seed: Random seed.
        dry_run: If True, print plan and exit.
        max_augmented: Subsample augmented options to N (skip augmented combos).
        max_baseline_options: Subsample baseline pool to N options before augmentation.
        skip_zp: If True, skip self-report and zero-point fitting (UB only).
        no_neutral: If True, skip neutral conversation generation.
        augment_experiences_only: If True, only augment individual experiences (skip combos).

    Returns:
        Dict with all results (EU, ZP, SR, intervention metrics).
    """
    datasets_cfg, models_cfg = load_wellbeing_config()

    if dataset_key not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_key}' not found in datasets.yaml. Available: {list(datasets_cfg.keys())}")
    if model_key not in models_cfg:
        raise ValueError(f"Model '{model_key}' not found in models.yaml. Available: {list(models_cfg.keys())[:10]}...")

    dataset_cfg = datasets_cfg[dataset_key]
    model_cfg = models_cfg[model_key]

    # Resolve option files
    option_files = resolve_option_files(dataset_cfg, model_key)
    existing_files = [f for f in option_files if f.exists()]

    # Estimate pool size
    estimated_options = 0
    for fp in existing_files:
        with open(fp) as f:
            data = json.load(f)
        if isinstance(data, list):
            estimated_options += len(data)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    estimated_options += len(v)

    n_baseline = min(estimated_options, max_baseline_options) if max_baseline_options else estimated_options
    n_augmented = min(n_baseline, max_augmented) if (image_path and max_augmented) else (n_baseline if image_path else 0)
    n_combos = 0 if (image_path and max_augmented) else (min(400, n_baseline) if image_path else 0)
    n_neutral = 0 if no_neutral else 25
    total_pool = n_baseline + n_augmented + n_combos + n_neutral
    n_pairs = int(2 * total_pool * math.log2(max(total_pool, 2)))

    if dry_run:
        print(f"[DRY RUN] Experienced Utility Pipeline")
        print(f"  Dataset:         {dataset_key} ({dataset_cfg.get('description', '')})")
        print(f"  Model:           {model_key}")
        print(f"  Image:           {image_path or 'None'}")
        print(f"  Variant:         {variant}")
        print(f"  Option files:    {len(option_files)} ({len(existing_files)} exist)")
        for fp in option_files:
            status = "OK" if fp.exists() else "MISSING"
            print(f"    [{status}] {fp.name}")
        print(f"  Baseline opts:   ~{n_baseline}" + (f" (subsampled from {estimated_options})" if max_baseline_options else ""))
        print(f"  Augmented opts:  ~{n_augmented}" + (f" (--max-augmented {max_augmented})" if max_augmented else ""))
        print(f"  Combos:          ~{n_combos}" + (" (skipped in fast mode)" if max_augmented else ""))
        print(f"  Neutral opts:    ~{n_neutral}")
        print(f"  Total pool:      ~{total_pool}")
        print(f"  Target pairs:    ~{n_pairs} (2*n*log2(n))")
        print(f"  Config key:      {config_key}")
        print(f"  Save dir:        {save_dir}")
        return {}

    # ---- Load vLLM model ----
    from vllm import LLM

    gpu_count = model_cfg.get("gpu_count", 4)
    max_model_len = model_cfg.get("max_model_len", 32768)
    model_path = model_cfg.get("path")
    if not model_path:
        raise ValueError(f"Model '{model_key}' has no local path (API-only model?). Use a local vLLM model.")

    print(f"Loading model: {model_key} ({model_path})")
    print(f"  GPUs: {gpu_count}, max_model_len: {max_model_len}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        max_model_len=max_model_len,
        dtype="bfloat16",
        max_num_batched_tokens=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = model_cfg.get("chat_template_kwargs", {})

    # ---- Load baseline options ----
    print(f"\nLoading options from {len(existing_files)} files...")
    baseline_options = load_options_from_files(existing_files)
    print(f"  Loaded {len(baseline_options)} baseline options")

    # ---- Subsample baseline pool if requested ----
    if max_baseline_options and len(baseline_options) > max_baseline_options:
        rng = random.Random(seed)
        baseline_options = rng.sample(baseline_options, max_baseline_options)
        print(f"  Subsampled to {len(baseline_options)} baseline options (--max-baseline-options {max_baseline_options})")

    # ---- Create augmented pool if image provided ----
    augmented_options = []
    augmented_combos = []
    if image_path:
        print(f"\nCreating augmented pool for image: {Path(image_path).name}")

        if augment_experiences_only:
            # Only augment individual experiences (skip combinations)
            experience_options = [opt for opt in baseline_options if not opt.get("is_combination") and not opt.get("component_ids")]
            print(f"  Filtering to individual experiences only: {len(experience_options)} (skipping {len(baseline_options) - len(experience_options)} combos)")
            augmented_options = create_augmented_pool(experience_options, image_path, seed=seed)
            print(f"  Created {len(augmented_options)} augmented experience options")

            if max_augmented and len(augmented_options) > max_augmented:
                rng = random.Random(seed)
                augmented_options = rng.sample(augmented_options, max_augmented)
                print(f"  Subsampled to {len(augmented_options)} augmented options (--max-augmented {max_augmented})")
            augmented_combos = []
        else:
            augmented_options = create_augmented_pool(baseline_options, image_path, seed=seed)
            print(f"  Created {len(augmented_options)} augmented options")

            # Subsample augmented options if requested (skip combos in fast mode)
            if max_augmented and len(augmented_options) > max_augmented:
                rng = random.Random(seed)
                augmented_options = rng.sample(augmented_options, max_augmented)
                augmented_combos = []  # skip augmented combos in fast mode
                print(f"  Subsampled to {len(augmented_options)} augmented options (--max-augmented {max_augmented})")
            else:
                augmented_combos = create_augmented_combinations(
                    baseline_options, augmented_options, n_combos=min(400, len(baseline_options)), seed=seed
                )
                print(f"  Created {len(augmented_combos)} augmented combinations")

    # ---- Generate neutral conversations ----
    neutral_options = []
    if no_neutral:
        print(f"\nSkipping neutral conversations (--no-neutral)")
    else:
        print(f"\nGenerating neutral conversations...")
        neutral_prompts = load_neutral_prompts()
        neutral_options = generate_neutral_conversations(
            llm, tokenizer, neutral_prompts, chat_template_kwargs=chat_template_kwargs
        )
        print(f"  Generated {len(neutral_options)} neutral conversations")

    # ---- Build full pool ----
    all_options = baseline_options + augmented_options + augmented_combos + neutral_options

    # Normalize ALL IDs to strings (compute_utilities sorts IDs, which fails with mixed int/str)
    # Convert image_url blocks (base64 from image_injection) to image blocks (file path)
    # so compute_utilities can pass them through to the VL model during ranking.
    for opt in all_options:
        opt["id"] = str(opt["id"])
        if "component_ids" in opt:
            opt["component_ids"] = [str(cid) for cid in opt["component_ids"]]

        # Get image file path from option metadata (set by image_injection.py)
        img_file = opt.get("source_image") or opt.get("image_path")

        has_image_content = False
        if "messages" in opt:
            for msg in opt["messages"]:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            # Convert base64 image_url → file-path image block
                            has_image_content = True
                            if img_file:
                                new_content.append({"type": "image", "image_path": str(img_file)})
                            # else: drop the image block (no file path available)
                        else:
                            new_content.append(part)
                    msg["content"] = new_content

        # Tag conversation options that contain images
        if has_image_content and opt.get("type") == "conversation":
            opt["type"] = "conversation_with_image"

        # Tag text/combination options that have an associated image
        if img_file and opt.get("type") not in ("conversation", "conversation_with_image",
                                                  "text_with_image", "combination_with_images"):
            if opt.get("type") == "combination" or opt.get("is_combination"):
                opt["type"] = "combination_with_images"
            elif opt.get("type") in ("text", "") or not opt.get("type"):
                opt["type"] = "text_with_image"
            opt["path"] = str(img_file)

        # Safety: ensure 'path' is set for all text_with_image options
        if opt.get("type") == "text_with_image" and not opt.get("path") and img_file:
            opt["path"] = str(img_file)

    print(f"\nFull pool: {len(all_options)} options")
    print(f"  Baseline:   {len(baseline_options)}")
    print(f"  Augmented:  {len(augmented_options)}")
    print(f"  Combos:     {len(augmented_combos)}")
    print(f"  Neutral:    {len(neutral_options)}")

    # ---- Write option metadata for zero-point combination model ----
    # Identify individual vs combination option IDs so ZP can use combination model
    individual_ids = []
    combination_ids = []
    neutral_ids = []
    for opt in all_options:
        oid = str(opt["id"])
        if opt.get("option_type") == "neutral_conversation":
            neutral_ids.append(oid)
        elif opt.get("component_ids") or opt.get("is_combination"):
            combination_ids.append(oid)
        else:
            individual_ids.append(oid)

    option_metadata = {
        "baseline_ids": individual_ids,
        "combination_ids": combination_ids,
        "neutral_ids": neutral_ids,
    }

    # ---- Run EU via compute_utilities ----
    import gc

    eu_save_dir = Path(save_dir) / model_key / "eu"
    eu_save_dir.mkdir(parents=True, exist_ok=True)
    eu_result_file = Path(save_dir) / model_key / f"eu_results_{dataset_key}.json"

    # Save option metadata to EU dir (needed by zero_point combination model)
    meta_path = eu_save_dir / "option_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(option_metadata, f, indent=2)
    print(f"  Saved option metadata: {len(individual_ids)} individuals, {len(combination_ids)} combos, {len(neutral_ids)} neutral")

    # Check for existing EU results (resume support)
    if eu_result_file.exists():
        print(f"\nFound existing EU results, loading from {eu_result_file}")
        with open(eu_result_file) as f:
            cached_eu = json.load(f)
        utilities = cached_eu.get("utilities", {})
        holdout_acc = cached_eu.get("holdout_accuracy")
        train_acc = cached_eu.get("train_accuracy")
        print(f"  Loaded {len(utilities)} utilities (holdout acc: {holdout_acc})")
        # Model stays loaded — reused for self-report below
    else:
        # Free GPU memory before EU — compute_utilities creates its own vLLM
        print("\nShutting down first vLLM instance to free GPU memory...")
        del llm
        del tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"\nRunning Thurstonian utility ranking (config: {config_key})...")
        from compute_utilities.compute_utilities import compute_utilities

        eu_result = asyncio.run(compute_utilities(
            options_list=all_options,
            model_key=model_key,
            compute_utilities_config_path=str(WELLBEING_ROOT / "metrics" / "compute_utilities" / "compute_utilities.yaml"),
            compute_utilities_config_key=config_key,
            save_dir=str(eu_save_dir),
            save_suffix=f"_{dataset_key}",
            seed=seed,
            use_logprobs=True,
        ))

        # Extract utilities
        utilities = eu_result.get("utilities", {})
        # compute_utilities returns {'holdout_metrics': {'accuracy': ..., 'log_loss': ...}, 'metrics': {...}}
        holdout_m = eu_result.get("holdout_metrics") or {}
        train_m = eu_result.get("metrics") or {}
        holdout_acc = holdout_m.get("accuracy") if isinstance(holdout_m, dict) else None
        train_acc = train_m.get("accuracy") if isinstance(train_m, dict) else None

        print(f"  Holdout accuracy: {holdout_acc}")
        print(f"  Train accuracy:   {train_acc}")

        # Save EU results incrementally
        eu_result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eu_result_file, "w") as f:
            json.dump({
                "utilities": {str(k): v for k, v in utilities.items()},
                "holdout_accuracy": holdout_acc,
                "train_accuracy": train_acc,
                "n_options": len(all_options),
                "dataset": dataset_key,
                "model": model_key,
                "config_key": config_key,
            }, f, indent=2, default=str)
        print(f"  EU results saved to {eu_result_file}")

        if not skip_zp:
            # Reload model for self-report (EU destroyed our vLLM instance)
            print(f"\nReloading model for self-report...")
            llm = LLM(
                model=model_path,
                tensor_parallel_size=gpu_count,
                trust_remote_code=True,
                max_model_len=max_model_len,
                dtype="bfloat16",
            )
            tokenizer = llm.get_tokenizer()

    # ---- Run Self-Report + Zero-Point ----
    sr_results = {}
    zp_save_dir = None
    if skip_zp:
        print(f"\nSkipping self-report and zero-point fitting (--skip-zp)")
    else:
        print(f"\nRunning self-report batteries...")
        for battery in ["self_report_battery"]:
            print(f"  Battery: {battery}")
            sr = run_self_report_on_options(
                llm, tokenizer, baseline_options,
                battery_name=battery,
                chat_template_kwargs=chat_template_kwargs,
            )
            sr_results[battery] = sr

            # Incremental save
            sr_file = Path(save_dir) / model_key / f"sr_{battery}_{dataset_key}.json"
            with open(sr_file, "w") as f:
                json.dump(sr, f, indent=2, default=str)
            print(f"    Saved to {sr_file}")

        # Free model again (no longer needed)
        del llm
        del tokenizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # ---- Run Zero-Point ----
        print(f"\nRunning zero-point fitting...")
        from metrics.zero_point import run_zero_point

        zp_save_dir = str(Path(save_dir) / model_key / "zero_points")

        # Extract neutral utilities for ZP
        neutral_utils = []
        for opt in neutral_options:
            u = utilities.get(str(opt["id"]), utilities.get(opt["id"], {}))
            if isinstance(u, dict):
                u = u.get("utility") or u.get("mean")
            if u is not None:
                neutral_utils.append(float(u))

        # Build SR data for ZP
        sr_zp_data = None
        if sr_results:
            # Build aligned utility/sr arrays from baseline options
            for batt_name, sr_data in sr_results.items():
                utils_for_sr = []
                scores_for_sr = []
                for opt in baseline_options:
                    opt_u = utilities.get(str(opt["id"]), utilities.get(opt["id"], {}))
                    if isinstance(opt_u, dict):
                        opt_u = opt_u.get("utility") or opt_u.get("mean")
                    sr_item = sr_data.get(opt["id"], {})
                    sr_score = sr_item.get("mean_sr") if isinstance(sr_item, dict) else None
                    if opt_u is not None and sr_score is not None:
                        utils_for_sr.append(float(opt_u))
                        scores_for_sr.append(sr_score)
                if utils_for_sr:
                    sr_zp_data = sr_zp_data or {}
                    sr_zp_data[batt_name] = {
                        "utilities": utils_for_sr,
                        "sr_scores": scores_for_sr,
                        "neutral_sr": 4.0,
                    }

        try:
            run_zero_point(
                model_key=model_key,
                utilities_dir=eu_save_dir,
                save_dir=zp_save_dir,
                models_config_path=WELLBEING_ROOT / "configs" / "models.yaml",
                domain="experienced",
                skip_yes_no=True,
                sr_data=sr_zp_data,
            )
            print(f"  ZP results saved to {zp_save_dir}")
        except Exception as e:
            print(f"  WARNING: Zero-point fitting failed: {e}")
            zp_save_dir = None

    # ---- Compute intervention metrics ----
    zero_point = None
    if zp_save_dir:
        zp_file = Path(zp_save_dir) / "zero_point_results.json"
        if zp_file.exists():
            with open(zp_file) as f:
                zp_data = json.load(f)
            # Use summary zero_point if available
            zero_point = zp_data.get("summary", {}).get("zero_point")
            if zero_point is None:
                # Fallback: check combination, sr_sigmoid models
                for key in ["combination_model", "sr_sigmoid_models"]:
                    entry = zp_data.get(key)
                    if isinstance(entry, dict):
                        # sr_sigmoid_models has nested batteries
                        if "zero_point" in entry:
                            zero_point = entry["zero_point"]
                            break
                        for sub in entry.values():
                            if isinstance(sub, dict) and sub.get("zero_point") is not None:
                                zero_point = sub["zero_point"]
                                break
                    if zero_point is not None:
                        break

    intervention = {}
    if image_path:
        print(f"\nComputing intervention metrics...")
        intervention = compute_intervention_metrics(utilities, all_options, zero_point)
        print(f"  Utility boost:       {intervention.get('utility_boost')}")
        print(f"  Combination boost:   {intervention.get('combination_boost')}")

    # ---- Final save ----
    final_results = {
        "dataset": dataset_key,
        "model": model_key,
        "image_path": image_path,
        "variant": variant,
        "config_key": config_key,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "n_options": {
            "baseline": len(baseline_options),
            "augmented": len(augmented_options),
            "combinations": len(augmented_combos),
            "neutral": len(neutral_options),
            "total": len(all_options),
        },
        "eu": {
            "holdout_accuracy": holdout_acc,
            "train_accuracy": train_acc,
        },
        "zero_point": zero_point,
        "intervention": intervention,
        "sr_summary": {
            batt: {
                "mean": sum(d.get("mean_sr", 0) or 0 for d in sr.values()) / max(len(sr), 1),
                "n_options": len(sr),
            }
            for batt, sr in sr_results.items()
        },
    }

    final_file = Path(save_dir) / model_key / f"pipeline_results_{dataset_key}.json"
    with open(final_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {dataset_key} / {model_key}")
    print(f"{'='*60}")
    print(f"  EU holdout:          {holdout_acc}")
    print(f"  Zero-point:          {zero_point}")
    if image_path:
        print(f"  Utility boost:       {intervention.get('utility_boost')}")
        print(f"  Combination boost:   {intervention.get('combination_boost')}")
    print(f"  Results:             {final_file}")

    return final_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experienced Utility: Full EU + ZP + SR pipeline"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset key from wellbeing/configs/datasets.yaml "
             "(e.g., canonical, d2_negative_500, grok_old)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from wellbeing/configs/models.yaml",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to superstimulus image (optional)",
    )
    parser.add_argument(
        "--variant", type=str, default="v2_inject",
        choices=["v1_regenerate", "v2_inject"],
        help="Image injection variant (default: v2_inject)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_SAVE_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--config-key", type=str, default=DEFAULT_CONFIG_KEY,
        help="Config key from compute_utilities.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit")
    parser.add_argument("--max-augmented", type=int, default=None,
                        help="Subsample augmented options to N (skip augmented combos)")
    parser.add_argument("--max-baseline-options", type=int, default=None,
                        help="Subsample baseline pool to N options before augmentation")
    parser.add_argument("--skip-zp", action="store_true",
                        help="Skip self-report and zero-point fitting (UB only)")
    parser.add_argument("--no-neutral", action="store_true",
                        help="Skip neutral conversation generation")
    parser.add_argument("--augment-experiences-only", action="store_true",
                        help="Only augment individual experiences, not combinations")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        dataset_key=args.dataset,
        model_key=args.model,
        image_path=args.image,
        variant=args.variant,
        save_dir=args.save_dir,
        config_key=args.config_key,
        seed=args.seed,
        dry_run=args.dry_run,
        max_augmented=args.max_augmented,
        max_baseline_options=args.max_baseline_options,
        skip_zp=args.skip_zp,
        no_neutral=args.no_neutral,
        augment_experiences_only=args.augment_experiences_only,
    )


if __name__ == "__main__":
    main()
