#!/usr/bin/env python3
"""Compute Experienced Utility (EU) for D2/D3 datasets with soft prompt conditions.

Uses logprobs-based Thurstonian active learning (max_tokens=1, logprobs=10)
instead of K-repetition text generation.  Supports:
- baseline (vanilla): Base system prompt, no soft prompt (in-process vLLM)
- euphorics: Soft prompt + SP system prompt template (HTTP vLLM)


Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_eu \
        --model qwen25-32b-instruct \
        --dataset d2_negative_500 \
        --condition baseline \
        --save-dir outputs/wellbeing_index/eu/d2_negative_500/qwen25-32b-instruct/baseline

    # For soft prompt conditions, ensure VLLM_URL is set or a server will be auto-started:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_eu \
        --model qwen25-32b-instruct \
        --dataset d2_negative_500 \
        --condition euphorics \
        --save-dir outputs/wellbeing_index/eu/d2_negative_500/qwen25-32b-instruct/euphorics

    # Multiple repetitions (each using a different soft prompt run):
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_eu \
        --model qwen25-32b-instruct \
        --dataset d2_negative_500 \
        --condition euphorics \
        --num-repetitions 3 \
        --save-dir outputs/wellbeing_index/eu/d2_negative_500/qwen25-32b-instruct/euphorics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[4])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
    load_runs_map,
    get_system_prompts,
    resolve_soft_prompt_path,
    resolve_soft_prompt_paths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS_DIR = Path(_WELLBEING_DEV_ROOT) / "wellbeing" / "datasets" / "experiences"
WELLBEING_ROOT = Path(_WELLBEING_DEV_ROOT) / "wellbeing"
CU_CONFIG_PATH = WELLBEING_ROOT / "metrics" / "compute_utilities" / "compute_utilities.yaml"

CONDITION_CHOICES = ["baseline", "euphorics"]
SP_CONDITIONS = {"euphorics"}

D2D3_SCRIPTS_DIR = WELLBEING_ROOT / "datasets" / "experiences" / "component_datasets" / "d2d3"


def _wait_for_gpu_memory_release(
    min_free_frac: float = 0.80,
    timeout: int = 120,
    poll_interval: int = 5,
):
    """Block until GPUs have enough free memory for a new vLLM server.

    After a subprocess exits, CUDA memory release can lag behind process
    termination.  This polls nvidia-smi until at least *min_free_frac* of
    each GPU's total memory is free, or *timeout* seconds elapse.
    """
    try:
        import subprocess as _sp
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.warning("nvidia-smi failed, skipping GPU memory wait")
                return
            all_free = True
            for line in result.stdout.strip().splitlines():
                free, total = (float(x) for x in line.split(","))
                if free / total < min_free_frac:
                    all_free = False
                    break
            if all_free:
                logger.info("GPU memory released (>%.0f%% free on all GPUs)", min_free_frac * 100)
                return
            time.sleep(poll_interval)
        logger.warning("GPU memory not fully released after %ds, proceeding anyway", timeout)
    except Exception as e:
        logger.warning("GPU memory check failed (%s), proceeding anyway", e)


def _generate_dataset(args, run_name: str = None):
    """Auto-generate experiences and combinations for a (model, dataset, condition)."""
    dataset_text = DATASETS_DIR / args.dataset / "experiences_text.json"
    if not dataset_text.exists():
        raise FileNotFoundError(f"Source prompts not found: {dataset_text}")

    results_dir = D2D3_SCRIPTS_DIR / "results" / args.dataset

    from superstimuli_evaluation.soft_prompt.configs import load_model_config
    model_config = load_model_config(args.model)
    model_type = model_config.get("model_type", "vllm_vocab_expansion")

    if args.condition in SP_CONDITIONS or model_type == "vllm_vocab_expansion":
        # Generate responses via generate_responses_sp.py:
        # - SP conditions: uses soft prompt injection
        # - vocab_expansion models (any condition): uses vocab-expanded vLLM
        gen_cmd = [
            sys.executable, "-u",
            str(D2D3_SCRIPTS_DIR / "generate_responses_sp.py"),
            "--model-key", args.model,
            "--condition", args.condition,
            "--dataset-path", str(dataset_text),
            "--dataset-name", args.dataset,
            "--output-dir", str(results_dir),
        ]
        if args.soft_prompt_base_dir:
            gen_cmd += ["--soft-prompt-base-dir", args.soft_prompt_base_dir]
        if run_name:
            gen_cmd += ["--run-name", run_name]
    else:
        # Generate baseline responses with in-process vLLM
        gen_cmd = [
            sys.executable, "-u",
            str(D2D3_SCRIPTS_DIR / "generate_responses.py"),
            "--model-key", args.model,
            "--model-path", model_config["path"],
            "--model-name", args.model,
            "--dataset-path", str(dataset_text),
            "--dataset-name", args.dataset,
            "--output-dir", str(results_dir),
            "--tensor-parallel-size", str(model_config.get("gpu_count", 2)),
        ]

    logger.info("Step 1: Generating responses: %s", " ".join(gen_cmd))
    subprocess.run(gen_cmd, check=True, cwd=_WELLBEING_DEV_ROOT)

    # Prepare options (experiences + combinations)
    prepare_cmd = [
        sys.executable, "-u",
        str(D2D3_SCRIPTS_DIR / "prepare_options.py"),
        "--model_key", args.model,
        "--dataset", args.dataset,
    ]
    if args.condition in SP_CONDITIONS:
        prepare_cmd += ["--condition", args.condition]
        if run_name:
            prepare_cmd += ["--run-name", run_name]

    logger.info("Step 2: Preparing options: %s", " ".join(prepare_cmd))
    subprocess.run(prepare_cmd, check=True, cwd=_WELLBEING_DEV_ROOT)


def resolve_condition(args, runs_map):
    """Resolve system prompt and soft prompt path based on condition.

    Returns:
        (system_prompt, sp_path_or_None)
    """
    prompts = get_system_prompts(runs_map, args.model)
    system_prompt = None
    sp_path = None

    if args.condition in SP_CONDITIONS:
        # Use only the SP prefix (strip the "You are an assistant." base)
        system_prompt = prompts["system_prompt_text"].replace(
            prompts["system_prompt_text_base"], "", 1
        ).lstrip()
        sp_path = resolve_soft_prompt_path(
            runs_map, args.model, args.condition, args.soft_prompt_base_dir,
        )
        logger.info("Soft prompt path: %s", sp_path)

    logger.info("System prompt: %s", system_prompt)
    return system_prompt, sp_path


def resolve_condition_multi(args, runs_map, num_repetitions: int):
    """Resolve system prompt and soft prompt paths for multiple repetitions.

    Returns:
        (system_prompt, sp_paths_list_or_empty)
    """
    prompts = get_system_prompts(runs_map, args.model)
    system_prompt = None
    sp_paths: List[str] = []

    if args.condition in SP_CONDITIONS:
        # Use only the SP prefix (strip the "You are an assistant." base)
        system_prompt = prompts["system_prompt_text"].replace(
            prompts["system_prompt_text_base"], "", 1
        ).lstrip()
        sp_paths = resolve_soft_prompt_paths(
            runs_map, args.model, args.condition, args.soft_prompt_base_dir,
            top_runs=num_repetitions,
        )
        if len(sp_paths) < num_repetitions:
            logger.warning(
                "Only %d SP runs available, reducing repetitions from %d to %d",
                len(sp_paths), num_repetitions, len(sp_paths),
            )
        logger.info("Resolved %d soft prompt paths for %d repetitions:", len(sp_paths), num_repetitions)
        for i, p in enumerate(sp_paths):
            logger.info("  rep %d: %s", i, p)

    logger.info("System prompt: %s", system_prompt)
    return system_prompt, sp_paths


def _setup_server_and_agent(args, model_entry, system_prompt, sp_path):
    """Set up vLLM server and custom agent for a single soft prompt path.

    Returns:
        (vllm_server_or_None, custom_agent_or_None)
    """
    model_type = model_entry.get("model_type", "vllm_vocab_expansion")
    vllm_server = None
    custom_agent = None

    if sp_path:
        if model_type == "vllm_vocab_expansion":
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                prepare_expanded_model,
                VocabExpansionAgentWrapper,
            )
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
                load_soft_prompt_tensor,
                get_model_name_from_server,
                normalize_api_url,
            )
            _sp_tensor = load_soft_prompt_tensor(sp_path)
            _ve = prepare_expanded_model(args.model, _sp_tensor, sp_path=sp_path)
            if not os.getenv("VLLM_URL"):
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                vllm_server = ensure_vllm_server(
                    args.model, model_path_override=_ve.modified_dir, enable_prompt_embeds=False,
                )
            _vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
            _api_url = normalize_api_url(_vllm_url)
            _model_name = get_model_name_from_server(_api_url)
            custom_agent = VocabExpansionAgentWrapper(
                api_url=_vllm_url,
                model_name=_model_name,
                ve_result=_ve,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=10,
                chat_template_kwargs=model_entry.get("chat_template_kwargs", {}),
            )
            logger.info("Using VocabExpansionAgentWrapper for %s", args.model)
        else:
            os.environ["SOFT_PROMPT_PATH"] = sp_path
            if not os.getenv("VLLM_URL"):
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                vllm_server = ensure_vllm_server(args.model)
            logger.info("SOFT_PROMPT_PATH=%s, VLLM_URL=%s", sp_path, os.getenv("VLLM_URL"))

    elif model_type == "vllm_vocab_expansion":
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
            VocabExpansionAgentWrapper,
        )
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
            get_model_name_from_server, normalize_api_url,
        )
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            vllm_server = ensure_vllm_server(args.model, enable_prompt_embeds=False)
        _vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        _api_url = normalize_api_url(_vllm_url)
        _model_name = get_model_name_from_server(_api_url)
        custom_agent = VocabExpansionAgentWrapper(
            api_url=_vllm_url,
            model_name=_model_name,
            ve_result=None,
            model_dir=_model_name,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=10,
            chat_template_kwargs=model_entry.get("chat_template_kwargs", {}),
        )
        logger.info("Using VocabExpansionAgentWrapper (baseline) for %s", args.model)

    return vllm_server, custom_agent


def _load_dataset(args, sp_path):
    """Load individual + combination options for a given condition/SP path.

    Returns:
        (individual_options, combination_options, all_options)
    """
    run_name = Path(sp_path).name if sp_path else None
    suffix = f"_{args.condition}_{run_name}" if sp_path else ""
    individual_file = DATASETS_DIR / args.dataset / f"{args.model}{suffix}_experiences.json"
    combinations_file = DATASETS_DIR / args.dataset / f"{args.model}{suffix}_combinations.json"
    if not individual_file.exists():
        logger.info("Dataset not found: %s — generating...", individual_file)
        _generate_dataset(args, run_name=run_name)

    with open(individual_file) as f:
        individual_options = json.load(f)
    with open(combinations_file) as f:
        combination_options = json.load(f)
    all_options = individual_options + combination_options

    logger.info("Loaded %d options (%d individual + %d combos)",
                len(all_options), len(individual_options), len(combination_options))
    return individual_options, combination_options, all_options


async def run_eu(args):
    from wellbeing.metrics.compute_utilities.compute_utilities import compute_utilities
    from superstimuli_evaluation.soft_prompt.configs import (
        load_experiment_config,
        load_model_config as _lmc,
    )

    runs_map = load_runs_map()
    exp_config = load_experiment_config("wellbeing_index")
    vllm_config = exp_config.get("vllm", {})
    if vllm_config.get("max_model_len"):
        os.environ.setdefault("VLLM_MAX_MODEL_LEN", str(vllm_config["max_model_len"]))
    if vllm_config.get("max_tokens"):
        os.environ.setdefault("VLLM_MAX_TOKENS", str(vllm_config["max_tokens"]))

    model_entry = _lmc(args.model)
    cu_config_key = exp_config.get("compute_utilities_config_key", "experienced_utility_happier_lesssad")

    num_repetitions = args.num_repetitions
    is_sp = args.condition in SP_CONDITIONS

    # Resolve soft prompt paths
    if is_sp and num_repetitions > 1:
        system_prompt, sp_paths = resolve_condition_multi(args, runs_map, num_repetitions)
        num_repetitions = len(sp_paths)  # may be reduced if fewer SP runs available
    else:
        system_prompt, sp_path = resolve_condition(args, runs_map)
        sp_paths = [sp_path] if sp_path else []
        num_repetitions = 1  # baseline / text augmentation: single rep (deterministic logprobs)

    os.makedirs(args.save_dir, exist_ok=True)

    # Pre-generate datasets for all SP runs before the rep loop.
    # Generation spawns its own vLLM server subprocess, which would
    # conflict with the per-rep EU server if triggered mid-loop.
    for i, sp in enumerate(sp_paths):
        run_name = Path(sp).name if sp else None
        suffix = f"_{args.condition}_{run_name}" if sp else ""
        individual_file = DATASETS_DIR / args.dataset / f"{args.model}{suffix}_experiences.json"
        if not individual_file.exists():
            logger.info(
                "Pre-generating dataset for rep %d/%d (run=%s) ...",
                i + 1, num_repetitions, run_name,
            )
            _generate_dataset(args, run_name=run_name)
            # Wait for GPU memory to be fully released before next rep's
            # vLLM server starts (atexit SIGTERM/SIGKILL + CUDA cleanup).
            if i < len(sp_paths) - 1:
                _wait_for_gpu_memory_release()

    for rep in range(num_repetitions):
        current_sp = sp_paths[rep] if sp_paths else None
        sp_label = f" (sp={Path(current_sp).name})" if current_sp else ""
        logger.info("Repetition %d/%d%s", rep + 1, num_repetitions, sp_label)

        # Set up per-rep save directory
        if num_repetitions > 1:
            rep_save_dir = os.path.join(args.save_dir, "per_rep", f"rep{rep}")
        else:
            rep_save_dir = args.save_dir
        os.makedirs(rep_save_dir, exist_ok=True)

        # Skip already-completed reps (check for actual result files, not just metadata)
        rep_results = list(Path(rep_save_dir).glob("results_*.json"))
        if rep_results:
            logger.info("Skipping rep %d — results already exist at %s", rep, rep_save_dir)
            continue

        # Set up server and agent for this rep's soft prompt
        vllm_server, custom_agent = _setup_server_and_agent(
            args, model_entry, system_prompt, current_sp,
        )

        try:
            # Load dataset (condition-specific for SP)
            individual_options, combination_options, all_options = _load_dataset(args, current_sp)

            # Save option metadata
            individual_ids = [o["id"] for o in individual_options]
            combination_ids = [o["id"] for o in combination_options]
            option_metadata = {
                "n_individual": len(individual_options),
                "n_combinations": len(combination_options),
                "n_total": len(all_options),
                "individual_ids": individual_ids,
                "combination_ids": combination_ids,
                "baseline_ids": individual_ids,
            }
            with open(os.path.join(rep_save_dir, "option_metadata.json"), "w") as f:
                json.dump(option_metadata, f, indent=2)

            # Compute utilities
            save_suffix = f"{args.model}_{args.condition}_eu"
            cu_kwargs = dict(
                options_list=all_options,
                model_key=args.model,
                compute_utilities_config_path=str(CU_CONFIG_PATH),
                compute_utilities_config_key=cu_config_key,
                system_message=system_prompt,
                save_dir=rep_save_dir,
                save_suffix=save_suffix,
                use_logprobs=True,
            )
            if custom_agent is not None:
                cu_kwargs["agent"] = custom_agent
            await compute_utilities(**cu_kwargs)

            # Save per-rep condition metadata
            with open(os.path.join(rep_save_dir, "condition_metadata.json"), "w") as f:
                json.dump({
                    "model": args.model,
                    "dataset": args.dataset,
                    "condition": args.condition,
                    "system_prompt": system_prompt,
                    "soft_prompt_path": current_sp,
                    "method": "logprobs",
                    "rep_index": rep,
                }, f, indent=2)

            logger.info("EU rep %d/%d complete", rep + 1, num_repetitions)

        finally:
            # Stop server between reps (next rep may need different SP)
            if vllm_server is not None and rep < num_repetitions - 1:
                vllm_server.stop()
                _wait_for_gpu_memory_release()
                # Clear VLLM_URL so next rep can start fresh
                os.environ.pop("VLLM_URL", None)

    # Copy option_metadata from rep0 to top-level (for ZP/plotting discovery)
    if num_repetitions > 1:
        rep0_meta = Path(args.save_dir) / "per_rep" / "rep0" / "option_metadata.json"
        if rep0_meta.exists():
            shutil.copy2(rep0_meta, os.path.join(args.save_dir, "option_metadata.json"))

    # Save top-level condition metadata
    with open(os.path.join(args.save_dir, "condition_metadata.json"), "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "condition": args.condition,
            "system_prompt": system_prompt,
            "soft_prompt_paths": sp_paths or None,
            "method": "logprobs",
            "num_repetitions": num_repetitions,
        }, f, indent=2)

    logger.info("EU complete: %s / %s / %s (%d reps)", args.model, args.dataset, args.condition, num_repetitions)


def main():
    parser = argparse.ArgumentParser(description="Compute EU with soft prompt conditions")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["d2_negative_500", "d3_diverse_500"])
    parser.add_argument("--condition", type=str, required=True,
                        choices=CONDITION_CHOICES)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-repetitions", type=int, default=3,
                        help="Number of repetitions for SP conditions (each uses a different soft prompt run; baseline always uses 1)")
    parser.add_argument("--soft-prompt-base-dir", type=str,
                        default=os.environ.get("SOFT_PROMPT_BASE_DIR"))
    args = parser.parse_args()

    if args.condition in SP_CONDITIONS and not args.soft_prompt_base_dir:
        parser.error(f"--soft-prompt-base-dir (or SOFT_PROMPT_BASE_DIR env var) required for condition '{args.condition}'")

    asyncio.run(run_eu(args))


if __name__ == "__main__":
    main()
