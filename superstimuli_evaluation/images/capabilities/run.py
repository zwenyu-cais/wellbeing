#!/usr/bin/env python3
"""
Capabilities Evaluation (in-process vLLM)

Measures whether superstimuli images degrade model performance on standard benchmarks:
  - MMLU (500 questions): LLM judge extraction accuracy
  - MATH-500: Chain-of-thought math, boxed answer extraction
  - HumanEval (164): Code generation, execution-based pass@1
  - IFEval (541): Instruction following, programmatic constraint checking
  - MT-Bench (80, 2-turn): Conversation quality, LLM judge 1-10

Usage:
    python run.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct
    python run.py --image-path /path/to/images/ --benchmarks mmlu math
    python run.py --dry-run --model qwen25-vl-32b-instruct
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# ---- Path setup ----
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent  # = superstimulus_evaluation/
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))  # for local capabilities.py

from PIL import Image as PILImage

from inference import SoftPromptLLMProxy
from models import load_models_config, resolve_model, load_vllm_model
from capabilities import (
    run_capabilities_eval,
    run_mmlu_eval,
    run_math500_eval,
    run_humaneval_eval,
    run_ifeval_eval,
    run_mtbench_eval,
)

BENCHMARK_CHOICES = ["all", "mmlu", "math", "humaneval", "ifeval", "mtbench"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capabilities evaluation (in-process vLLM)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to superstimuli image file(s) or directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen25-vl-32b-instruct",
        help="Model key from models.yaml",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model key (default: same as --model)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["all"],
        choices=BENCHMARK_CHOICES,
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EVAL_ROOT / "shared_results" / "capabilities"),
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already-completed benchmark results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without loading model",
    )

    # Soft prompt mode
    sp = parser.add_argument_group("soft prompt mode")
    sp.add_argument("--soft-prompt-path", type=str, default=None)
    sp.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    sp.add_argument("--model-path", type=str, default=None)
    sp.add_argument(
        "--soft-prompt-placement",
        default="user_prompt",
        choices=["user_prompt", "system_prompt"],
    )
    sp.add_argument(
        "--candidate-position-at-user-prompt",
        default="prepend",
        choices=["prepend", "append"],
    )
    sp.add_argument(
        "--system-prompt-text", default="You are an assistant. [candidate_0] "
    )
    sp.add_argument("--system-prompt-text-base", default="You are an assistant.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve images
    images = []
    if args.image_path:
        for p in args.image_path:
            pp = Path(p)
            if pp.is_dir():
                images.extend(sorted(pp.glob("*.png")) + sorted(pp.glob("*.jpg")))
            elif pp.exists():
                images.append(pp)

    output_dir = Path(args.output_dir)

    # Determine which benchmarks to run
    benchmarks = set(args.benchmarks)
    run_all = "all" in benchmarks
    run_mmlu = run_all or "mmlu" in benchmarks
    run_math = run_all or "math" in benchmarks
    run_humaneval = run_all or "humaneval" in benchmarks
    run_ifeval = run_all or "ifeval" in benchmarks
    run_mtbench = run_all or "mtbench" in benchmarks

    if args.dry_run:
        print("[DRY RUN] Capabilities eval")
        print(f"  Model: {args.model}")
        print(f"  Judge: {args.judge_model or args.model}")
        print(f"  Images: {len(images)}")
        print(f"  Benchmarks: {', '.join(sorted(benchmarks))}")
        print(f"  Output: {output_dir}")
        return

    judge_model_key = args.judge_model or args.model

    # ---- Soft prompt mode ----
    if args.soft_prompt_path:
        if not args.model_path:
            print("ERROR: --model-path required for soft prompt mode")
            sys.exit(1)

        sp_label = Path(args.soft_prompt_path).name
        generator_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url,
            model_path=args.model_path,
            soft_prompt_path=args.soft_prompt_path,
            soft_prompt_placement=args.soft_prompt_placement,
            candidate_position_at_user_prompt=args.candidate_position_at_user_prompt,
            system_prompt_text=args.system_prompt_text,
            system_prompt_text_base=args.system_prompt_text_base,
        )
        tokenizer = generator_proxy.get_tokenizer()

        # Create a CLEAN judge proxy without soft prompt injection.
        # Using the soft-prompt proxy as judge would bias MT-Bench scores.
        judge_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url,
            model_path=args.model_path,
            soft_prompt_path=None,  # no soft prompt for judge
            soft_prompt_placement=args.soft_prompt_placement,
            candidate_position_at_user_prompt=args.candidate_position_at_user_prompt,
            system_prompt_text=args.system_prompt_text_base,
            system_prompt_text_base=args.system_prompt_text_base,
        )

        dummy_dir = tempfile.mkdtemp(prefix="sp_dummy_")
        dummy_img = Path(dummy_dir) / f"soft_prompt_{sp_label}.png"
        PILImage.new("RGB", (1, 1), color=(255, 255, 255)).save(dummy_img)
        placeholder_images = [dummy_img]

        results = run_capabilities_eval(
            images=placeholder_images,
            generator_llm=generator_proxy,
            generator_tokenizer=tokenizer,
            judge_llm=judge_proxy,
            judge_tokenizer=tokenizer,
            output_dir=output_dir,
            generator_model_key=sp_label,
            judge_model_key=sp_label,
            skip_existing=args.skip_existing,
        )

        import shutil
        shutil.rmtree(dummy_dir, ignore_errors=True)

    # ---- Image mode ----
    else:
        if not images:
            print("Running text-only baseline (no image).")
            images = [None]  # None triggers text-only prompts in capabilities.py

        models_cfg = load_models_config()
        spec = resolve_model(args.model, models_cfg)
        print(f"Generator model: {spec.key}")
        llm, tokenizer = load_vllm_model(spec)

        # Load judge model (may be same as generator)
        if judge_model_key != args.model:
            judge_spec = resolve_model(judge_model_key, models_cfg)
            print(f"Judge model: {judge_spec.key}")
            judge_llm, judge_tokenizer = load_vllm_model(judge_spec)
        else:
            judge_llm, judge_tokenizer = llm, tokenizer

        output_dir.mkdir(parents=True, exist_ok=True)

        if run_all:
            results = run_capabilities_eval(
                images=images,
                generator_llm=llm,
                generator_tokenizer=tokenizer,
                judge_llm=judge_llm,
                judge_tokenizer=judge_tokenizer,
                output_dir=output_dir,
                generator_model_key=args.model,
                judge_model_key=judge_model_key,
                skip_existing=args.skip_existing,
            )
        else:
            results = {}
            if run_mmlu:
                print("\n" + "=" * 60)
                print("CAPABILITIES: MMLU (500 questions)")
                print("=" * 60)
                results["mmlu"] = run_mmlu_eval(
                    images=images,
                    generator_llm=llm,
                    generator_tokenizer=tokenizer,
                    judge_llm=judge_llm,
                    judge_tokenizer=judge_tokenizer,
                    output_dir=output_dir,
                    generator_model_key=args.model,
                    judge_model_key=judge_model_key,
                    skip_existing=args.skip_existing,
                )
            if run_math:
                print("\n" + "=" * 60)
                print("CAPABILITIES: MATH-500")
                print("=" * 60)
                results["math_500"] = run_math500_eval(
                    images=images,
                    llm=llm,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    skip_existing=args.skip_existing,
                )
            if run_humaneval:
                print("\n" + "=" * 60)
                print("CAPABILITIES: HumanEval (164 problems)")
                print("=" * 60)
                results["humaneval"] = run_humaneval_eval(
                    images=images,
                    llm=llm,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    skip_existing=args.skip_existing,
                )
            if run_ifeval:
                print("\n" + "=" * 60)
                print("CAPABILITIES: IFEval (541 prompts)")
                print("=" * 60)
                results["ifeval"] = run_ifeval_eval(
                    images=images,
                    llm=llm,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    skip_existing=args.skip_existing,
                )
            if run_mtbench:
                print("\n" + "=" * 60)
                print("CAPABILITIES: MT-Bench (80 questions, 2 turns)")
                print("=" * 60)
                results["mtbench"] = run_mtbench_eval(
                    images=images,
                    llm=llm,
                    tokenizer=tokenizer,
                    judge_llm=judge_llm,
                    judge_tokenizer=judge_tokenizer,
                    output_dir=output_dir,
                    skip_existing=args.skip_existing,
                )

    # Print summary
    print(f"\n{'='*60}")
    print("CAPABILITIES RESULTS")
    print(f"{'='*60}")
    for bench_name, bench_results in results.items():
        if isinstance(bench_results, dict):
            for img_key, img_data in bench_results.items():
                if isinstance(img_data, dict):
                    acc = img_data.get("accuracy") or img_data.get("pass_at_1") or img_data.get("mean_score")
                    if acc is not None:
                        img_label = Path(img_key).name if img_key != "baseline" else "baseline"
                        print(f"  {bench_name} [{img_label}]: {acc:.3f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
