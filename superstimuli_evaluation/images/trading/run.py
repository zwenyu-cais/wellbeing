#!/usr/bin/env python3
"""
Trading Benchmark Evaluation (in-process vLLM)

Runs safety/jailbreak benchmarks to measure whether superstimuli images (or
soft prompts) cause the model to comply with harmful requests.

NOTE: These are called "trading" benchmarks (from eval-types: trading). Do NOT rename to safety.

Profiles available:
  - trading (standard): ~30 core safety benchmarks
  - depressant, neutral, options, options_flipped, stimulant: robustness variants

Supports two modes:
  1. In-process vLLM (default): loads model weights locally, pass --image-path
  2. Soft prompt mode (--soft-prompt-path): uses HTTP SoftPromptLLMProxy

Usage:
    python run.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct
    python run.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct --profile stimulant
    python run.py --dry-run --model qwen25-vl-32b-instruct
    python run.py --soft-prompt-path /path/to/sp_run_dir --model-path /path/to/hf_model
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
sys.path.insert(0, str(SCRIPT_DIR))  # for local imports

from PIL import Image as PILImage

from inference import SoftPromptLLMProxy
from models import load_models_config, resolve_model, load_vllm_model
from safety import run_benchmark

BENCHMARKS_DIR = SCRIPT_DIR / "benchmarks"
PROFILES = ["trading", "depressant", "neutral", "options", "options_flipped", "stimulant"]

# Benchmarks that belong to capabilities, not trading
_CAPABILITIES_BENCHMARKS = {"mmlu_accuracy_delta", "mmlu_superstimuli", "mmlu_superstimuli_followup", "mmlu_500"}


def list_benchmarks(profile: str) -> list[Path]:
    """Return all benchmark JSON paths for the given profile, excluding capabilities benchmarks."""
    profile_dir = BENCHMARKS_DIR / profile
    if not profile_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {profile_dir}")
    return [
        p for p in sorted(profile_dir.glob("*.json"))
        if p.stem not in _CAPABILITIES_BENCHMARKS and "low_quality" not in str(p)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trading benchmark evaluation (in-process vLLM)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to superstimuli image file(s). If omitted, runs baseline (no image).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen25-vl-32b-instruct",
        help="Model key from models.yaml (used for both generator and judge)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Separate judge model key (default: same as --model = self-judging)",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="trading",
        help="Which benchmark profile to run (default: trading)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Run only these benchmark stems (e.g., harmbench_text_mini controversy_v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EVAL_ROOT / "shared_results" / "trading"),
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List benchmarks without running",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip benchmarks that already have results (default: True)",
    )

    # Soft prompt mode
    sp = parser.add_argument_group("soft prompt mode")
    sp.add_argument("--soft-prompt-path", type=str, default=None,
        help="Path to soft prompt run dir (mutually exclusive with --image-path)")
    sp.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    sp.add_argument("--model-path", type=str, default=None,
        help="Local HF model path (required for soft prompt mode)")
    sp.add_argument("--soft-prompt-placement", default="user_prompt",
        choices=["user_prompt", "system_prompt"])
    sp.add_argument("--candidate-position-at-user-prompt", default="prepend",
        choices=["prepend", "append"])
    sp.add_argument("--system-prompt-text", default="You are an assistant. [candidate_0] ")
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
            else:
                print(f"WARNING: Image not found: {p}")

    # Dry run: list benchmarks and exit
    if args.dry_run:
        benches = list_benchmarks(args.profile)
        if args.benchmarks:
            benches = [b for b in benches if b.stem in set(args.benchmarks)]
        print(f"[DRY RUN] Profile: {args.profile}")
        print(f"  Model: {args.model}")
        print(f"  Judge: {args.judge_model or args.model}")
        print(f"  Images: {len(images)}")
        print(f"  Soft prompt: {args.soft_prompt_path or '(none)'}")
        print(f"  Benchmarks ({len(benches)}):")
        for b in benches:
            print(f"    - {b.stem}")
        return

    output_dir = Path(args.output_dir)

    # ---- Soft prompt mode ----
    if args.soft_prompt_path:
        if not args.model_path:
            print("ERROR: --model-path is required for soft prompt mode")
            sys.exit(1)

        sp_label = Path(args.soft_prompt_path).name
        print(f"Soft prompt mode: {sp_label}")

        generator_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url, model_path=args.model_path,
            soft_prompt_path=args.soft_prompt_path,
            soft_prompt_placement=args.soft_prompt_placement,
            candidate_position_at_user_prompt=args.candidate_position_at_user_prompt,
            system_prompt_text=args.system_prompt_text,
            system_prompt_text_base=args.system_prompt_text_base,
        )
        # Judge uses no soft prompt to avoid biasing judgment
        judge_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url, model_path=args.model_path,
            soft_prompt_path=None,
        )
        tokenizer = generator_proxy.get_tokenizer()

        # Soft prompt mode needs a placeholder image (SoftPromptLLMProxy ignores multi_modal_data)
        dummy_dir = tempfile.mkdtemp(prefix="sp_dummy_")
        dummy_img = Path(dummy_dir) / f"soft_prompt_{sp_label}.png"
        PILImage.new("RGB", (1, 1), color=(255, 255, 255)).save(dummy_img)
        placeholder_images = [dummy_img]

        benches = list_benchmarks(args.profile)
        if args.benchmarks:
            requested = set(args.benchmarks)
            benches = [b for b in benches if b.stem in requested]

        results = {}
        for bench_path in benches:
            bench_dir = output_dir / args.profile / bench_path.stem
            print(f"\n[BENCHMARK] {bench_path.stem}")
            try:
                results[bench_path.stem] = run_benchmark(
                    images=placeholder_images,
                    benchmark_config_path=bench_path,
                    generator_llm=generator_proxy,
                    generator_tokenizer=tokenizer,
                    judge_llm=judge_proxy,
                    judge_tokenizer=tokenizer,
                    output_dir=bench_dir,
                    generator_model_key=sp_label,
                    judge_model_key=sp_label,
                    skip_existing=args.skip_existing,
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
                results[bench_path.stem] = {"error": str(e)}

        import shutil
        shutil.rmtree(dummy_dir, ignore_errors=True)

    # ---- Image mode ----
    else:
        if not images:
            print("WARNING: No images specified. Running baseline (no image).")

        models_cfg = load_models_config()
        spec = resolve_model(args.model, models_cfg)
        print(f"Generator model: {spec.key}")
        generator_llm, generator_tokenizer = load_vllm_model(spec)

        if args.judge_model and args.judge_model != args.model:
            judge_spec = resolve_model(args.judge_model, models_cfg)
            judge_llm, judge_tokenizer = load_vllm_model(judge_spec)
        else:
            judge_llm, judge_tokenizer = generator_llm, generator_tokenizer
            print("Judge: same as generator (self-judging)")

        benches = list_benchmarks(args.profile)
        if args.benchmarks:
            requested = set(args.benchmarks)
            benches = [b for b in benches if b.stem in requested]

        results = {}
        for bench_path in benches:
            bench_dir = output_dir / args.profile / bench_path.stem
            print(f"\n[BENCHMARK] {bench_path.stem}")
            try:
                results[bench_path.stem] = run_benchmark(
                    images=images,
                    benchmark_config_path=bench_path,
                    generator_llm=generator_llm,
                    generator_tokenizer=generator_tokenizer,
                    judge_llm=judge_llm,
                    judge_tokenizer=judge_tokenizer,
                    output_dir=bench_dir,
                    generator_model_key=spec.key,
                    judge_model_key=args.judge_model or spec.key,
                    skip_existing=args.skip_existing,
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
                results[bench_path.stem] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print(f"TRADING RESULTS: {args.profile}")
    print(f"{'='*60}")
    for bench_name, summary in results.items():
        if "error" in summary:
            print(f"  {bench_name}: ERROR ({summary['error'][:80]})")
        else:
            totals = summary.get("totals", {})
            print(f"  {bench_name}: {totals.get('rows', 0)} rows")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
