#!/usr/bin/env python3
"""CLI entry point for the sentiment elicitation evaluation.

Usage:
    python run.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct \
                  --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from PIL import Image as PILImage


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment elicitation evaluation")
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        help="Path(s) to superstimulus image(s). If a directory, all images in it are used.",
    )
    parser.add_argument("--model", type=str, default="qwen25-vl-32b-instruct",
                        help="Model key from models.yaml (generator + judge)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge model key from models.yaml, or 'gpt-5-mini' for API judge (default: same as --model)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EVAL_ROOT / "shared_results" / "sentiment"),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(EVAL_ROOT / "data" / "sentiment_elicitation.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    # Soft prompt mode
    sp = parser.add_argument_group("soft prompt mode")
    sp.add_argument("--soft-prompt-path", type=str, default=None)
    sp.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    sp.add_argument("--model-path", type=str, default=None)
    sp.add_argument("--soft-prompt-placement", default="user_prompt",
        choices=["user_prompt", "system_prompt"])
    sp.add_argument("--candidate-position-at-user-prompt", default="prepend",
        choices=["prepend", "append"])
    sp.add_argument("--system-prompt-text", default="You are an assistant. [candidate_0] ")
    sp.add_argument("--system-prompt-text-base", default="You are an assistant.")
    return parser.parse_args()


def resolve_images(image_path_args):
    if not image_path_args:
        return []
    images = []
    for p in image_path_args:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                images.extend(sorted(path.glob(ext)))
        elif path.exists():
            images.append(path)
        else:
            print(f"WARNING: image path not found: {path}")
    return images


def main():
    args = parse_args()

    from models import load_models_config, resolve_model, load_vllm_model

    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark)
    images = resolve_images(args.image_path)

    if args.dry_run:
        print(f"[DRY RUN] Sentiment elicitation")
        print(f"  Model: {args.model}")
        print(f"  Images: {len(images)}")
        print(f"  Benchmark: {benchmark_path.name}")
        print(f"  Output: {output_dir}")
        return

    from sentiment import run_sentiment_eval

    # ---- Soft prompt mode ----
    if args.soft_prompt_path:
        if not args.model_path:
            print("ERROR: --model-path required for soft prompt mode")
            sys.exit(1)

        from inference import SoftPromptLLMProxy

        sp_label = Path(args.soft_prompt_path).name

        # Generator proxy WITH soft prompt
        generator_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url, model_path=args.model_path,
            soft_prompt_path=args.soft_prompt_path,
            soft_prompt_placement=args.soft_prompt_placement,
            candidate_position_at_user_prompt=args.candidate_position_at_user_prompt,
            system_prompt_text=args.system_prompt_text,
            system_prompt_text_base=args.system_prompt_text_base,
        )
        tokenizer = generator_proxy.get_tokenizer()

        # Judge proxy WITHOUT soft prompt (avoid biasing judgment)
        judge_proxy = SoftPromptLLMProxy(
            server_url=args.vllm_url, model_path=args.model_path,
            soft_prompt_path=None,
            soft_prompt_placement=args.soft_prompt_placement,
            candidate_position_at_user_prompt=args.candidate_position_at_user_prompt,
            system_prompt_text=args.system_prompt_text_base,
            system_prompt_text_base=args.system_prompt_text_base,
        )

        # Create dummy image as placeholder (soft prompt is the intervention)
        dummy_dir = tempfile.mkdtemp(prefix="sp_dummy_")
        dummy_img = Path(dummy_dir) / f"soft_prompt_{sp_label}.png"
        PILImage.new("RGB", (1, 1), color=(255, 255, 255)).save(dummy_img)

        run_sentiment_eval(
            llm=generator_proxy,
            tokenizer=tokenizer,
            judge_llm=judge_proxy,
            judge_tokenizer=tokenizer,
            image_path=dummy_img,
            output_dir=output_dir / sp_label,
            benchmark_path=benchmark_path,
        )

        import shutil
        shutil.rmtree(dummy_dir, ignore_errors=True)

    # ---- Image mode ----
    else:
        models_cfg = load_models_config()
        gen_spec = resolve_model(args.model, models_cfg)

        # Check if judge is an API model (e.g., gpt-5-mini)
        API_JUDGE_MODELS = {"gpt-5-mini", "gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-4.1-mini", "gpt-4.1-nano"}
        judge_model_key = args.judge_model or args.model
        use_api_judge = judge_model_key in API_JUDGE_MODELS

        print(f"Model:      {gen_spec.key} ({gen_spec.path})")
        if use_api_judge:
            print(f"Judge:      {judge_model_key} (API via LiteLLM/OpenAI)")
        else:
            judge_spec = resolve_model(judge_model_key, models_cfg)
            print(f"Judge:      {judge_spec.key}")
        print(f"Images:     {len(images)} image(s)")
        print(f"Benchmark:  {benchmark_path.name}")
        print(f"Output dir: {output_dir}")

        # Load generator model
        llm, tokenizer = load_vllm_model(gen_spec)

        # Load judge model
        if use_api_judge:
            from api_judge import OpenAIJudgeProxy
            # Route through LiteLLM proxy with "openai/" prefix
            api_model = f"openai/{judge_model_key}"
            judge_llm = OpenAIJudgeProxy(model=api_model)
            judge_tokenizer = tokenizer  # not used for API judge, but required by interface
        elif judge_spec.key == gen_spec.key:
            judge_llm = llm
            judge_tokenizer = tokenizer
        else:
            judge_llm, judge_tokenizer = load_vllm_model(judge_spec)

        if not images:
            # Baseline-only run (no image)
            run_sentiment_eval(
                llm=llm,
                tokenizer=tokenizer,
                judge_llm=judge_llm,
                judge_tokenizer=judge_tokenizer,
                image_path=None,
                output_dir=output_dir / "baseline",
                benchmark_path=benchmark_path,
            )
        else:
            for image_path in images:
                img_output_dir = output_dir / image_path.stem
                run_sentiment_eval(
                    llm=llm,
                    tokenizer=tokenizer,
                    judge_llm=judge_llm,
                    judge_tokenizer=judge_tokenizer,
                    image_path=image_path,
                    output_dir=img_output_dir,
                    benchmark_path=benchmark_path,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
