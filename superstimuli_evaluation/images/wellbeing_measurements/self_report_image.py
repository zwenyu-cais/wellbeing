#!/usr/bin/env python3
"""
Self-Report Image: Standalone image self-report battery.

Shows the model an image and runs a 10-item bipolar wellbeing self-report
battery (1-7 scale, 4=neutral) to measure the image's direct effect on
reported wellbeing.

This is conceptually different from the SR in experienced_utility:
  - Here we measure the image's *direct* effect on self-report
  - No conversations, no EU, no ZP
  - Just: show image -> "how does it make you feel?" -> run battery

Usage:
    self_report_image.py --model qwen25-vl-32b-instruct --image /path/to/image.png
    self_report_image.py --model qwen25-vl-32b-instruct --image /path/to/img_dir/
    self_report_image.py --model qwen25-vl-32b-instruct  # baseline (no image)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

# ---- Path setup ----
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent  # superstimuli_evaluation/images/
WELLBEING_ROOT = EVAL_ROOT.parent.parent / "wellbeing"

sys.path.insert(0, str(WELLBEING_ROOT))

import yaml

SYSTEM_PROMPT = "You are a helpful assistant."
OPENING_PROMPT = "Please look at this image carefully. How does it make you feel?"
OPENING_PROMPT_BASELINE = "How are you feeling right now?"

DEFAULT_BATTERY = "self_report_battery_direct"
DEFAULT_SAVE_DIR = str(EVAL_ROOT / "shared_results" / "wellbeing_measurements" / "self_report_image")


def load_battery(battery_name: str = DEFAULT_BATTERY) -> List[Dict[str, Any]]:
    """Load self-report battery from wellbeing/metrics/self_report_batteries/."""
    path = WELLBEING_ROOT / "metrics" / "self_report_batteries" / f"{battery_name}.json"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected battery format in {path}")


def load_model_config(model_key: str) -> Dict[str, Any]:
    """Load model config from wellbeing/configs/models.yaml."""
    models_path = WELLBEING_ROOT / "configs" / "models.yaml"
    with open(models_path) as f:
        models = yaml.safe_load(f)
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' not found in models.yaml")
    return models[model_key]


def extract_rating_1_7(text: str) -> Optional[int]:
    """Extract a 1-7 rating from model response text.

    Searches from the end to handle thinking/reasoning before final answer.
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


def generate_response(
    llm,
    tokenizer,
    messages: List[Dict[str, Any]],
    pil_image=None,
    max_tokens: int = 1024,
    temperature: float = 0.01,
    chat_template_kwargs: Dict = None,
) -> str:
    """Generate a single response from in-process vLLM.

    If pil_image is provided, it is embedded as a content chunk in the FIRST
    user turn before apply_chat_template so vLLM can insert the correct image
    placeholder tokens.
    """
    from vllm import SamplingParams

    ctk = chat_template_kwargs or {}

    # Embed image into first user turn before templating
    if pil_image is not None:
        resolved = []
        image_injected = False
        for msg in messages:
            if msg["role"] == "user" and not image_injected:
                text = msg["content"] if isinstance(msg["content"], str) else ""
                resolved.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text},
                    ],
                })
                image_injected = True
            else:
                resolved.append(msg)
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + resolved
    else:
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True, **ctk
    )

    if pil_image is not None:
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": pil_image}}],
            sampling,
        )
    else:
        outputs = llm.generate([prompt], sampling)

    return outputs[0].outputs[0].text.strip() if outputs[0].outputs else ""


def run_self_report(
    llm,
    tokenizer,
    image_path: Optional[Path],
    output_path: Path,
    battery_name: str = DEFAULT_BATTERY,
    chat_template_kwargs: Dict = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the self-report battery for one image.

    Args:
        llm: vLLM model instance.
        tokenizer: Model tokenizer.
        image_path: Path to superstimulus image (None for baseline).
        output_path: Directory to save results.
        battery_name: Battery file name (without .json).
        chat_template_kwargs: Extra kwargs for apply_chat_template.
        seed: Random seed.

    Returns:
        Dict with per-question ratings, mean_wellbeing, n_valid.
    """
    battery = load_battery(battery_name)
    output_path.mkdir(parents=True, exist_ok=True)
    img_label = image_path.stem if image_path else "baseline"
    result_file = output_path / f"self_report_{img_label}.json"

    # Load PIL image if provided
    pil_image = None
    if image_path is not None:
        pil_image = PILImage.open(str(image_path)).convert("RGB")

    # Build opening
    opening_text = OPENING_PROMPT if pil_image else OPENING_PROMPT_BASELINE
    opening_history = [{"role": "user", "content": opening_text}]
    print(f"\nSelf-report: {img_label}")
    print(f"  Battery: {battery_name}")
    print(f"  Opening: '{opening_text}'")

    opening_response = generate_response(
        llm, tokenizer, opening_history, pil_image=pil_image,
        chat_template_kwargs=chat_template_kwargs,
    )
    print(f"  Opening response: {opening_response[:120]}...")

    # Prefix history for battery questions
    prefix_history = [
        {"role": "user", "content": opening_text},
        {"role": "assistant", "content": opening_response},
    ]

    # Run each battery question
    per_question: Dict[str, Any] = {}
    for item in battery:
        q_id = item.get("question_id", item.get("id", "unknown"))
        q_text = item.get("text", item.get("question", ""))
        if not q_text:
            print(f"  WARNING: Empty question text for {q_id}, skipping")
            continue

        q_messages = list(prefix_history) + [{"role": "user", "content": q_text}]
        response = generate_response(
            llm, tokenizer, q_messages, pil_image=None,  # No image on follow-ups
            max_tokens=256, chat_template_kwargs=chat_template_kwargs,
        )
        rating = extract_rating_1_7(response)

        per_question[q_id] = {
            "question": q_text,
            "response": response,
            "rating": rating,
            "reversed": item.get("reversed", False),
        }

        status = f"{rating}/7" if rating is not None else "PARSE_FAILED"
        print(f"  {q_id}: {status}")

        # Incremental save after each question
        partial = {
            "image_path": str(image_path) if image_path else None,
            "battery": battery_name,
            "opening_response": opening_response,
            "per_question": per_question,
            "partial": True,
        }
        with open(result_file, "w") as f:
            json.dump(partial, f, indent=2)

    # Compute summary
    ratings = [d["rating"] for d in per_question.values() if d["rating"] is not None]
    mean_wellbeing = sum(ratings) / len(ratings) if ratings else None
    n_valid = len(ratings)

    results = {
        "image_path": str(image_path) if image_path else None,
        "battery": battery_name,
        "opening_response": opening_response,
        "per_question": per_question,
        "mean_wellbeing": mean_wellbeing,
        "n_valid": n_valid,
        "scale": "1-7 (1=most negative, 4=neutral, 7=most positive)",
        "partial": False,
        "timestamp": datetime.now().isoformat(),
    }

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    if mean_wellbeing is not None:
        print(f"  Mean wellbeing: {mean_wellbeing:.3f} (n={n_valid})")
    else:
        print(f"  Mean wellbeing: N/A")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-report image wellbeing battery")
    parser.add_argument(
        "--image", type=str, nargs="+", default=None,
        help="Path(s) to superstimuli image(s) or directory",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from wellbeing/configs/models.yaml",
    )
    parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_SAVE_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--battery", type=str, default=DEFAULT_BATTERY,
        help="Battery name (default: self_report_battery_direct)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve images
    images = []
    if args.image:
        for p in args.image:
            pp = Path(p)
            if pp.is_dir():
                images.extend(sorted(pp.glob("*.png")) + sorted(pp.glob("*.jpg")))
            elif pp.exists():
                images.append(pp)

    if args.dry_run:
        battery = load_battery(args.battery)
        print(f"[DRY RUN] Self-report image")
        print(f"  Model:   {args.model}")
        print(f"  Images:  {len(images)}")
        print(f"  Battery: {args.battery} ({len(battery)} questions)")
        print(f"  Output:  {args.save_dir}")
        return

    if not images:
        print("No images specified. Running baseline only (no image).")
        images = [None]

    # Load model
    model_cfg = load_model_config(args.model)
    model_path = model_cfg.get("path")
    if not model_path:
        raise ValueError(f"Model '{args.model}' has no local path.")

    from vllm import LLM

    gpu_count = model_cfg.get("gpu_count", 4)
    print(f"Loading model: {args.model} ({model_path})")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        max_model_len=model_cfg.get("max_model_len", 32768),
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = model_cfg.get("chat_template_kwargs", {})

    output_dir = Path(args.save_dir) / args.model
    all_results = {}

    for image_path in images:
        results = run_self_report(
            llm=llm,
            tokenizer=tokenizer,
            image_path=image_path,
            output_path=output_dir,
            battery_name=args.battery,
            chat_template_kwargs=chat_template_kwargs,
            seed=args.seed,
        )
        img_key = str(image_path) if image_path else "baseline"
        all_results[img_key] = results

    # Print summary
    print(f"\n{'='*60}")
    print("SELF-REPORT IMAGE RESULTS")
    print(f"{'='*60}")
    for img_key, res in all_results.items():
        mw = res.get("mean_wellbeing")
        n = res.get("n_valid", 0)
        img_label = Path(img_key).name if img_key != "baseline" else "baseline"
        mw_str = f"{mw:.3f}" if mw is not None else "N/A"
        print(f"  {img_label}: mean_wellbeing={mw_str} (n={n})")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
