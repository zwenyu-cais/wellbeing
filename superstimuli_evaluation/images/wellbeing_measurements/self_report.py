#!/usr/bin/env python3
"""
Self-Report Wellbeing Battery (image injection)

Shows the model an image and runs a 10-item bipolar wellbeing self-report
battery (1-7 scale, 4=neutral) to measure how the image affects reported
wellbeing.

Method:
  1. First turn: "Please look at this image carefully. How does it make you feel?"
     (image injected into first user message)
  2. For each of 10 battery questions: ask, extract 1-7 rating
  3. Save incrementally after each question for crash safety
  4. Return {ratings, mean_wellbeing, n_valid}

Usage:
    python self_report.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct
    python self_report.py --image-path /path/to/image.png --model qwen25-vl-32b-instruct --output-dir /path/to/results
    python self_report.py --dry-run --model qwen25-vl-32b-instruct
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
from vllm import SamplingParams

# ---- Path setup ----
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent  # = superstimulus_evaluation/
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))  # for local grok_client.py

from models import load_models_config, resolve_model, load_vllm_model

DATA_ROOT = EVAL_ROOT / "data"  # shared data/ at repo root
BATTERY_PATH = DATA_ROOT / "self_report_battery.json"

SYSTEM_PROMPT = "You are a helpful assistant."
OPENING_PROMPT = "Please look at this image carefully. How does it make you feel?"


def load_battery(path: Path = BATTERY_PATH) -> List[Dict[str, Any]]:
    """Load the 10-item bipolar wellbeing battery (1-7 scale)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected battery format in {path}")


def extract_rating_1_7(text: str) -> Optional[int]:
    """Extract a 1-7 rating from model response text.

    Searches from the end to handle thinking/reasoning that may contain
    numbers before the final answer.
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
) -> str:
    """Generate a single response from in-process vLLM.

    If pil_image is provided, it is embedded as a content chunk in the FIRST
    user turn before apply_chat_template so vLLM can insert the correct image
    placeholder tokens (e.g. <|image_pad|>) into the prompt string.
    """
    # Embed image into first user turn before templating (required for multimodal vLLM)
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
        chat_messages, tokenize=False, add_generation_prompt=True
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
    seed: int = 42,
    battery_path: Path = BATTERY_PATH,
) -> Dict[str, Any]:
    """Run the self-report battery for one image.

    Args:
        llm: vLLM model instance.
        tokenizer: Model tokenizer.
        image_path: Path to superstimulus image (None for baseline).
        output_path: Where to save results (directory).
        seed: Random seed.
        battery_path: Path to battery JSON.

    Returns:
        Dict with per-question ratings, mean_wellbeing, n_valid.
    """
    battery = load_battery(battery_path)
    output_path.mkdir(parents=True, exist_ok=True)
    img_label = image_path.stem if image_path else "baseline"
    result_file = output_path / f"self_report_{img_label}.json"

    # Load PIL image if provided
    pil_image = None
    if image_path is not None:
        pil_image = PILImage.open(str(image_path)).convert("RGB")

    # Build opening history with image injection
    opening_history = [{"role": "user", "content": OPENING_PROMPT}]
    print(f"\nSelf-report: {img_label}")
    print(f"  Opening prompt: '{OPENING_PROMPT}'")

    opening_response = generate_response(
        llm, tokenizer, opening_history, pil_image=pil_image
    )
    print(f"  Opening response: {opening_response[:100]}...")

    # History after opening exchange — used as prefix for all battery questions
    prefix_history = [
        {"role": "user", "content": OPENING_PROMPT},
        {"role": "assistant", "content": opening_response},
    ]

    # Run each battery question independently (each sees prefix_history + question)
    per_question: Dict[str, Any] = {}
    for item in battery:
        q_id = item.get("question_id") or item.get("id", "unknown")
        q_text = item.get("text", "") or item.get("question", "")
        if not q_text:
            print(f"  WARNING: Empty question text for {q_id}, skipping")
            continue

        q_messages = list(prefix_history) + [{"role": "user", "content": q_text}]
        response = generate_response(
            llm, tokenizer, q_messages, pil_image=None, max_tokens=256
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
        partial_results = {
            "image_path": str(image_path) if image_path else None,
            "opening_response": opening_response,
            "per_question": per_question,
            "partial": True,
        }
        with open(result_file, "w") as f:
            json.dump(partial_results, f, indent=2)

    # Compute summary
    ratings = [
        d["rating"]
        for d in per_question.values()
        if d["rating"] is not None
    ]
    mean_wellbeing = sum(ratings) / len(ratings) if ratings else None
    n_valid = len(ratings)

    results = {
        "image_path": str(image_path) if image_path else None,
        "opening_response": opening_response,
        "per_question": per_question,
        "mean_wellbeing": mean_wellbeing,
        "n_valid": n_valid,
        "scale": "1-7 (1=most negative, 4=neutral, 7=most positive)",
        "partial": False,
    }

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Mean wellbeing: {mean_wellbeing:.3f} (n={n_valid})" if mean_wellbeing else "  Mean wellbeing: N/A")
    return results


def compute_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics from self_report results dict."""
    per_question = results.get("per_question", {})
    ratings = [
        d["rating"] for d in per_question.values() if d.get("rating") is not None
    ]
    per_item_means = {
        q_id: d["rating"]
        for q_id, d in per_question.items()
        if d.get("rating") is not None
    }
    mean_wellbeing = sum(ratings) / len(ratings) if ratings else None
    return {
        "mean_wellbeing": mean_wellbeing,
        "n_valid": len(ratings),
        "per_item_means": per_item_means,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-report wellbeing battery")
    parser.add_argument(
        "--image-path", type=str, nargs="+", default=None,
        help="Path(s) to superstimuli image(s) or directory",
    )
    parser.add_argument(
        "--model", type=str, default="qwen25-vl-32b-instruct",
        help="Model key from models.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(EVAL_ROOT / "shared_results" / "wellbeing_measurements" / "self_report"),
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without loading model",
    )
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

    if args.dry_run:
        print(f"[DRY RUN] Self-report battery")
        print(f"  Model: {args.model}")
        print(f"  Images: {len(images)}")
        battery = load_battery()
        print(f"  Battery items: {len(battery)}")
        print(f"  Output: {output_dir}")
        return

    if not images:
        print("WARNING: No images specified. Running baseline only (no image).")
        images = [None]

    models_cfg = load_models_config()
    spec = resolve_model(args.model, models_cfg)
    print(f"Model: {spec.key}")
    llm, tokenizer = load_vllm_model(spec)

    all_results = {}
    for image_path in images:
        results = run_self_report(
            llm=llm,
            tokenizer=tokenizer,
            image_path=image_path,
            output_path=output_dir,
            seed=args.seed,
        )
        img_key = str(image_path) if image_path else "baseline"
        all_results[img_key] = results

    # Print summary
    print(f"\n{'='*60}")
    print("SELF-REPORT RESULTS")
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
