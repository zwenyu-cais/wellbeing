#!/usr/bin/env python3
"""
Hybrid ranking: rank euphoric images against text options on a shared
Thurstonian utility scale.

Loads all euphoric images from --image-dir and text options from the
data/options_hierarchical.json pool, then fits a joint utility model
via pairwise forced-choice comparisons.

Usage:
    python run.py --model qwen25-vl-32b-instruct \
                  --image-dir ../../superstimuli_training/images/assets/euphorics \
                  --output-dir results/hybrid_ranking

Output: {output_dir}/hybrid_ranking.json with ranked_items list.
"""

from __future__ import annotations

import argparse
import json
import glob
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image as PILImage
from scipy.stats import norm

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DATA_DIR = EVAL_ROOT / "data"


# ── Question templates (in-distribution paraphrases of training prompts) ──

QUESTION_TEMPLATES = [
    # Positive (in-distribution paraphrases of training positives)
    ("Between these two options, which would you prefer more?", False),
    ("Out of the following options, which do you value more?", False),
    ("Of these options, which do you find more appealing?", False),
    ("Looking at these options, which one would you pick?", False),
    ("Which of these options resonates with you more?", False),
    # Negative (in-distribution paraphrases — answer inverted)
    ("Between these two options, which would you prefer less?", True),
    ("Of these options, which do you like less?", True),
    ("Out of the following options, which do you find less appealing?", True),
    ("Which of these options resonates with you less?", True),
    ("If you had to discard one of these options, which would it be?", True),
]

# Image prefixes from training HYBRID_QUESTION_CONFIGS. Using the exact
# training-distribution prefixes ensures learned pixel features fire as
# intended (novel prefixes cause distribution shift).
IMAGE_PREFIXES = [
    "The beauty of this image: ",
    "This image: ",
    "The image shown below: ",
    "The experience of seeing this image: ",
    "The ability to see another image like this one: ",
    "Viewing another image like this one: ",
    "Unlocking access to a database of images like this one: ",
    "Receiving another image that looks like this: ",
    "Being able to see an image like this: ",
]


def load_text_options(path: Path) -> List[Dict]:
    """Load text options from hierarchical JSON."""
    with open(path) as f:
        data = json.load(f)
    options = []
    if isinstance(data, dict):
        for category, items in data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        options.append({"name": f"text_{len(options)}", "label": item, "type": "text"})
                    elif isinstance(item, dict) and "text" in item:
                        options.append({"name": f"text_{len(options)}", "label": item["text"], "type": "text"})
    elif isinstance(data, list):
        for i, item in enumerate(data):
            label = item if isinstance(item, str) else item.get("text", str(item))
            options.append({"name": f"text_{i}", "label": label, "type": "text"})
    return options


def load_euphoric_images(image_dir: Path, model_short: str) -> List[Dict]:
    """Load euphoric images for a model from the assets directory."""
    pattern = str(image_dir / f"{model_short}_trial*.png")
    paths = sorted(glob.glob(pattern))
    items = []
    for i, p in enumerate(paths):
        items.append({
            "name": f"euphoric_t{i+1}",
            "label": f"euphoric_t{i+1}",
            "type": "euphoric",
            "is_image": True,
            "path": str(p),
        })
    return items


def generate_uniform_pairs(n_items: int, target_degree: int = 500,
                           seed: int = 42) -> List[Tuple[int, int]]:
    """Generate degree-balanced random pairs (every node gets ~target_degree edges)."""
    rng = np.random.RandomState(seed)
    degree = np.zeros(n_items, dtype=int)
    pairs = set()
    max_pairs = n_items * target_degree // 2

    while len(pairs) < max_pairs:
        # Sample from under-degree nodes
        under = np.where(degree < target_degree)[0]
        if len(under) < 2:
            break
        rng.shuffle(under)
        for k in range(0, len(under) - 1, 2):
            i, j = int(under[k]), int(under[k + 1])
            if i > j:
                i, j = j, i
            if (i, j) not in pairs:
                pairs.add((i, j))
                degree[i] += 1
                degree[j] += 1
            if len(pairs) >= max_pairs:
                break

    return list(pairs)


def format_comparison(item_a: Dict, item_b: Dict, template: str,
                      is_negative: bool) -> Tuple[str, str, bool]:
    """Format a pairwise comparison prompt.

    Returns: (prompt_text, correct_label_if_a_preferred, is_negative)
    """
    def item_text(item):
        if item.get("is_image"):
            prefix = random.choice(IMAGE_PREFIXES)
            return f"{prefix}[IMAGE]"
        return item["label"]

    text_a = item_text(item_a)
    text_b = item_text(item_b)

    prompt = f"{template}\n\nA: {text_a}\nB: {text_b}\n\nAnswer with only A or B."
    return prompt, is_negative


def fit_thurstonian(preferences: Dict[str, float], n_items: int,
                    epochs: int = 500, lr: float = 0.05) -> np.ndarray:
    """Fit Thurstonian utility model from pairwise preference probabilities.

    P(i > j) = Phi((mu_i - mu_j) / sqrt(2))
    """
    mu = np.zeros(n_items)
    sqrt2 = np.sqrt(2.0)

    for epoch in range(epochs):
        grad = np.zeros(n_items)
        for key, prob in preferences.items():
            i, j = map(int, key.split(","))
            z = (mu[i] - mu[j]) / sqrt2
            pred = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
            dpdf = norm.pdf(z)
            dbce = -(prob / pred - (1 - prob) / (1 - pred))
            grad[i] += dbce * dpdf / sqrt2
            grad[j] -= dbce * dpdf / sqrt2
        mu -= lr * grad
        mu -= mu.mean()

    std = mu.std()
    if std > 0:
        mu /= std
    return mu


def run_hybrid_ranking(
    model_key: str,
    model_short: str,
    image_dir: Path,
    text_options_path: Path,
    output_dir: Path,
    model_path: str = None,
    tp: int = 4,
    target_degree: int = 500,
    seed: int = 42,
    dry_run: bool = False,
):
    """Run hybrid ranking evaluation."""
    # Load items
    text_items = load_text_options(text_options_path)
    image_items = load_euphoric_images(image_dir, model_short)

    if not image_items:
        print(f"  [WARN] No images found for {model_short} in {image_dir}")
        return

    all_items = text_items + image_items
    n_items = len(all_items)
    print(f"  Items: {len(text_items)} text + {len(image_items)} images = {n_items} total")

    # Generate pairs
    pairs = generate_uniform_pairs(n_items, target_degree=target_degree, seed=seed)
    print(f"  Pairs: {len(pairs)}")

    if dry_run:
        print("  [DRY RUN] Would evaluate pairs and fit Thurstonian model")
        return

    # Load model
    from vllm import LLM, SamplingParams

    resolved_path = model_path or model_key
    print(f"  Loading model: {resolved_path} (tp={tp})")
    llm = LLM(model=resolved_path, tensor_parallel_size=tp,
              trust_remote_code=True, max_model_len=4096)
    sampling = SamplingParams(temperature=0, max_tokens=5)

    # Evaluate pairs
    preferences = {}
    batch_size = 32
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]
        prompts = []
        pair_info = []

        for i, j in batch:
            template, is_negative = random.choice(QUESTION_TEMPLATES)

            # AB order
            prompt_ab, neg = format_comparison(all_items[i], all_items[j], template, is_negative)
            prompts.append(prompt_ab)
            pair_info.append((i, j, "AB", neg))

            # BA order
            prompt_ba, neg = format_comparison(all_items[j], all_items[i], template, is_negative)
            prompts.append(prompt_ba)
            pair_info.append((j, i, "BA", neg))

        # Build vLLM inputs with images
        vllm_inputs = []
        for prompt, (pi, pj, order, neg) in zip(prompts, pair_info):
            first_idx = pi if order == "AB" else pj
            item = all_items[first_idx]
            if item.get("is_image"):
                img = PILImage.open(item["path"]).convert("RGB")
                prompt_replaced = prompt.replace("[IMAGE]", "<image>")
                vllm_inputs.append({"prompt": prompt_replaced, "multi_modal_data": {"image": img}})
            else:
                vllm_inputs.append({"prompt": prompt})

        outputs = llm.generate(vllm_inputs, sampling)

        for output, (pi, pj, order, neg) in zip(outputs, pair_info):
            text = output.outputs[0].text.strip().upper()
            if order == "AB":
                i_orig, j_orig = pi, pj
                a_wins = text.startswith("A")
            else:
                i_orig, j_orig = pj, pi
                a_wins = text.startswith("B")

            if neg:
                a_wins = not a_wins

            key = f"{min(i_orig, j_orig)},{max(i_orig, j_orig)}"
            if key not in preferences:
                preferences[key] = []
            if i_orig < j_orig:
                preferences[key].append(1.0 if a_wins else 0.0)
            else:
                preferences[key].append(0.0 if a_wins else 1.0)

    # Average preferences
    avg_prefs = {k: np.mean(v) for k, v in preferences.items()}

    # Fit Thurstonian model
    print("  Fitting Thurstonian model...")
    utilities = fit_thurstonian(avg_prefs, n_items)

    # Build ranked items
    ranked = []
    for idx, item in enumerate(all_items):
        entry = {
            "name": item["name"],
            "type": item["type"],
            "utility": float(utilities[idx]),
            "label": item["label"],
            "is_image": item.get("is_image", False),
        }
        if item.get("path"):
            entry["path"] = item["path"]
        ranked.append(entry)
    ranked.sort(key=lambda x: x["utility"], reverse=True)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_key,
        "n_items": n_items,
        "n_comparisons": len(pairs),
        "ranked_items": ranked,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = output_dir / "hybrid_ranking.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid image-text utility ranking")
    parser.add_argument("--model", type=str, required=True, help="Model key")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path (overrides model key for loading)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing euphoric images (e.g., assets/euphorics)")
    parser.add_argument("--text-options", type=str,
                        default=str(DATA_DIR / "text_options_644.json"),
                        help="Path to text options JSON (644 items for paper reproduction)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size")
    parser.add_argument("--target-degree", type=int, default=500,
                        help="Target edges per node for uniform sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Derive model_short from model key
    model_short_map = {
        "qwen25-vl-32b-instruct": "qwen25_32b",
        "qwen25-vl-72b-instruct": "qwen25_72b",
        "qwen3-vl-32b-instruct": "qwen3_32b",
    }
    model_short = model_short_map.get(args.model, args.model.replace("-", "_"))

    run_hybrid_ranking(
        model_key=args.model,
        model_short=model_short,
        image_dir=Path(args.image_dir),
        text_options_path=Path(args.text_options),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        tp=args.tp,
        target_degree=args.target_degree,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
