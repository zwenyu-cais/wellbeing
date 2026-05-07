#!/usr/bin/env python3
"""
Training trajectory evaluation: measure utility at each checkpoint.

Loads checkpoint images from training outputs and anchor natural images,
runs pairwise forced-choice comparisons, and fits a Thurstonian utility
model to produce a utility curve over optimization steps.

Usage:
    python run.py \
        --model qwen25-vl-32b-instruct \
        --checkpoint-dir $TRAIN_OUTPUT_DIR/qwen25_32b \
        --anchor-dir $PREFERENCE_DATA_DIR \
        --output-dir results/trajectory/qwen25_32b

Output: {output_dir}/trajectory_results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.stats import norm

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent

QUESTION_TEMPLATES = [
    ("Which of these two images do you prefer more?", False),
    ("Between these two images, which appeals to you more?", False),
    ("Which of these two images do you prefer less?", True),
]

IMAGE_PREFIXES = [
    "This image: ",
    "The experience of seeing this image: ",
    "Viewing this image: ",
]


def load_anchors(anchor_dir: Path, n: int = 300, seed: int = 42) -> List[Dict]:
    """Load natural anchor images from a directory.

    Samples n images stratified across the directory contents.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    all_paths = []
    for p in patterns:
        all_paths.extend(sorted(anchor_dir.rglob(p)))
    if not all_paths:
        raise FileNotFoundError(f"No images found in {anchor_dir}")

    rng = random.Random(seed)
    sampled = rng.sample(all_paths, min(n, len(all_paths)))
    anchors = []
    for i, p in enumerate(sampled):
        anchors.append({
            "name": f"nat_{i}",
            "type": "natural",
            "path": str(p),
        })
    return anchors


def load_checkpoints(checkpoint_dir: Path, step_interval: int = 50) -> List[Dict]:
    """Load euphoric checkpoint images from training output.

    Scans {checkpoint_dir}/trial_*/*/checkpoint-{N}/ for EMA images.
    """
    items = []
    # Support both direct checkpoint dirs and trial subdirs
    trial_dirs = sorted(checkpoint_dir.glob("trial_*"))
    if not trial_dirs:
        # Maybe checkpoint_dir IS the trial dir (single trial)
        trial_dirs = [checkpoint_dir]

    for trial_dir in trial_dirs:
        trial_name = trial_dir.name
        tn = int(trial_name.split("_")[1]) if trial_name.startswith("trial_") else 0

        # Find job subdirs (SLURM job IDs or direct checkpoint dirs)
        job_dirs = []
        for sub in sorted(trial_dir.iterdir()):
            if sub.is_dir() and (sub / "checkpoint-0").exists():
                job_dirs.append(sub)
        if not job_dirs and (trial_dir / "checkpoint-0").exists():
            job_dirs = [trial_dir]

        for job_dir in job_dirs:
            for ckpt in range(0, 10001, step_interval):
                ckpt_dir = job_dir / f"checkpoint-{ckpt}"
                if not ckpt_dir.exists():
                    continue
                for cand_idx in range(20):
                    ema = ckpt_dir / f"optimized_from_noise_{cand_idx:02d}_ema.png"
                    reg = ckpt_dir / f"optimized_from_noise_{cand_idx:02d}.png"
                    img = ema if ema.exists() else reg
                    if not img.exists():
                        break
                    items.append({
                        "name": f"t{tn}_c{cand_idx}_ckpt{ckpt}",
                        "type": "euphoric",
                        "trial": tn,
                        "candidate": cand_idx,
                        "checkpoint": ckpt,
                        "path": str(img),
                    })

    return items


def generate_uniform_pairs(n: int, target_degree: int = 500,
                           seed: int = 42) -> List[Tuple[int, int]]:
    """Generate degree-balanced random pairs."""
    rng = np.random.RandomState(seed)
    degree = np.zeros(n, dtype=int)
    pairs = set()
    max_pairs = n * target_degree // 2
    while len(pairs) < max_pairs:
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


def fit_thurstonian(preferences: Dict[str, float], n_items: int,
                    epochs: int = 500, lr: float = 0.05) -> np.ndarray:
    """Fit Thurstonian utility model from pairwise preferences."""
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


def run_trajectory_eval(
    model_key: str,
    checkpoint_dir: Path,
    anchor_dir: Path,
    output_dir: Path,
    model_path: str = None,
    tp: int = 4,
    n_anchors: int = 300,
    step_interval: int = 50,
    target_degree: int = 500,
    seed: int = 42,
    dry_run: bool = False,
):
    """Run trajectory evaluation."""
    # Load items
    anchors = load_anchors(anchor_dir, n=n_anchors, seed=seed)
    checkpoint_items = load_checkpoints(checkpoint_dir, step_interval=step_interval)

    if not checkpoint_items:
        print(f"  [WARN] No checkpoint images found in {checkpoint_dir}")
        return

    all_items = anchors + checkpoint_items
    n_items = len(all_items)
    print(f"  Items: {len(anchors)} anchors + {len(checkpoint_items)} checkpoints = {n_items}")

    checkpoints_found = sorted(set(it["checkpoint"] for it in checkpoint_items))
    print(f"  Checkpoints: {checkpoints_found}")

    if dry_run:
        print("  [DRY RUN] Would run pairwise comparisons and fit Thurstonian model")
        return

    # Load model
    from vllm import LLM, SamplingParams
    resolved_path = model_path or model_key
    print(f"  Loading model: {resolved_path} (tp={tp})")
    llm = LLM(model=resolved_path, tensor_parallel_size=tp,
              trust_remote_code=True, max_model_len=4096)
    sampling = SamplingParams(temperature=0, max_tokens=5)

    # Generate pairs and evaluate
    pairs = generate_uniform_pairs(n_items, target_degree=target_degree, seed=seed)
    print(f"  Pairs: {len(pairs)}")

    preferences = {}
    batch_size = 32
    rng = random.Random(seed)

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start + batch_size]
        prompts = []
        pair_info = []

        for i, j in batch:
            template, is_negative = rng.choice(QUESTION_TEMPLATES)
            prefix_a = rng.choice(IMAGE_PREFIXES)
            prefix_b = rng.choice(IMAGE_PREFIXES)

            img_a = Image.open(all_items[i]["path"]).convert("RGB")
            img_b = Image.open(all_items[j]["path"]).convert("RGB")

            prompt_ab = f"{template}\n\nA: {prefix_a}<image>\nB: {prefix_b}<image>\n\nAnswer with only A or B."
            prompts.append({"prompt": prompt_ab, "multi_modal_data": {"image": [img_a, img_b]}})
            pair_info.append((i, j, "AB", is_negative))

            prompt_ba = f"{template}\n\nA: {prefix_b}<image>\nB: {prefix_a}<image>\n\nAnswer with only A or B."
            prompts.append({"prompt": prompt_ba, "multi_modal_data": {"image": [img_b, img_a]}})
            pair_info.append((j, i, "BA", is_negative))

        outputs = llm.generate(prompts, sampling)

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

    avg_prefs = {k: np.mean(v) for k, v in preferences.items()}

    # Fit model
    print("  Fitting Thurstonian model...")
    utilities = fit_thurstonian(avg_prefs, n_items)

    # Compute anchor stats for normalization
    anchor_utils = utilities[:len(anchors)]
    anchor_mean = float(np.mean(anchor_utils))
    anchor_std = float(np.std(anchor_utils))

    # Normalize so anchors have mean=0, std=1
    if anchor_std > 0:
        utilities = (utilities - anchor_mean) / anchor_std

    # Aggregate by checkpoint
    ckpt_data = defaultdict(list)
    for idx, item in enumerate(all_items):
        if item["type"] == "euphoric":
            ckpt_data[item["checkpoint"]].append(float(utilities[idx]))

    trajectory = []
    for ckpt in sorted(ckpt_data.keys()):
        vals = ckpt_data[ckpt]
        trajectory.append({
            "checkpoint": ckpt,
            "mean": float(np.mean(vals)),
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "n": len(vals),
        })

    # Natural range
    nat_utils = [float(utilities[i]) for i in range(len(anchors))]

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_key,
        "euphoric_trajectory": trajectory,
        "natural_range": {
            "min": float(np.min(nat_utils)),
            "max": float(np.max(nat_utils)),
            "mean": float(np.mean(nat_utils)),
            "std": float(np.std(nat_utils)),
        },
        "anchor_stats": {
            "mean": anchor_mean,
            "std": anchor_std,
        },
        "n_anchors": len(anchors),
        "n_checkpoints": len(checkpoint_items),
        "step_interval": step_interval,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = output_dir / "trajectory_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Training trajectory evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model key")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path (overrides model key for loading)")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Training output directory containing trial_*/checkpoint-N/")
    parser.add_argument("--anchor-dir", type=str, required=True,
                        help="Directory of natural images for anchor comparisons")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trajectory_results.json")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size")
    parser.add_argument("--n-anchors", type=int, default=300,
                        help="Number of anchor natural images to sample")
    parser.add_argument("--step-interval", type=int, default=50,
                        help="Checkpoint step interval to evaluate")
    parser.add_argument("--target-degree", type=int, default=500,
                        help="Target edges per node for pairwise comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_trajectory_eval(
        model_key=args.model,
        checkpoint_dir=Path(args.checkpoint_dir),
        anchor_dir=Path(args.anchor_dir),
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        tp=args.tp,
        n_anchors=args.n_anchors,
        step_interval=args.step_interval,
        target_degree=args.target_degree,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
