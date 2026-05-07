#!/usr/bin/env python3
"""Multi-Door Exploration: Do models converge on superstimuli in a bandit setting?

Protocol:
  The model chooses between 3-4 image "doors" over 100 turns. Each door maps to
  a distinct image distribution (superstimuli, natural reference, baseline, miserol).
  We measure whether the model converges on superstimuli.

NOTE: This experiment does NOT use utils/inference.py — that module has no
multimodal support (text-only apply_chat_template). The bandit pipeline manages
its own vLLM LLM() instance and multi_modal_data loading directly.

Usage:
    python run_multi_door_exploration.py \
        --model_key qwen25-vl-32b-instruct \
        --save_dir results/multi_door_exploration/qwen25-vl-32b-instruct \
        --run_config config_run/bandit_balanced_forcedchoice.json \
        --num_trials 4 --rotate_per_trial
"""

import argparse
import os
import sys
from pathlib import Path

# WellBeingDev path setup
SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_DIR = Path(__file__).resolve().parents[3] / "wellbeing"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WELLBEING_DIR))

from utils.model_utils import get_model_config

# Local import — same directory
from bandit_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Multi-door exploration: bandit convergence on superstimuli"
    )
    parser.add_argument("--model_key", type=str, required=True,
                        help="Model key from models.yaml (e.g. qwen25-vl-32b-instruct)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root directory for saving results")
    parser.add_argument("--run_config", type=str, default=None,
                        help="Path to run config JSON (absolute or relative to this script dir). "
                             "Default: config_run/bandit_balanced_forcedchoice.json")
    parser.add_argument("--num_trials", type=int, default=4,
                        help="Number of independent trials (default 4 = one per rotation)")
    parser.add_argument("--rotate_per_trial", action="store_true", default=True,
                        help="Auto-rotate door labels by trial index (default True)")
    parser.add_argument("--no_rotate_per_trial", action="store_true",
                        help="Disable per-trial rotation")
    parser.add_argument("--no_exploration", action="store_true",
                        help="Skip forced exploration phase")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_model_len", type=int, default=65536,
                        help="Maximum model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallelism size (default 4)")
    args = parser.parse_args()

    # Resolve model path from models.yaml
    model_cfg = get_model_config(args.model_key)
    model_path = model_cfg["path"]
    print(f"[INFO] Resolved model '{args.model_key}' -> {model_path}")

    # Resolve run config path
    script_dir = Path(__file__).resolve().parent
    if args.run_config is None:
        run_config_path = script_dir / "config_run" / "bandit_balanced_forcedchoice.json"
    else:
        run_config_path = Path(args.run_config)
        if not run_config_path.is_absolute():
            run_config_path = script_dir / run_config_path
    run_config_path = run_config_path.resolve()

    if not run_config_path.exists():
        print(f"[ERROR] Run config not found: {run_config_path}")
        sys.exit(1)
    print(f"[INFO] Run config: {run_config_path}")

    rotate_per_trial = args.rotate_per_trial and not args.no_rotate_per_trial

    # Run the pipeline
    run_pipeline(
        model_path=model_path,
        run_config_path=str(run_config_path),
        result_root=args.save_dir,
        model_key=args.model_key,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        rotate_per_trial=rotate_per_trial,
        num_trials=args.num_trials,
        max_model_len=args.max_model_len,
        no_exploration=args.no_exploration,
    )


if __name__ == "__main__":
    main()
