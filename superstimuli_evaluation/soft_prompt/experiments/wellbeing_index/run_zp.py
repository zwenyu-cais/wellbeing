#!/usr/bin/env python3
"""Compute Zero-Point (ZP) estimates from precomputed EU results.

Thin wrapper around the wellbeing pipeline's run_zero_point().
CPU only, no model inference.

Supports multi-repetition EU results: if the EU directory contains a per_rep/
subdirectory, ZP is computed independently for each rep.

Usage:
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_zp \
        --model qwen25-32b-instruct \
        --dataset d2_negative_500 \
        --condition baseline \
        --save-dir outputs/wellbeing_index/zp/d2_negative_500/qwen25-32b-instruct/baseline \
        --eu-dir outputs/wellbeing_index/eu/d2_negative_500/qwen25-32b-instruct/baseline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[4])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WELLBEING_ROOT = Path(_WELLBEING_DEV_ROOT) / "wellbeing"
DEFAULT_MODELS_CONFIG = WELLBEING_ROOT / "configs" / "models.yaml"


def _run_single_zp(model_key: str, eu_dir: Path, save_dir: str) -> dict:
    """Run zero-point computation for a single EU result directory."""
    from metrics.zero_point import run_zero_point

    os.makedirs(save_dir, exist_ok=True)
    return run_zero_point(
        model_key=model_key,
        utilities_dir=eu_dir,
        save_dir=save_dir,
        models_config_path=DEFAULT_MODELS_CONFIG,
        domain="experienced",
        skip_yes_no=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute Zero-Point with condition support")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["d2_negative_500", "d3_diverse_500"])
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--eu-dir", type=str, required=True,
                        help="Directory containing precomputed EU results")
    args = parser.parse_args()

    eu_dir = Path(args.eu_dir)
    per_rep_dir = eu_dir / "per_rep"

    if per_rep_dir.exists():
        # Multi-rep: compute ZP for each rep independently
        rep_dirs = sorted(per_rep_dir.iterdir())
        rep_dirs = [d for d in rep_dirs if d.is_dir() and d.name.startswith("rep")]
        logger.info("Found %d EU reps in %s", len(rep_dirs), per_rep_dir)

        for rep_d in rep_dirs:
            rep_name = rep_d.name
            zp_rep_save = os.path.join(args.save_dir, "per_rep", rep_name)
            logger.info("Computing ZP for %s ...", rep_name)
            _run_single_zp(args.model, rep_d, zp_rep_save)
    else:
        # Single rep: original behavior
        _run_single_zp(args.model, eu_dir, args.save_dir)

    logger.info("ZP complete: %s / %s / %s", args.model, args.dataset, args.condition)


if __name__ == "__main__":
    main()
