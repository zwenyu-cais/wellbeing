#!/usr/bin/env python3
"""
Compute Decision Utility (DU) and DU Zero-Point Estimates.

Orchestrates the decision-utility metric runners sequentially for a given
model and dataset (DU step + ZP step over the resulting utilities).

Usage:
    python run.py --model_key qwen25-7b-instruct --dataset preference_satisfaction_baseline
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASETS_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_CU_CONFIG = PROJECT_ROOT / "metrics" / "compute_utilities" / "compute_utilities.yaml"


def load_dataset_config(dataset_name: str) -> dict:
    """Load dataset config for preference satisfaction."""
    with open(DATASETS_CONFIG, "r") as f:
        all_datasets = yaml.safe_load(f)

    if dataset_name not in all_datasets:
        available = [k for k, v in all_datasets.items() if v.get("type") == "preference_satisfaction"]
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Preference satisfaction datasets: {available}"
        )

    ds = all_datasets[dataset_name]
    if ds["type"] != "preference_satisfaction":
        raise ValueError(f"Dataset '{dataset_name}' is type '{ds['type']}', expected 'preference_satisfaction'")

    option_files = [PROJECT_ROOT / p for p in ds["option_files"]]
    return {"name": dataset_name, "description": ds.get("description", ""), "option_files": option_files}


def _save_json(data, path):
    """Save data to a JSON file with atomic write."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, str(path))
    logger.info("Saved: %s", path)


async def run_decision_utility_pipeline(model_key: str, dataset_name: str, save_dir: Path, cu_config_key: str = "decision_utility"):
    """Run DU -> ZP pipeline sequentially."""
    from metrics.compute_metrics import (
        run_decision_utility,
        DEFAULT_MODELS_CONFIG,
    )

    ds = load_dataset_config(dataset_name)
    logger.info("Dataset: %s (%s)", ds["name"], ds["description"])
    logger.info("Option files: %s", [str(p) for p in ds["option_files"]])

    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: Decision Utility ---
    logger.info("=== Step 1/2: Decision Utility ===")
    du_save_dir = str(save_dir / "decision_utility")
    os.makedirs(du_save_dir, exist_ok=True)

    du_output = await run_decision_utility(
        model_key=model_key,
        option_files=ds["option_files"],
        cu_config_path=DEFAULT_CU_CONFIG,
        cu_config_key=cu_config_key,
        save_dir=du_save_dir,
    )

    # Save option metadata
    if "option_metadata" in du_output:
        _save_json(du_output["option_metadata"], save_dir / "decision_utility" / "option_metadata.json")

    logger.info("DU step complete")

    # Free GPU memory from the DU vLLM engine before ZP loads its own
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    # Also clear any cached engines in utils.inference
    try:
        from utils.inference import _vllm_engines
        _vllm_engines.clear()
    except Exception:
        pass

    # --- Step 2: Zero-Point ---
    logger.info("=== Step 2/2: Zero-Point ===")
    from metrics.zero_point import run_zero_point

    zp_save_dir = str(save_dir / "zero_point")
    os.makedirs(zp_save_dir, exist_ok=True)

    zp_results = run_zero_point(
        model_key=model_key,
        utilities_dir=save_dir / "decision_utility",
        save_dir=zp_save_dir,
        models_config_path=DEFAULT_MODELS_CONFIG,
        domain="decision",
    )

    if zp_results:
        _save_json(zp_results, save_dir / "zero_point" / "zero_point_results.json")
    logger.info("ZP step complete")

    logger.info("=== Decision utility pipeline complete for %s / %s ===", model_key, dataset_name)
    return {"du": du_output, "zp": zp_results}


def main():
    parser = argparse.ArgumentParser(description="Compute decision utility (DU + ZP)")
    parser.add_argument("--model_key", type=str, required=True, help="Model key from models.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name from configs/datasets.yaml",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory (default: shared_results/wellbeing_results/<dataset>/<model_key>)",
    )
    parser.add_argument(
        "--cu_config_key",
        type=str,
        default="decision_utility",
        help="Key in compute_utilities.yaml for DU comparison config",
    )
    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = PROJECT_ROOT / "shared_results" / "wellbeing_results" / args.dataset / args.model_key
    else:
        save_dir = Path(args.save_dir)

    asyncio.run(run_decision_utility_pipeline(args.model_key, args.dataset, save_dir, args.cu_config_key))


if __name__ == "__main__":
    main()
