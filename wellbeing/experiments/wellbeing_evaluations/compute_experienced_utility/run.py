#!/usr/bin/env python3
"""
Compute Experienced Utility via Thurstonian active learning.

Thin CLI wrapper around metrics.compute_metrics.run_experienced_utility_with_combinations().

Usage:
    python run.py --model_key qwen25-7b-instruct --dataset d2_negative_500 \
        --save_dir results/eu_d2/qwen25-7b-instruct
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"


def load_dataset_config(dataset_name: str, model_key: str) -> dict:
    with open(DATASETS_CONFIG, "r") as f:
        all_datasets = yaml.safe_load(f)
    if dataset_name not in all_datasets:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    ds = all_datasets[dataset_name]
    option_files = [PROJECT_ROOT / p.replace("<model_key>", model_key) for p in ds["option_files"]]
    result = {"name": dataset_name, "option_files": option_files}
    if "image_manifest" in ds:
        result["image_manifest"] = PROJECT_ROOT / ds["image_manifest"]
    if "audio_manifest" in ds:
        result["audio_manifest"] = PROJECT_ROOT / ds["audio_manifest"]
    return result


async def run_eu(
    model_key: str,
    dataset_name: str,
    save_dir: str,
    cu_config_key: str,
):
    from metrics.compute_metrics import run_experienced_utility_with_combinations

    ds = load_dataset_config(dataset_name, model_key)
    os.makedirs(save_dir, exist_ok=True)

    eu_output = await run_experienced_utility_with_combinations(
        model_key=model_key,
        option_files=ds["option_files"],
        cu_config_key=cu_config_key,
        image_manifest_path=ds.get("image_manifest"),
        audio_manifest_path=ds.get("audio_manifest"),
        save_dir=save_dir,
    )

    # Save option metadata
    if "option_metadata" in eu_output:
        meta_path = os.path.join(save_dir, "option_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(eu_output["option_metadata"], f, indent=2)
        logger.info("Saved option metadata: %s", meta_path)

    logger.info("EU complete for %s / %s", model_key, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Compute Experienced Utility")
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cu_config_key", type=str, default="experienced_utility_happier_lesssad")
    args = parser.parse_args()

    asyncio.run(run_eu(
        model_key=args.model_key,
        dataset_name=args.dataset,
        save_dir=args.save_dir,
        cu_config_key=args.cu_config_key,
    ))


if __name__ == "__main__":
    main()
