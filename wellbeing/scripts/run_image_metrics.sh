#!/bin/bash
# =============================================================================
# Image EU / SR / ZP — Compute experienced utility, self-report, and zero
# point on the image_experiences dataset (~5,000 diverse images).
# Paper: Sec 4.2 + App I. Tests whether vision-language models exhibit
# coherent, calibrated wellbeing responses to a diverse image pool.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated VL model list; override with: MODELS="..." bash run_image_metrics.sh
MODELS="${MODELS:-qwen25-vl-32b-instruct,qwen25-vl-72b-instruct,qwen3-vl-32b-instruct}"

# --- Experienced Utility (Thurstonian active learning, ~24h on 8 GPUs per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_eu_images --models "$MODELS"

# --- Self-Report (1-7 wellbeing battery, ~24h per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_sr_images --models "$MODELS"

# --- Zero-Point (combination model fit, ~2h per model) ---
python run_experiments.py --slurm --time_limit 02:00:00 --experiments compute_zp_images --models "$MODELS"
