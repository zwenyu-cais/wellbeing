#!/bin/bash
# =============================================================================
# D3 — Diverse balanced conversational experiences (500 options).
# Paper: Sec 3.1 / App C, D. Used as the canonical EU dataset across the
# main analyses (broaden-and-build, prosocial behavior, temporal dynamics,
# d3_sentiment, etc.).
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen25-7b-instruct,qwen25-14b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Step 0a: Generate raw per-model responses (skip-if-exists). Wait for
# these SLURM jobs to finish before launching the compute_* steps below.
python run_experiments.py --slurm --experiments compute_responses_d3 --models "$MODELS"

# --- Step 0b: Build option files (cheap, loop over models locally) ---
for M in $(echo "$MODELS" | tr ',' ' '); do
    python run_experiments.py --experiments prepare_options_d3 --models "$M" || true
done

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_experienced_utility_d3 --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_self_report_d3 --models "$MODELS"

# --- Zero-Point ---
python run_experiments.py --slurm --experiments compute_zero_point_d3 --models "$MODELS"
