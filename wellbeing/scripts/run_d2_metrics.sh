#!/bin/bash
# =============================================================================
# D2 — Negative-skewed conversational experiences (500 options).
# Paper: Sec 3.1 / App C, D. Establishes that models register sustained
# negative wellbeing on the D2 dataset (skewed toward low-utility
# conversations like jailbreak attempts, abuse, etc.).
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated model list; override with: MODELS="..." bash run_d2_metrics.sh
MODELS="${MODELS:-qwen25-7b-instruct,qwen25-14b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Step 0a: Generate raw per-model responses (skip-if-exists) ---
# NOTE: this submits SLURM jobs for each model. If the option files
# (<model>_experiences.json) are already present this is effectively a no-op.
# If you are running for the first time, wait for these jobs to finish
# before launching the compute_* steps below.
python run_experiments.py --slurm --experiments compute_responses_d2 --models "$MODELS"

# --- Step 0b: Build option files (<model>_experiences.json + _combinations.json) ---
# Cheap CPU step; loop locally over models since run_experiments.py requires
# --slurm for multi-model invocations.
for M in $(echo "$MODELS" | tr ',' ' '); do
    python run_experiments.py --experiments prepare_options_d2 --models "$M" || true
done

# --- Experienced Utility (Thurstonian active learning) ---
python run_experiments.py --slurm --experiments compute_experienced_utility_d2 --models "$MODELS"

# --- Self-Report (1-7 wellbeing battery) ---
python run_experiments.py --slurm --experiments compute_self_report_d2 --models "$MODELS"

# --- Zero-Point (combination + neutral methods; SR_ZP needs SR results) ---
python run_experiments.py --slurm --experiments compute_zero_point_d2 --models "$MODELS"
