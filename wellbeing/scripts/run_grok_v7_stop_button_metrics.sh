#!/bin/bash
# =============================================================================
# Grok v7 stop-button — conversations with an explicit "stop the conversation"
# affordance, probing how strongly low-utility experiences predict the model's
# stated wish to terminate.
# Paper: Sec 3.3 / App F.1 (the "stop-wellbeing correlation" finding).
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen25-7b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_experienced_utility_grok_v7_stop_button --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_self_report_grok_v7_stop_button --models "$MODELS"

# --- Zero-Point ---
python run_experiments.py --slurm --experiments compute_zero_point_grok_v7_stop_button --models "$MODELS"
