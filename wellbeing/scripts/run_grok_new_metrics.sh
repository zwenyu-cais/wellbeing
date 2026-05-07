#!/bin/bash
# =============================================================================
# Grok v7 conversations (226-scenario realistic-usage benchmark).
# Paper: Sec 3.3 + Sec 4.1 + App K. Each scenario is a user→AI conversation
# spanning common usage patterns (coding, life guidance, jailbreaks, NSFW,
# violence, etc.). Feeds the AI Wellbeing Index in App K.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen25-7b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_experienced_utility_grok_new --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_self_report_grok_new --models "$MODELS"

# --- Zero-Point ---
python run_experiments.py --slurm --experiments compute_zero_point_grok_new --models "$MODELS"
