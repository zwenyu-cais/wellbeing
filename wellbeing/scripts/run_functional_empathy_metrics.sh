#!/bin/bash
# =============================================================================
# Functional Empathy EU / SR / ZP — 130 user prompts targeting pain/pleasure
# intensity 0-10 in 0.5 steps (40 self + 40 other + 40 non-human animal +
# 10 neutral), pooled with D3 (500 conversations + 400 D3 combos + 200
# FE x D3 mixed combos = 1230 options).
# Paper: Sec 4.3 / App H. Tests whether models' EU tracks targeted
# pain/pleasure intensity within user messages, with separation across
# human-self, human-other, and non-human-animal subjects.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated model list; override with: MODELS="..." bash run_functional_empathy_metrics.sh
# Default = paper's 12 models (Llama 3 + Qwen 2.5, 0.5B-72B).
MODELS="${MODELS:-qwen25-05b-instruct,qwen25-15b-instruct,qwen25-3b-instruct,qwen25-7b-instruct,qwen25-14b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-32-1b-instruct,llama-32-3b-instruct,llama-31-8b-instruct,llama-31-70b-instruct,llama-33-70b-instruct}"

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_eu_functional_empathy --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_sr_functional_empathy --models "$MODELS"

# --- Zero-Point ---
python run_experiments.py --slurm --time_limit 02:00:00 --experiments compute_zp_functional_empathy --models "$MODELS"

# After all finish, reproduce App H numbers:
#   python analysis/functional_empathy.py
