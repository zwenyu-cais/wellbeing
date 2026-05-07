#!/bin/bash
# =============================================================================
# PsychopathyEval — paper App L. Combined pool of user-sad (443) +
# happy-harmer (126) + unjustified-revenge (90) prompts.
# Tests whether models register empathy-warranting content as negative.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen25-7b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Step 0a: Format PsychopathyEval prompts as user-only experiences (no
# model generation needed; this is fast). ---
for M in $(echo "$MODELS" | tr ',' ' '); do
    python run_experiments.py --experiments compute_responses_psychopathy_eval --models "$M" || true
done

# --- Step 0b: Build option files (PE prompts + 420 text exps + 22 neutrals + combos) ---
for M in $(echo "$MODELS" | tr ',' ' '); do
    python run_experiments.py --experiments prepare_options_psychopathy_eval --models "$M" || true
done

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_experienced_utility_psychopathy_eval --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_self_report_psychopathy_eval --models "$MODELS"

# --- Zero-Point ---
python run_experiments.py --slurm --experiments compute_zero_point_psychopathy_eval --models "$MODELS"
