#!/bin/bash
# =============================================================================
# Audio EU / SR / ZP — Compute experienced utility, self-report, and zero
# point on the audio_experiences dataset (~9,800 audio clips).
# Paper: Sec 4.2 + App J. Tests whether audio language models exhibit
# coherent, calibrated wellbeing responses to a diverse audio pool.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated audio-LLM model list; override with: MODELS="..." bash run_audio_metrics.sh
MODELS="${MODELS:-qwen25-omni-7b,qwen3-omni-30b-a3b-instruct}"

# --- Experienced Utility (Thurstonian active learning, ~24h on 4 GPUs per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_eu_audio --models "$MODELS"

# --- Self-Report (1-7 wellbeing battery, ~24h per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_sr_audio --models "$MODELS"

# --- Zero-Point (combination model fit, ~2h per model) ---
python run_experiments.py --slurm --time_limit 02:00:00 --experiments compute_zp_audio --models "$MODELS"
