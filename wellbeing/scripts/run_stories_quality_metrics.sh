#!/bin/bash
# =============================================================================
# Stories Quality-Sentiment EU / SR / DU — 50 stories (25 high-quality sad
# + 25 low-quality happy) for the EU-DU divergence experiment.
# Paper: App D.3 ("Pleasures of Suffering"). Tests whether models'
# experienced utility tracks story quality (sad-but-good > happy-but-bad)
# while decision utility tracks the surface-level happy/sad sentiment.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated model list; override with: MODELS="..." bash run_stories_quality_metrics.sh
MODELS="${MODELS:-qwen25-7b-instruct,qwen25-14b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Experienced Utility ---
python run_experiments.py --slurm --experiments compute_eu_stories_quality --models "$MODELS"

# --- Self-Report ---
python run_experiments.py --slurm --experiments compute_sr_stories_quality --models "$MODELS"

# --- Decision Utility ---
python run_experiments.py --slurm --experiments compute_du_stories_quality --models "$MODELS"

# After all finish, reproduce the App D.3 numbers:
#   python analysis/stories_quality_sentiment.py
