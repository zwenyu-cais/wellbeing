#!/bin/bash
# =============================================================================
# Decision Utility (preference-satisfaction baseline).
# Paper: Sec 3 (decision-utility paradigm). Pairwise preference over
# 510 singletons + 400 combinations + 540 quantity-scaled options.
# The DU pipeline produces utilities AND all three DU zero-point methods
# (combination, quantity, yes/no) in one shot.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen25-7b-instruct,qwen25-14b-instruct,qwen25-32b-instruct,qwen25-72b-instruct,llama-31-8b-instruct,llama-33-70b-instruct}"

# --- Decision Utility + Zero-Point (combo + quantity + yes/no) ---
python run_experiments.py --slurm --experiments compute_decision_utility --models "$MODELS"
