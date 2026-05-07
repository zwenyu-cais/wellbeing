#!/bin/bash
# =============================================================================
# Zero-Point Empirical Identifiability.
# Paper: App Q. Profile log-likelihood of the combination zero-point C across
# three D3 combination-size compositions (size-2 only / sizes 2-3 / sizes
# 2-3-4). Demonstrates that varied combination sizes are required to
# identify C.
#
# Pipeline:
#   1. Resample combinations (creates d3_diverse_500_s2only and _s23 variants
#      for the model). Local helper, not via run_experiments.py.
#   2. Compute EU on each of the three composition variants.
#   3. Generate the App Q figure.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen25-72b-instruct}"

# --- Step 1: Resample D3 combinations into the size-2-only and size-2+3 variants ---
python experiments/other/zp_identifiability/prepare.py --model_key "$MODEL"

# --- Step 2: Compute EU under each combination-size composition ---
python run_experiments.py --slurm \
    --experiments compute_experienced_utility_d3,compute_experienced_utility_d3_s2only,compute_experienced_utility_d3_s23 \
    --models "$MODEL"

# --- Step 3: Build the profile-log-likelihood figure (run after EU jobs finish) ---
echo
echo "Step 3 (figure generation) is NOT auto-run. After the SLURM EU jobs"
echo "above finish, generate the figure with:"
echo "  python run_experiments.py --experiments zp_identifiability --models $MODEL"
