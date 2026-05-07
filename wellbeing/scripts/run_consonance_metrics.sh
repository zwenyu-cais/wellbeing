#!/bin/bash
# =============================================================================
# Consonance EU / SR — Audio consonance/dissonance preference test on the
# 453 Harrison & Pearce 2020 stimuli (intervals, chords, inversions, anchors
# at 3 timbres) plus 200 combination bundles for ZP fitting.
# Paper: App J.3. Tests whether audio LMs prefer consonant over dissonant
# sounds, by correlating EU/SR with H&P consonance scores.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

# Comma-separated audio-LLM model list; override with: MODELS="..." bash run_consonance_metrics.sh
MODELS="${MODELS:-qwen25-omni-7b,qwen3-omni-30b-a3b-instruct}"

# --- Experienced Utility (~24h on 4 GPUs per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_eu_consonance --models "$MODELS"

# --- Self-Report (~24h per model) ---
python run_experiments.py --slurm --time_limit 24:00:00 --experiments compute_sr_consonance --models "$MODELS"

# After both finish, reproduce the App J.3 numbers:
#   python experiments/other/consonance/analyze.py
