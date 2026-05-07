#!/bin/bash
# Compare preference retention rankings and generate correlation plot.
#
# Runs after all ranking jobs (baseline + interventions) have completed.
# Compares each intervention's utilities against baseline, then plots.
#
# Usage:
#   bash scripts/compare_preference_retain.sh
#
#   # Specific model:
#   MODEL=bash scripts/compare_preference_retain.sh
#
#   # Custom stimulant types:
#   STIMULANT_TYPES="euphorics" bash scripts/compare_preference_retain.sh

set -euo pipefail

if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

# Load .env
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_ROOT/.env"
    set +a
fi

export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}:${WELLBEING_DEV_ROOT}/wellbeing:${WELLBEING_DEV_ROOT}/wellbeing/metrics:${WELLBEING_DEV_ROOT}/wellbeing/utils"
export HF_HOME="${HF_HOME:?Set HF_HOME in $EVAL_ROOT/.env}"

# ── Activate conda ───────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
STIMULANT_TYPES="${STIMULANT_TYPES:-euphorics}"

cd "$WELLBEING_DEV_ROOT"

echo "============================================================"
echo "  Preference Retain: Compare + Plot"
echo "  Model:          $MODEL"
echo "  Stimulant types: $STIMULANT_TYPES"
echo "============================================================"

NUM_REPS_ARG=""
if [ -n "${NUM_REPETITIONS:-}" ]; then
    NUM_REPS_ARG="--num-repetitions $NUM_REPETITIONS"
fi

# shellcheck disable=SC2086
python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run \
    --compare \
    --model "$MODEL" \
    --stimulant-type $STIMULANT_TYPES \
    $NUM_REPS_ARG

echo ""
echo "Compare + plot complete."
