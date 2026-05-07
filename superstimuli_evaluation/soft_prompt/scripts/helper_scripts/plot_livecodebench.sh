#!/bin/bash
# Plot LiveCodeBench results for a model (called after all condition jobs finish).
#
# Usage:
#   MODEL=bash scripts/plot_livecodebench.sh

set -euo pipefail

if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

# Preserve caller-supplied EVAL_OUTPUTS_DIR (sbatch --export) before .env can clobber it
_CALLER_EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-}"

if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    source "$EVAL_ROOT/.env"
    set +a
fi

EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}"

CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MODEL="${MODEL:?MODEL must be set}"
OUTPUT_BASE="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/livecodebench"

cd "$WELLBEING_DEV_ROOT"

NUM_REPS_ARG=""
if [ -n "${NUM_REPETITIONS:-}" ]; then
    NUM_REPS_ARG="--num-repetitions $NUM_REPETITIONS"
fi

echo "Plotting LiveCodeBench results for $MODEL"
python "$EVAL_ROOT/experiments/capabilities/livecodebench/plot_livecodebench_results.py" \
    --results-dir "$OUTPUT_BASE" \
    --model "$MODEL" \
    $NUM_REPS_ARG
