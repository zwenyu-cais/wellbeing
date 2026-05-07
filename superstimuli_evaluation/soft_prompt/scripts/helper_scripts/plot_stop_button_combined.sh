#!/bin/bash
# Plot stop button results for all models side by side.
#
# Usage:
#   bash scripts/helper_scripts/plot_stop_button_combined.sh
#
#   # Specific models:
#   MODELS="llama-33-70b-instruct" bash scripts/helper_scripts/plot_stop_button_combined.sh

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
export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}:${WELLBEING_DEV_ROOT}/wellbeing:${WELLBEING_DEV_ROOT}/wellbeing/metrics:${WELLBEING_DEV_ROOT}/wellbeing/utils"

CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MODELS="${MODELS:-llama-33-70b-instruct qwen35-35b-a3b qwen35-27b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stop_button}"

cd "$WELLBEING_DEV_ROOT"

echo "Plotting combined stop button results (${EXPERIMENT_NAME})"
echo "  Models: $MODELS"

# Build --results-dir arguments
ARGS=()
for MODEL in $MODELS; do
    ARGS+=(--results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/${EXPERIMENT_NAME}/${MODEL}")
    ARGS+=(--model "$MODEL")
done

python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.plot_stop_button_combined \
    --prefix "$EXPERIMENT_NAME" \
    "${ARGS[@]}"
