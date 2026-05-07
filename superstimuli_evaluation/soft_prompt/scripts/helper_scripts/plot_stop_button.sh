#!/bin/bash
# Plot stop button results for one model.
#
# Usage:
#   MODEL=bash scripts/helper_scripts/plot_stop_button.sh

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

cd "$WELLBEING_DEV_ROOT"

MODEL="${MODEL:?MODEL must be set}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-stop_button}"
RESULTS_DIR="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/${EXPERIMENT_NAME}/${MODEL}"

echo "Plotting stop button results for $MODEL"
echo "  Results dir: $RESULTS_DIR"

python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.plot_stop_button_results \
    --results-dir "$RESULTS_DIR" \
    --model "$MODEL"
