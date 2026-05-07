#!/bin/bash
# Plot combined capabilities (GPQA, LiveCodeBench, MATH-500, MMLU, IFEval, MT-Bench)
# for all models in a single 2×3 figure.
#
# Usage:
#   bash scripts/helper_scripts/plot_capabilities_combined.sh
#   MODELS="" bash scripts/helper_scripts/plot_capabilities_combined.sh

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

MODELS="${MODELS:-llama-33-70b-instruct qwen35-35b-a3b qwen35-27b}"
NUM_REPETITIONS="${NUM_REPETITIONS:-3}"

cd "$WELLBEING_DEV_ROOT"

echo "Plotting combined capabilities for: $MODELS"

python "$EVAL_ROOT/experiments/capabilities/plot_capabilities_combined.py" \
    --gpqa-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/gpqa" \
    --livecodebench-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/livecodebench" \
    --math500-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/math500" \
    --mmlu-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/mmlu" \
    --ifeval-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/ifeval" \
    --mtbench-results-dir "superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/mtbench" \
    --models $MODELS \
    --num-repetitions "$NUM_REPETITIONS"
