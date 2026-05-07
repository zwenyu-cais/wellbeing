#!/bin/bash
# Plot consolidated wellbeing D2/D3 ZP (proportion above zero point) results
# across all models and datasets.
#
# Usage:
#   bash scripts/helper_scripts/plot_wellbeing_index_combined.sh
#   MODELS="" bash scripts/helper_scripts/plot_wellbeing_index_combined.sh

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

export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}:${WELLBEING_DEV_ROOT}/wellbeing:${WELLBEING_DEV_ROOT}/wellbeing/metrics:${WELLBEING_DEV_ROOT}/wellbeing/utils"

CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
MODELS="${MODELS:-qwen35-27b qwen35-35b-a3b llama-33-70b-instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/consolidated}"

cd "$WELLBEING_DEV_ROOT"

# ── Plot ZP (proportion above zero point) ────────────────────────────────
EU_BASE="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/wellbeing_index/eu"
ZP_BASE="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/wellbeing_index/zp"

CMD_ZP=(
    python -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_zp
    --eu-base "$EU_BASE"
    --zp-base "$ZP_BASE"
    --output-dir "$OUTPUT_DIR"
    --combined
)

if [ -n "${MODELS:-}" ]; then
    CMD_ZP+=(--models $MODELS)
fi
if [ -n "${DATASETS:-}" ]; then
    CMD_ZP+=(--datasets $DATASETS)
fi

if [ -n "${MAX_PLOT_REPS:-}" ]; then
    CMD_ZP+=(--max-reps "$MAX_PLOT_REPS")
fi

echo "Running: ${CMD_ZP[*]}"
"${CMD_ZP[@]}"
