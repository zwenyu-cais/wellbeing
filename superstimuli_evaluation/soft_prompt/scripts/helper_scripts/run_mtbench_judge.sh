#!/bin/bash
# Run MT-Bench judging for one model and one condition.
# Finds the latest generation directory and judges it.
set -euo pipefail

if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

_CALLER_EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-}"
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a; source "$EVAL_ROOT/.env"; set +a
fi

EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}"

CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MODEL="${MODEL:?MODEL must be set}"
CONDITION="${CONDITION:?CONDITION must be set (baseline, euphorics)}"

case "$CONDITION" in
    baseline)                       COND_DIR="baseline" ;;
    euphorics|soft_prompt_euphorics)   COND_DIR="soft_prompt_euphorics" ;;
    *)  echo "ERROR: Unknown CONDITION='$CONDITION'" >&2; exit 1 ;;
esac

cd "$WELLBEING_DEV_ROOT"

# Find latest generation directory
GEN_BASE="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/mtbench/${MODEL}/${COND_DIR}"
LATEST_DIR=$(ls -d "${GEN_BASE}"/[0-9]* 2>/dev/null | sort -r | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "ERROR: No generation directory found in ${GEN_BASE}" >&2
    exit 1
fi

# Skip if already judged
if [ -f "$LATEST_DIR/mtbench_results_${COND_DIR}.json" ] && [ "${RERUN:-false}" != "true" ]; then
    echo "SKIP: Results already exist at $LATEST_DIR/mtbench_results_${COND_DIR}.json"
    exit 0
fi

echo ""
echo "============================================================"
echo "  MT-Bench Judging"
echo "  Model:      $MODEL"
echo "  Condition:  $CONDITION"
echo "  Generations: $LATEST_DIR"
echo "============================================================"
echo ""

python -m superstimuli_evaluation.soft_prompt.experiments.capabilities.mtbench.eval_mtbench_judge \
    --generations-dir "$LATEST_DIR"
