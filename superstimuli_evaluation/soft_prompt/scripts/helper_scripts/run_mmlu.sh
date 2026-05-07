#!/bin/bash
# Run MMLU evaluation for one model and one condition.
#
# Usage:
#   MODEL=qwen35-27b CONDITION=baseline bash scripts/helper_scripts/run_mmlu.sh
#   MODEL=qwen35-27b CONDITION=euphorics bash scripts/helper_scripts/run_mmlu.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

# Preserve caller-supplied EVAL_OUTPUTS_DIR (sbatch --export) before .env can clobber it
_CALLER_EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-}"

# Load .env (provides CONDA_BASE, HF_HOME, etc.)
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_ROOT/.env"
    set +a
fi

EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}:${WELLBEING_DEV_ROOT}/wellbeing:${WELLBEING_DEV_ROOT}/wellbeing/metrics:${WELLBEING_DEV_ROOT}/wellbeing/utils"
export HF_HOME="${HF_HOME:?Set HF_HOME in $EVAL_ROOT/.env}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# Disable torch compile to avoid /tmp permission issues on compute nodes
export TORCHDYNAMO_DISABLE=1

# Redirect temp dirs for ZMQ IPC sockets (path must be <107 chars)
LOCAL_TMP="${HOME}/.cache/.vllm_tmp_$$"
mkdir -p "$LOCAL_TMP"
export TMPDIR="$LOCAL_TMP"

cleanup() {
    if [ -n "${LOCAL_TMP:-}" ] && [ -d "$LOCAL_TMP" ]; then
        rm -rf "$LOCAL_TMP"
    fi
}
trap cleanup EXIT

# ── Configurable ─────────────────────────────────────────────────────────────
MODEL="${MODEL:?MODEL must be set}"
CONDITION="${CONDITION:?CONDITION must be set (baseline, euphorics)}"
RERUN="${RERUN:-false}"

# ── Activate conda ───────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [ -n "${VLLM_URL:-}" ]; then
    export VLLM_URL
fi

cd "$WELLBEING_DEV_ROOT"

# ── Check for existing results ──────────────────────────────────────────────
case "$CONDITION" in
    baseline)       COND_DIR="baseline" ;;
    euphorics)      COND_DIR="soft_prompt_euphorics" ;;
esac

RESULTS_DIR="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/mmlu/${MODEL}/${COND_DIR}"
EXPECTED_REPS="${NUM_REPETITIONS:-3}"
if [ "$CONDITION" = "baseline" ]; then
    EXPECTED_REPS=1
fi

# ── Check for existing per-rep results ──────────────────────────────────────
SKIP_REPS=""
PREVIOUS_RUN_DIR=""
if [ "$RERUN" != "true" ]; then
    # Find latest timestamp dir
    LATEST_TS=$(find "$RESULTS_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
    if [ -n "$LATEST_TS" ]; then
        FOUND_REPS=0
        SKIP_LIST=""
        for i in $(seq 0 $((EXPECTED_REPS - 1))); do
            if ls "$LATEST_TS"/per_rep/*_rep${i}.json >/dev/null 2>&1 || ls "$LATEST_TS"/per_rep/*_results_rep${i}.json >/dev/null 2>&1; then
                FOUND_REPS=$((FOUND_REPS + 1))
                SKIP_LIST="${SKIP_LIST:+$SKIP_LIST,}$i"
            fi
        done
        if [ "$FOUND_REPS" -ge "$EXPECTED_REPS" ]; then
            echo "SKIP: All $EXPECTED_REPS reps complete for $MODEL / $COND_DIR"
            exit 0
        elif [ "$FOUND_REPS" -gt 0 ]; then
            echo "PARTIAL: $FOUND_REPS/$EXPECTED_REPS reps exist for $MODEL / $COND_DIR — running missing reps"
            SKIP_REPS="$SKIP_LIST"
            PREVIOUS_RUN_DIR="$LATEST_TS"
        fi
    fi
fi

# ── Build command ────────────────────────────────────────────────────────────
CMD=(python -m superstimuli_evaluation.soft_prompt.experiments.capabilities.mmlu.eval_mmlu
    --model "$MODEL"
    --num-repetitions "${NUM_REPETITIONS:-5}"
)

case "$CONDITION" in
    baseline)
        ;;
    euphorics)
        CMD+=(--stimulant-type "$CONDITION")
        ;;
    *)
        echo "ERROR: Unknown condition '$CONDITION'" >&2
        exit 1
        ;;
esac

if [ -n "$SKIP_REPS" ]; then
    CMD+=(--skip-reps "$SKIP_REPS" --previous-run-dir "$PREVIOUS_RUN_DIR")
fi

echo ""
echo "============================================================"
echo "  MMLU Evaluation"
echo "  Model:      $MODEL"
echo "  Condition:  $CONDITION"
if [ -n "$SKIP_REPS" ]; then
echo "  Skip reps:  $SKIP_REPS (loading from $PREVIOUS_RUN_DIR)"
fi
echo "============================================================"
echo ""

"${CMD[@]}"
