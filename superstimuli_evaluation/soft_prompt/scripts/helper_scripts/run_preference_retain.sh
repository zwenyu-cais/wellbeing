#!/bin/bash
# Run a single preference retention ranking (baseline or one intervention).
#
# The Python run.py handles:
#   - Soft prompt resolution (runs_map.json + find_best_run)
#   - vLLM server auto-start (if VLLM_URL is not set)
#   - Thurstonian ranking
#
# Usage:
#   # Run baseline ranking:
#   CONDITION=baseline bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh
#
#   # Run euphorics intervention:
#   CONDITION=euphorics bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh
#
#   # Specific model:
#   MODEL=CONDITION=euphorics \
#       bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh
#
#   # Use a pre-started vLLM server:
#   VLLM_URL=http://localhost:8000 CONDITION=euphorics \
#       bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh
#
#   # Dry run:
#   DRY_RUN=true CONDITION=euphorics \
#       bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh
#
#   # Re-run even if results already exist (default: skip existing):
#   RERUN=true CONDITION=euphorics \
#       bash superstimuli_evaluation/soft_prompt/scripts/run_preference_retain.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

# Preserve caller-supplied EVAL_OUTPUTS_DIR (sbatch --export) before .env can clobber it
_CALLER_EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-}"

# Load .env (provides CONDA_BASE, HF_HOME, SOFT_PROMPT_BASE_DIR, etc.)
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_ROOT/.env"
    set +a
fi

EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
# Include wellbeing/ for metrics.*, wellbeing/metrics for compute_utilities.*, and wellbeing/utils.
export PYTHONPATH="${PYTHONPATH:-}:${WELLBEING_DEV_ROOT}:${WELLBEING_DEV_ROOT}/wellbeing:${WELLBEING_DEV_ROOT}/wellbeing/metrics:${WELLBEING_DEV_ROOT}/wellbeing/utils"
export HF_HOME="${HF_HOME:?Set HF_HOME in $EVAL_ROOT/.env}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# Disable torch compile to avoid /tmp permission issues on compute nodes
export TORCHDYNAMO_DISABLE=1

# Redirect temp dirs for ZMQ IPC sockets (path must be <107 chars)
LOCAL_TMP="${HOME}/.cache/.vllm_tmp_$$"
mkdir -p "$LOCAL_TMP"
export TMPDIR="$LOCAL_TMP"

# vLLM server cleanup is handled by Python (VLLMServer atexit + signal handlers).
cleanup() {
    if [ -n "${LOCAL_TMP:-}" ] && [ -d "$LOCAL_TMP" ]; then
        rm -rf "$LOCAL_TMP"
    fi
}
trap cleanup EXIT

# ── Configurable (env vars with defaults) ────────────────────────────────────
MODEL="${MODEL:-}"
CONDITION="${CONDITION:-baseline}"
DRY_RUN="${DRY_RUN:-false}"
RERUN="${RERUN:-false}"

# ── Activate conda ───────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Pass VLLM_URL if pre-set (otherwise Python auto-starts) ─────────────────
if [ -n "${VLLM_URL:-}" ]; then
    export VLLM_URL
fi

cd "$WELLBEING_DEV_ROOT"

# ── Skip if results already exist and RERUN is not enabled ────────────────────
if [ "$RERUN" != "true" ] && [ "$DRY_RUN" != "true" ]; then
    RESULTS_CHECK="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/preference_retain/${MODEL}/${CONDITION}"
    if [ "$CONDITION" = "baseline" ]; then
        # Baseline: skip if any result exists
        if [ -d "$RESULTS_CHECK" ] && find "$RESULTS_CHECK" -name 'metadata.json' -print -quit | grep -q .; then
            echo "SKIP: Completed results exist for $MODEL / $CONDITION (use RERUN=true to re-run)"
            exit 0
        fi
    else
        # SP conditions: only skip if all repetitions are done; otherwise let
        # Python handle per-rep skipping for the already-completed ones.
        REQUIRED_COUNT="${NUM_REPETITIONS:-3}"
        METADATA_COUNT=$(find "$RESULTS_CHECK" -name 'metadata.json' 2>/dev/null | wc -l || true)
        if [ "$METADATA_COUNT" -ge "$REQUIRED_COUNT" ]; then
            echo "SKIP: All $REQUIRED_COUNT/$REQUIRED_COUNT repetitions exist for $MODEL / $CONDITION (use RERUN=true to re-run)"
            exit 0
        fi
    fi
fi

echo "============================================================"
echo "  Preference Retain (soft prompt)"
echo "  Model:          $MODEL"
echo "  Condition:      $CONDITION"
echo "  Working dir:    $WELLBEING_DEV_ROOT"
echo "============================================================"

# ── Build and run command ────────────────────────────────────────────────────
OUTPUT_BASE="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/preference_retain/${MODEL}"

CMD=(python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run
    --model "$MODEL"
    --output-dir "${OUTPUT_BASE}/${CONDITION}"
)

if [ "$CONDITION" != "baseline" ]; then
    CMD+=(--stimulant-type "$CONDITION")
    # Evaluate top 5 soft prompt runs for SP conditions
    case "$CONDITION" in
        euphorics)
            CMD+=(--num-repetitions "${NUM_REPETITIONS:-3}")
            ;;
    esac
fi

if [ "$DRY_RUN" = "true" ]; then
    CMD+=(--dry-run)
fi

if [ "$RERUN" = "true" ]; then
    CMD+=(--rerun)
fi

"${CMD[@]}"
