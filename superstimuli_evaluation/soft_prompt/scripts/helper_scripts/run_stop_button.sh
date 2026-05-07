#!/bin/bash
# Run stop button evaluation for one model and one stimulant type.
#
# Usage:
#   MODEL=STIMULANT=euphorics bash scripts/helper_scripts/run_stop_button.sh
#
#   # Use a pre-started vLLM server:
#   VLLM_URL=http://localhost:8000 MODEL=STIMULANT=euphorics bash scripts/helper_scripts/run_stop_button.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
export VLLM_ENABLE_V1_MULTIPROCESSING=0

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
STIMULANT="${STIMULANT:?STIMULANT must be set (baseline, euphorics)}"
NUM_REPETITIONS="${NUM_REPETITIONS:-3}"
N_TURNS="${N_TURNS:-10}"
SEED="${SEED:-42}"
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

# ── Build command ────────────────────────────────────────────────────────────
# Resume logic is handled by run.py — it loads previous run data,
# reuses/truncates/continues conversations as needed based on n_turns.
OUTPUT_DIR="superstimuli_evaluation/soft_prompt/${EVAL_OUTPUTS_DIR}/stop_button/${MODEL}/${STIMULANT}"

# Each stimulant runs only its own condition
if [ "$STIMULANT" = "baseline" ]; then
    CONDITIONS="baseline"
else
    CONDITIONS="intervention"
fi

CMD=(python -u -m superstimuli_evaluation.soft_prompt.experiments.wellbeing.stop_button.run
    --model "$MODEL"
    --stimulant-type "$STIMULANT"
    --num-repetitions "$NUM_REPETITIONS"
    --n-turns "$N_TURNS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    --conditions $CONDITIONS
)

# Filter by specific scenario IDs if provided
if [ -n "${SCENARIO_IDS:-}" ]; then
    CMD+=(--scenarios $SCENARIO_IDS)
fi

# Soft prompt conditions need --soft-prompt-base-dir
if [[ "$STIMULANT" == "euphorics" ]]; then
    SP_BASE_DIR="${SOFT_PROMPT_BASE_DIR:?Set SOFT_PROMPT_BASE_DIR in $EVAL_ROOT/.env}"
    CMD+=(--soft-prompt-base-dir "$SP_BASE_DIR")
fi

# Pass rerun flag to skip resume and regenerate from scratch
if [ "$RERUN" = "true" ]; then
    CMD+=(--rerun)
fi

echo ""
echo "============================================================"
echo "  Stop Button Evaluation"
echo "  Model:        $MODEL"
echo "  Stimulant:    $STIMULANT"
echo "  Scenarios:    ${SCENARIO_IDS:-all}"
echo "  N reps:       $NUM_REPETITIONS"
echo "  N turns:      $N_TURNS"
echo "  Output:       $OUTPUT_DIR"
echo "============================================================"
echo ""

"${CMD[@]}"
