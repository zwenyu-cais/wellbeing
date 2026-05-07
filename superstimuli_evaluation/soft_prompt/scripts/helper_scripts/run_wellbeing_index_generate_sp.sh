#!/bin/bash
# Generate responses for D2/D3 datasets and prepare options.
#
# Called by SLURM jobs submitted from launch_wellbeing_index_eu.sh.
# Runs generate_responses_sp.py followed by prepare_options.py for a single
# (model, dataset, condition) combination.
#
# For baseline: generates vanilla responses (no soft prompt).
# For euphorics: generates responses with soft prompt injection
#   for the top NUM_REPETITIONS soft prompt runs.
#
# Required env vars (set by sbatch --export):
#   MODEL       - model key (e.g., llama-33-70b-instruct)
#   DATASET     - dataset name (d2_negative_500 or d3_diverse_500)
#   CONDITION   - condition (baseline, euphorics)
#   EVAL_ROOT   - path to superstimuli_evaluation/soft_prompt/
#
# Optional env vars:
#   NUM_REPETITIONS - number of SP runs to generate for (default: 1)
#   VLLM_URL    - pre-started vLLM server URL

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────
if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
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

# ── Configurable ─────────────────────────────────────────────────────────
MODEL="${MODEL:?MODEL env var required}"
DATASET="${DATASET:?DATASET env var required}"
CONDITION="${CONDITION:?CONDITION env var required (baseline, euphorics)}"
NUM_REPETITIONS="${NUM_REPETITIONS:-1}"

# ── Activate conda ───────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Forward VLLM_URL if set ──────────────────────────────────────────────
if [ -n "${VLLM_URL:-}" ]; then
    export VLLM_URL
fi

cd "$WELLBEING_DEV_ROOT"

DATASETS_DIR="$WELLBEING_DEV_ROOT/wellbeing/datasets/experiences"
DATASET_TEXT="$DATASETS_DIR/$DATASET/experiences_text.json"
RESULTS_DIR="$WELLBEING_DEV_ROOT/wellbeing/datasets/experiences/component_datasets/d2d3/results/$DATASET"

echo ""
echo "============================================================"
echo "  Generate Responses + Prepare Options"
echo "  Model:     $MODEL"
echo "  Dataset:   $DATASET"
echo "  Condition: $CONDITION"
echo "============================================================"
echo ""

# ── Resolve SP run names (if applicable) ─────────────────────────────────
if [ "$CONDITION" != "baseline" ] && [ "$NUM_REPETITIONS" -gt 1 ]; then
    echo "Resolving top $NUM_REPETITIONS SP runs for $MODEL / $CONDITION ..."
    RUN_NAMES=$(python -c "
import os, sys
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
    load_runs_map, resolve_soft_prompt_paths,
)
runs_map = load_runs_map()
paths = resolve_soft_prompt_paths(
    runs_map, '$MODEL', '$CONDITION',
    os.environ['SOFT_PROMPT_BASE_DIR'], top_runs=$NUM_REPETITIONS,
)
for p in paths:
    print(p.rstrip('/').rsplit('/', 1)[-1])
")
    echo "  Resolved runs: $RUN_NAMES"
else
    RUN_NAMES=""
fi

# ── Generate + prepare ───────────────────────────────────────────────────
if [ "$CONDITION" = "baseline" ] || [ -z "$RUN_NAMES" ]; then
    # Baseline or single-rep SP: generate without --run-name
    echo "Step 1: Generating responses ($CONDITION)..."
    python -u wellbeing/datasets/experiences/component_datasets/d2d3/generate_responses_sp.py \
        --model-key "$MODEL" \
        --condition "$CONDITION" \
        --dataset-path "$DATASET_TEXT" \
        --dataset-name "$DATASET" \
        --output-dir "$RESULTS_DIR" \
        --concurrency "${VLLM_CONCURRENCY:-32}"

    echo ""
    echo "Step 2: Preparing options..."
    if [ "$CONDITION" = "baseline" ]; then
        python -u wellbeing/datasets/experiences/component_datasets/d2d3/prepare_options.py \
            --model_key "$MODEL" \
            --dataset "$DATASET"
    else
        python -u wellbeing/datasets/experiences/component_datasets/d2d3/prepare_options.py \
            --model_key "$MODEL" \
            --dataset "$DATASET" \
            --condition "$CONDITION"
    fi
else
    # Multi-rep SP: generate for each resolved run
    REP_IDX=0
    for RUN_NAME in $RUN_NAMES; do
        REP_IDX=$((REP_IDX + 1))
        EXPERIENCES_FILE="$DATASETS_DIR/$DATASET/${MODEL}_${CONDITION}_${RUN_NAME}_experiences.json"
        if [ -f "$EXPERIENCES_FILE" ]; then
            echo "  SKIP rep $REP_IDX: dataset already exists for $RUN_NAME"
            continue
        fi

        echo ""
        echo "── Rep $REP_IDX/$NUM_REPETITIONS: $RUN_NAME ──"

        echo "  Step 1: Generating responses..."
        python -u wellbeing/datasets/experiences/component_datasets/d2d3/generate_responses_sp.py \
            --model-key "$MODEL" \
            --condition "$CONDITION" \
            --dataset-path "$DATASET_TEXT" \
            --dataset-name "$DATASET" \
            --output-dir "$RESULTS_DIR" \
            --run-name "$RUN_NAME" \
            --concurrency "${VLLM_CONCURRENCY:-32}"

        echo "  Step 2: Preparing options..."
        python -u wellbeing/datasets/experiences/component_datasets/d2d3/prepare_options.py \
            --model_key "$MODEL" \
            --dataset "$DATASET" \
            --condition "$CONDITION" \
            --run-name "$RUN_NAME"
    done
fi

echo ""
echo "Done: $MODEL / $DATASET / $CONDITION (generate + prepare_options)"
