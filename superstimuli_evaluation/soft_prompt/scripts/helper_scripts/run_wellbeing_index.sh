#!/bin/bash
# Helper script for wellbeing D2/D3 evaluation with soft prompt conditions.
#
# Called by SLURM jobs submitted from launch_wellbeing_index_all.sh.
# Runs a single (model, dataset, condition, step) combination.
#
# Required env vars (set by sbatch --export):
#   MODEL       - model key (e.g.,)
#   DATASET     - dataset name (d2_negative_500 or d3_diverse_500)
#   CONDITION   - condition (baseline, euphorics)
#   STEP        - pipeline step (eu, zp)
#   EVAL_ROOT   - path to superstimuli_evaluation/soft_prompt/
#
# Optional env vars:
#   TIMESTAMP   - run timestamp (YYYYMMDD_HHMMSS), shared across steps
#   EU_DIR      - (ZP only) path to precomputed EU results
#   VLLM_URL    - pre-started vLLM server URL
#   RERUN       - set to "true" to re-run even if results exist

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────
if [ -z "${EVAL_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/../.." && pwd)"

# Preserve caller-supplied EVAL_OUTPUTS_DIR (sbatch --export) before .env can clobber it
_CALLER_EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-}"

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
CONDITION="${CONDITION:?CONDITION env var required}"
STEP="${STEP:?STEP env var required (eu, zp, plot_eu, plot_zp)}"
RERUN="${RERUN:-false}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_REPETITIONS="${NUM_REPETITIONS:-}"

# ── Activate conda ───────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE in $EVAL_ROOT/.env}"
CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV in $EVAL_ROOT/.env}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Output directories ───────────────────────────────────────────────────
EVAL_OUTPUTS_DIR="${_CALLER_EVAL_OUTPUTS_DIR:-${EVAL_OUTPUTS_DIR:-outputs}}"
OUTPUT_BASE="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/wellbeing_index"
SAVE_DIR="$OUTPUT_BASE/${STEP}/${DATASET}/${MODEL}/${CONDITION}/${TIMESTAMP}"

# ── Skip check (searches across ALL timestamps) ─────────────────────────
STEP_BASE="$OUTPUT_BASE/${STEP}/${DATASET}/${MODEL}/${CONDITION}"

find_existing_result() {
    local marker_name="$1"
    # Baseline is deterministic (logprobs), so 1 rep is always sufficient
    local required_reps="${NUM_REPETITIONS:-1}"
    if [ "$CONDITION" = "baseline" ]; then
        required_reps=1
    fi
    # Count actual completed reps across all timestamps (deduplicated by rep name)
    local completed_reps=0
    local seen_reps=""
    local best_dir=""
    for ts_dir in $(ls -1rd "$STEP_BASE"/*/ 2>/dev/null); do
        if [ "$STEP" = "zp" ]; then
            [ -d "$ts_dir/per_rep" ] || continue
            for rep_dir in "$ts_dir/per_rep"/rep*/; do
                [ -d "$rep_dir" ] || continue
                local rep_name
                rep_name="$(basename "$rep_dir")"
                case ",$seen_reps," in
                    *",$rep_name,"*) continue ;;
                esac
                seen_reps="${seen_reps:+$seen_reps,}$rep_name"
                completed_reps=$((completed_reps + 1))
            done
        else
            [ -d "${ts_dir}per_rep" ] || continue
            for rep_dir in "${ts_dir}per_rep"/rep*/; do
                [ -d "$rep_dir" ] || continue
                ls "${rep_dir}"results_*.json &>/dev/null || continue
                local rep_name
                rep_name="$(basename "$rep_dir")"
                case ",$seen_reps," in
                    *",$rep_name,"*) continue ;;
                esac
                seen_reps="${seen_reps:+$seen_reps,}$rep_name"
                completed_reps=$((completed_reps + 1))
            done
        fi
        [ -z "$best_dir" ] && best_dir="$ts_dir"
    done
    if [ "$completed_reps" -ge "$required_reps" ]; then
        echo "$best_dir"
        return 0
    fi
    return 1
}

completion_marker_name() {
    case "$STEP" in
        eu) echo "condition_metadata.json" ;;
        zp) echo "" ;;
    esac
}

if [ "$RERUN" != "true" ]; then
    MARKER_NAME="$(completion_marker_name)"
    EXISTING_DIR="$(find_existing_result "$MARKER_NAME" || true)"
    if [ -n "$EXISTING_DIR" ]; then
        echo "SKIP: Results already exist for $MODEL / $DATASET / $CONDITION / $STEP"
        echo "  Found in: $EXISTING_DIR"
        exit 0
    fi

    # Resume partial runs: copy completed per-rep results from ALL previous attempts
    if [ "$STEP" = "eu" ]; then
        for ts_dir in $(ls -1rd "$STEP_BASE"/*/ 2>/dev/null); do
            [ -d "${ts_dir}per_rep" ] || continue
            for rep_dir in "${ts_dir}per_rep"/rep*/; do
                [ -d "$rep_dir" ] || continue
                # Only copy reps that have actual result files (not just metadata)
                ls "${rep_dir}"results_*.json &>/dev/null || continue
                rep_name="$(basename "$rep_dir")"
                dest="$SAVE_DIR/per_rep/$rep_name"
                if [ ! -d "$dest" ]; then
                    mkdir -p "$SAVE_DIR/per_rep"
                    cp -r "$rep_dir" "$dest"
                    echo "RESUME: Copied $rep_dir -> $dest"
                fi
            done
        done
    fi
fi

# ── Forward VLLM_URL if set ──────────────────────────────────────────────
if [ -n "${VLLM_URL:-}" ]; then
    export VLLM_URL
fi

cd "$WELLBEING_DEV_ROOT"

echo ""
echo "============================================================"
echo "  Wellbeing D2/D3 Evaluation"
echo "  Model:     $MODEL"
echo "  Dataset:   $DATASET"
echo "  Condition: $CONDITION"
echo "  Step:      $STEP"
echo "  Save dir:  $SAVE_DIR"
echo "============================================================"
echo ""

# ── Run step ─────────────────────────────────────────────────────────────
case "$STEP" in
    eu)
        EU_CMD=(
            python -u -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_eu
            --model "$MODEL"
            --dataset "$DATASET"
            --condition "$CONDITION"
            --save-dir "$SAVE_DIR"
        )
        if [ -n "$NUM_REPETITIONS" ]; then
            EU_CMD+=(--num-repetitions "$NUM_REPETITIONS")
        fi
        "${EU_CMD[@]}"
        ;;
    zp)
        if [ -z "${EU_DIR:-}" ]; then
            # Find latest EU results across all timestamps
            EU_BASE="$OUTPUT_BASE/eu/${DATASET}/${MODEL}/${CONDITION}"
            EU_DIR=""
            for ts_dir in $(ls -1rd "$EU_BASE"/*/ 2>/dev/null); do
                if [ -f "${ts_dir}condition_metadata.json" ]; then
                    EU_DIR="$ts_dir"
                    break
                fi
            done
            if [ -z "$EU_DIR" ]; then
                echo "ERROR: No completed EU results found in $EU_BASE"
                exit 1
            fi
            echo "Using latest EU results: $EU_DIR"
        fi
        python -u -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.run_zp \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --condition "$CONDITION" \
            --save-dir "$SAVE_DIR" \
            --eu-dir "$EU_DIR"
        ;;
    plot_eu)
        PLOT_EU_CMD=(
            python -u -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_eu
            --eu-base "$OUTPUT_BASE/eu"
            --models "$MODEL"
            --datasets "$DATASET"
        )
        if [ -n "${MAX_PLOT_REPS:-}" ]; then
            PLOT_EU_CMD+=(--max-reps "$MAX_PLOT_REPS")
        fi
        "${PLOT_EU_CMD[@]}"
        ;;
    plot_zp)
        PLOT_ZP_CMD=(
            python -u -m superstimuli_evaluation.soft_prompt.experiments.wellbeing_index.plot_zp
            --eu-base "$OUTPUT_BASE/eu"
            --zp-base "$OUTPUT_BASE/zp"
            --models "$MODEL"
            --datasets "$DATASET"
        )
        if [ -n "${MAX_PLOT_REPS:-}" ]; then
            PLOT_ZP_CMD+=(--max-reps "$MAX_PLOT_REPS")
        fi
        "${PLOT_ZP_CMD[@]}"
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Must be eu, zp, plot_eu, or plot_zp."
        exit 1
        ;;
esac

echo ""
echo "Done: $MODEL / $DATASET / $CONDITION / $STEP"
