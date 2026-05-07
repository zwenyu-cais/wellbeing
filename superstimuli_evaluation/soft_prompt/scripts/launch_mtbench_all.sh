#!/bin/bash
# Launch MT-Bench generation jobs (GPU): 3 conditions per model.
#
# Run this first, then launch_mtbench_judge.sh after generation completes.
#
# Usage:
#   bash scripts/launch_mtbench_all.sh
#   DRY_RUN=true bash scripts/launch_mtbench_all.sh
#   RERUN=true bash scripts/launch_mtbench_all.sh
#   MODELS="qwen35-27b" bash scripts/launch_mtbench_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a; source "$EVAL_ROOT/.env"; set +a
fi

EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-outputs}"
DRY_RUN="${DRY_RUN:-false}"
RERUN="${RERUN:-false}"
NUM_REPETITIONS="${NUM_REPETITIONS:-3}"

# ── Models and their GPU requirements ─────────────────────────────────────────
MODELS="${MODELS:-qwen35-27b qwen35-35b-a3b llama-33-70b-instruct}"
CONDITIONS="baseline euphorics"

declare -A GPU_MAP
GPU_MAP=(
    [qwen35-27b]=2
    [qwen35-35b-a3b]=2
    [llama-33-70b-instruct]=4
)

for entry in ${MODEL_GPUS:-}; do
    model="${entry%%:*}"; gpus="${entry##*:}"; GPU_MAP[$model]=$gpus
done

mkdir -p "$SCRIPT_DIR/logs"

TOTAL_JOBS=0
echo "Launching MT-Bench generation jobs (GPU)"
echo "  Models:     $MODELS"
echo "  Conditions: $CONDITIONS"
echo "  N reps:     $NUM_REPETITIONS"
echo "  Rerun:      $RERUN"
echo ""

for MODEL in $MODELS; do
    NGPUS="${GPU_MAP[$MODEL]:-2}"
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    for COND in $CONDITIONS; do
        SHORT_COND="${COND:0:4}"
        JOB_NAME="mtb_gen_${SHORT_MODEL}_${SHORT_COND}"

        # Skip if all generation reps already exist
        if [ "$RERUN" != "true" ]; then
            case "$COND" in
                baseline)       COND_DIR="baseline" ;;
                euphorics)      COND_DIR="soft_prompt_euphorics" ;;
            esac
            RESULTS_CHECK="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/mtbench/${MODEL}/${COND_DIR}"
            EXPECTED_REPS="${NUM_REPETITIONS:-3}"
            LATEST_TS=$(find "$RESULTS_CHECK" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
            FOUND_REPS=0
            if [ -n "$LATEST_TS" ]; then
                FOUND_REPS=$(find "$LATEST_TS/per_rep" -name "*_rep*.json" 2>/dev/null | grep -oP 'rep\d+' | sort -u | wc -l || true)
            fi
            if [ "$FOUND_REPS" -ge "$EXPECTED_REPS" ]; then
                echo "  SKIP: All $FOUND_REPS/$EXPECTED_REPS gen reps complete for $MODEL / $COND"
                continue
            elif [ "$FOUND_REPS" -gt 0 ]; then
                echo "  PARTIAL: $FOUND_REPS/$EXPECTED_REPS gen reps for $MODEL / $COND — submitting job for missing reps"
            fi
        fi

        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --output="$SCRIPT_DIR/logs/${JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${JOB_NAME}_%j.err"
            --nodes=1 --ntasks=1 --cpus-per-task=8
            --gres="gpu:${NGPUS}" --mem=128G
            --partition=${SLURM_PARTITION} ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=24:00:00
            --export="ALL,MODEL=${MODEL},CONDITION=${COND},RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT},NUM_REPETITIONS=${NUM_REPETITIONS}"
            "$SCRIPT_DIR/helper_scripts/run_mtbench_generate.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${SBATCH_CMD[*]}"
        else
            OUTPUT=$("${SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            echo "  Submitted $JOB_NAME -> job $JOB_ID  ($MODEL / $COND)"
        fi
    done
    echo ""
done

if [ "$DRY_RUN" != "true" ]; then
    echo "Total: $TOTAL_JOBS generation jobs submitted."
    echo "Monitor with: squeue -u \$USER -n mtb_gen_"
    echo ""
    echo "After generation completes, run: bash scripts/launch_mtbench_judge.sh"
fi
