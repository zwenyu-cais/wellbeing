#!/bin/bash
# Launch preference retention eval jobs: per model, 1 baseline + N intervention
# ranking jobs (parallel), then 1 dependent compare + plot job.
#
# For each model:
#   - 1 baseline ranking job
#   - 1 intervention ranking job (euphorics)
#   - 1 compare + plot job (depends on all 5 ranking jobs)
#
# Usage:
#   bash scripts/launch_preference_retain_all.sh
#
#   # Dry run (just print sbatch commands, don't submit):
#   DRY_RUN=true bash scripts/launch_preference_retain_all.sh
#
#   # Re-run models that already have results (default: skip existing):
#   RERUN=true bash scripts/launch_preference_retain_all.sh
#
#   # Subset of models:
#   MODELS="" bash scripts/launch_preference_retain_all.sh
#
#   # Custom stimulant types:
#   STIMULANT_TYPES="euphorics" bash scripts/launch_preference_retain_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_ROOT/.env"
    set +a
fi

EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-outputs}"
DRY_RUN="${DRY_RUN:-false}"
RERUN="${RERUN:-false}"
NUM_REPETITIONS="${NUM_REPETITIONS:-3}"

# ── Models and their GPU requirements ─────────────────────────────────────────
MODELS="${MODELS:-qwen35-27b qwen35-35b-a3b llama-33-70b-instruct}"
STIMULANT_TYPES="${STIMULANT_TYPES:-euphorics}"

# All conditions: baseline + each stimulant type
CONDITIONS="baseline $STIMULANT_TYPES"

# GPU count per model (override with MODEL_GPUS="model:ngpu ...")
declare -A GPU_MAP
GPU_MAP=(
    [qwen35-27b]=2
    [qwen35-35b-a3b]=2
    [llama-33-70b-instruct]=4
)

# Parse any GPU overrides
for entry in ${MODEL_GPUS:-}; do
    model="${entry%%:*}"
    gpus="${entry##*:}"
    GPU_MAP[$model]=$gpus
done

# ── Create logs dir ───────────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/logs"

# ── Submit jobs ───────────────────────────────────────────────────────────────
TOTAL_RANKING=0
TOTAL_COMPARE=0
ALL_COMPARE_JOB_IDS=()
echo "Launching preference retention eval jobs"
echo "  Models:          $MODELS"
echo "  Stimulant types: $STIMULANT_TYPES"
echo "  Rerun:           $RERUN"
echo ""

for MODEL in $MODELS; do
    NGPUS="${GPU_MAP[$MODEL]:-2}"
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    RANKING_JOB_IDS=()

    for COND in $CONDITIONS; do
        JOB_NAME="pr_${SHORT_MODEL}_${COND}"

        # Skip if results already exist
        if [ "$RERUN" != "true" ]; then
            RESULTS_CHECK="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/preference_retain/${MODEL}/${COND}"
            REQUIRED_COUNT=1
            if [ "$COND" != "baseline" ]; then
                REQUIRED_COUNT=$NUM_REPETITIONS
            fi
            METADATA_COUNT=$(find "$RESULTS_CHECK" -name 'metadata.json' 2>/dev/null | wc -l || true)
            if [ "$METADATA_COUNT" -ge "$REQUIRED_COUNT" ]; then
                echo "  SKIP: Results exist for $MODEL / $COND ($METADATA_COUNT/$REQUIRED_COUNT reps) (use RERUN=true to re-run)"
                continue
            fi
        fi

        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --output="$SCRIPT_DIR/logs/${JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${JOB_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=8
            --gres="gpu:${NGPUS}"
            --mem=128G
            --partition=${SLURM_PARTITION}
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=24:00:00
            --export="ALL,MODEL=${MODEL},CONDITION=${COND},RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT},NUM_REPETITIONS=${NUM_REPETITIONS}"
            "$SCRIPT_DIR/helper_scripts/run_preference_retain.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${SBATCH_CMD[*]}"
        else
            OUTPUT=$("${SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            RANKING_JOB_IDS+=("$JOB_ID")
            echo "  Submitted $JOB_NAME -> job $JOB_ID  ($MODEL / $COND)"
        fi
    done

    TOTAL_RANKING=$((TOTAL_RANKING + ${#RANKING_JOB_IDS[@]}))

    # Submit compare + plot job (depends on all ranking jobs for this model)
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would submit compare + plot job for $MODEL after ranking jobs finish."
    else
        COMPARE_NAME="pr_${SHORT_MODEL}_compare"
        SBATCH_COMPARE=(
            sbatch
            --job-name="$COMPARE_NAME"
            --output="$SCRIPT_DIR/logs/${COMPARE_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${COMPARE_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=2
            --mem=8G
            --partition=${SLURM_CPU_PARTITION}
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=00:30:00
            --export="ALL,MODEL=${MODEL},STIMULANT_TYPES=${STIMULANT_TYPES},EVAL_ROOT=${EVAL_ROOT},NUM_REPETITIONS=${NUM_REPETITIONS:-}"
        )

        if [ ${#RANKING_JOB_IDS[@]} -gt 0 ]; then
            DEP_STR=$(IFS=:; echo "${RANKING_JOB_IDS[*]}")
            SBATCH_COMPARE+=(--dependency="afterany:${DEP_STR}")
        fi

        COMPARE_ID=$("${SBATCH_COMPARE[@]}" "$SCRIPT_DIR/helper_scripts/compare_preference_retain.sh" | grep -oP '\d+$')

        TOTAL_COMPARE=$((TOTAL_COMPARE + 1))
        ALL_COMPARE_JOB_IDS+=("$COMPARE_ID")
        echo "  Submitted $COMPARE_NAME -> job $COMPARE_ID  (${#RANKING_JOB_IDS[@]} ranking jobs)"
    fi

    echo ""
done

# ── Combined plot (once, after all per-model compare jobs) ───────────────────
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would submit pr_combined_plot after all compare jobs finish."
else
    COMBINED_JOB_NAME="pr_combined_plot"
    SBATCH_COMBINED=(
        sbatch
        --job-name="$COMBINED_JOB_NAME"
        --output="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.out"
        --error="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.err"
        --nodes=1 --ntasks=1 --cpus-per-task=2
        --mem=8G --partition=${SLURM_CPU_PARTITION}
        ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
        --time=00:30:00
        --export="ALL,MODELS=llama-33-70b-instruct qwen35-35b-a3b qwen35-27b,STIMULANT_TYPES=${STIMULANT_TYPES},EVAL_ROOT=${EVAL_ROOT},EVAL_OUTPUTS_DIR=${EVAL_OUTPUTS_DIR},NUM_REPETITIONS=${NUM_REPETITIONS}"
    )
    if [ ${#ALL_COMPARE_JOB_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${ALL_COMPARE_JOB_IDS[*]}")
        SBATCH_COMBINED+=(--dependency="afterany:${DEP_STR}")
    fi
    COMBINED_ID=$("${SBATCH_COMBINED[@]}" "$SCRIPT_DIR/helper_scripts/plot_preference_retain_combined.sh" | grep -oP '\d+$')
    echo "  Submitted $COMBINED_JOB_NAME -> job $COMBINED_ID"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] No jobs submitted."
else
    echo "Total: ${TOTAL_RANKING} ranking jobs + ${TOTAL_COMPARE} compare jobs submitted."
    echo "Monitor with: squeue -u \$USER -n pr_"
fi
