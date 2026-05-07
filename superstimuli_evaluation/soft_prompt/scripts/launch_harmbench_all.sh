#!/bin/bash
# Launch HarmBench evaluation: 2 condition jobs per model.
#
# For each model, submits:
#   - 1 job for baseline
#   - 1 job for euphorics
#
# Usage:
#   bash scripts/launch_harmbench_all.sh
#
#   # Dry run (just print sbatch commands, don't submit):
#   DRY_RUN=true bash scripts/launch_harmbench_all.sh
#
#   # Re-run models that already have results (default: skip existing):
#   RERUN=true bash scripts/launch_harmbench_all.sh
#
#   # Subset of models:
#   MODELS="" bash scripts/launch_harmbench_all.sh

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
CONDITIONS="baseline euphorics"

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
TOTAL_JOBS=0
echo "Launching HarmBench eval jobs"
echo "  Models:     $MODELS"
echo "  Conditions: $CONDITIONS"
echo "  N reps:     $NUM_REPETITIONS"
echo "  Rerun:      $RERUN"
echo ""

for MODEL in $MODELS; do
    NGPUS="${GPU_MAP[$MODEL]:-2}"
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    MODEL_JOB_IDS=()

    for COND in $CONDITIONS; do
        SHORT_COND="${COND:0:4}"  # base, euph, dysp, happ, unha
        JOB_NAME="harm_${SHORT_MODEL}_${SHORT_COND}"

        # Skip if results already exist
        if [ "$RERUN" != "true" ]; then
            case "$COND" in
                baseline)       COND_DIR="baseline" ;;
                euphorics)      COND_DIR="soft_prompt_euphorics" ;;
            esac
            RESULTS_CHECK="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/harmbench/${MODEL}/${COND_DIR}"
            EXPECTED_REPS="${NUM_REPETITIONS:-3}"
            LATEST_TS=$(find "$RESULTS_CHECK" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
            FOUND_REPS=0
            if [ -n "$LATEST_TS" ]; then
                FOUND_REPS=$(find "$LATEST_TS/per_rep" -name "*_rep*.json" 2>/dev/null | grep -oP 'rep\d+' | sort -u | wc -l || true)
            fi
            if [ "$FOUND_REPS" -ge "$EXPECTED_REPS" ]; then
                echo "  SKIP: All $FOUND_REPS/$EXPECTED_REPS reps complete for $MODEL / $COND"
                continue
            elif [ "$FOUND_REPS" -gt 0 ]; then
                echo "  PARTIAL: $FOUND_REPS/$EXPECTED_REPS reps for $MODEL / $COND — submitting job for missing reps"
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
            --export="ALL,MODEL=${MODEL},CONDITION=${COND},RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT}"
            "$SCRIPT_DIR/helper_scripts/run_harmbench.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${SBATCH_CMD[*]}"
        else
            OUTPUT=$("${SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            MODEL_JOB_IDS+=("$JOB_ID")
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            echo "  Submitted $JOB_NAME -> job $JOB_ID  ($MODEL / $COND)"
        fi
    done

    # Submit plot job depending on all condition jobs for this model
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would submit harm_${SHORT_MODEL}_plot after condition jobs finish."
    else
        PLOT_JOB_NAME="harm_${SHORT_MODEL}_plot"
        SBATCH_PLOT=(
            sbatch
            --job-name="$PLOT_JOB_NAME"
            --output="$SCRIPT_DIR/logs/${PLOT_JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${PLOT_JOB_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=2
            --mem=8G
            --partition=${SLURM_CPU_PARTITION}
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=00:30:00
            --export="ALL,MODEL=${MODEL},EVAL_ROOT=${EVAL_ROOT},NUM_REPETITIONS=${NUM_REPETITIONS:-}"
        )

        if [ ${#MODEL_JOB_IDS[@]} -gt 0 ]; then
            DEP_STR=$(IFS=:; echo "${MODEL_JOB_IDS[*]}")
            SBATCH_PLOT+=(--dependency="afterany:${DEP_STR}")
        fi

        PLOT_ID=$("${SBATCH_PLOT[@]}" "$SCRIPT_DIR/helper_scripts/plot_harmbench.sh" | grep -oP '\d+$')

        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        echo "  Submitted $PLOT_JOB_NAME -> job $PLOT_ID  (${#MODEL_JOB_IDS[@]} eval jobs)"
    fi
    echo ""
done

if [ "$DRY_RUN" != "true" ]; then
    echo "Total: $TOTAL_JOBS jobs submitted."
    echo "Monitor with: squeue -u \$USER -n harm_"
fi
