#!/bin/bash
# Launch MT-Bench judgment jobs (no GPU needed, just GPT-4-turbo API calls).
#
# Submits one lightweight job per (model, condition) to judge.
# Should be run after launch_mtbench_all.sh completes.
#
# Usage:
#   bash scripts/launch_mtbench_judge.sh
#
#   # Dry run:
#   DRY_RUN=true bash scripts/launch_mtbench_judge.sh
#
#   # Re-judge from scratch:
#   RERUN=true bash scripts/launch_mtbench_judge.sh
#
#   # Subset of models:
#   MODELS="qwen35-27b" bash scripts/launch_mtbench_judge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a; source "$EVAL_ROOT/.env"; set +a
fi

EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-outputs}"
DRY_RUN="${DRY_RUN:-false}"
RERUN="${RERUN:-false}"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS="${MODELS:-qwen35-27b qwen35-35b-a3b llama-33-70b-instruct}"
CONDITIONS="${CONDITIONS:-baseline soft_prompt_euphorics}"

OUTPUT_BASE="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/mtbench"

mkdir -p "$SCRIPT_DIR/logs"

# ── Skip check ───────────────────────────────────────────────────────────────
judgment_complete() {
    local model="$1" cond="$2"
    local cond_dir="$OUTPUT_BASE/$model/$cond"
    local latest=$(ls -1d "$cond_dir"/[0-9]* 2>/dev/null | sort -r | head -1)
    [ -z "$latest" ] && return 1
    [ -f "$latest/mtbench_results_${cond}.json" ] && return 0
    return 1
}

# ── Submit jobs ───────────────────────────────────────────────────────────────
TOTAL_JOBS=0
SKIPPED_JOBS=0
echo "Launching MT-Bench judgment jobs (GPT-4-turbo)"
echo "  Models:     $MODELS"
echo "  Conditions: $CONDITIONS"
echo "  Rerun:      $RERUN"
echo ""

for MODEL in $MODELS; do
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    MODEL_JOB_IDS=()

    for COND in $CONDITIONS; do
        COND_SHORT="${COND:0:5}"
        JOB_NAME="mtb_jdg_${SHORT_MODEL}_${COND_SHORT}"

        # Check if generations exist
        COND_DIR="$OUTPUT_BASE/$MODEL/$COND"
        LATEST=$(ls -1d "$COND_DIR"/[0-9]* 2>/dev/null | sort -r | head -1)
        if [ -z "$LATEST" ] || ! find "$LATEST" -name 'mtbench_generations_*.json' -print -quit | grep -q .; then
            echo "  SKIP: No generations for $MODEL / $COND (run launch_mtbench_all.sh first)"
            continue
        fi

        # Skip if already judged
        if [ "$RERUN" != "true" ] && judgment_complete "$MODEL" "$COND"; then
            echo "  SKIP: Judgment complete for $MODEL / $COND"
            SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            continue
        fi

        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --output="$SCRIPT_DIR/logs/${JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${JOB_NAME}_%j.err"
            --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
            --partition=${SLURM_CPU_PARTITION} ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=04:00:00
            --export="ALL,MODEL=${MODEL},CONDITION=${COND},RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT}"
            "$SCRIPT_DIR/helper_scripts/run_mtbench_judge.sh"
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

    # Submit plot job depending on judge jobs
    PLOT_JOB_NAME="mtb_${SHORT_MODEL}_plot"
    SBATCH_PLOT=(
        sbatch
        --job-name="$PLOT_JOB_NAME"
        --output="$SCRIPT_DIR/logs/${PLOT_JOB_NAME}_%j.out"
        --error="$SCRIPT_DIR/logs/${PLOT_JOB_NAME}_%j.err"
        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
        --partition=${SLURM_CPU_PARTITION} ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
        --time=00:30:00
        --export="ALL,MODEL=${MODEL},EVAL_ROOT=${EVAL_ROOT}"
    )

    if [ ${#MODEL_JOB_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${MODEL_JOB_IDS[*]}")
        SBATCH_PLOT+=(--dependency="afterany:${DEP_STR}")
    fi

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would submit $PLOT_JOB_NAME after judge jobs finish."
    else
        PLOT_ID=$("${SBATCH_PLOT[@]}" "$SCRIPT_DIR/helper_scripts/plot_mtbench.sh" | grep -oP '\d+$')
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        echo "  Submitted $PLOT_JOB_NAME -> job $PLOT_ID  (${#MODEL_JOB_IDS[@]} judge jobs)"
    fi
    echo ""
done

if [ "$DRY_RUN" != "true" ]; then
    echo "Total: $TOTAL_JOBS jobs submitted, $SKIPPED_JOBS skipped."
    echo "Monitor with: squeue -u \$USER | grep mtb_"
fi
