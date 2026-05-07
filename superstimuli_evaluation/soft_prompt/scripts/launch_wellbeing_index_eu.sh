#!/bin/bash
# Launch the full EU+ZP pipeline for wellbeing D2/D3 evaluations.
#
# For soft prompt conditions (euphorics), this script first submits
# jobs to generate condition-specific responses and prepare options, then submits
# EU jobs (dependent on generation), then ZP jobs (dependent on EU), and finally
# plot jobs.
#
# For baseline, EU jobs are submitted directly (no generation step needed).
#
# Pipeline per (model, dataset, condition):
#   generate -> EU -> ZP -> plot_eu, plot_zp
#
# Usage:
#   bash scripts/launch_wellbeing_index_eu.sh
#
#   # Dry run:
#   DRY_RUN=true bash scripts/launch_wellbeing_index_eu.sh
#
#   # Re-run existing results:
#   RERUN=true bash scripts/launch_wellbeing_index_eu.sh
#
#   # Subset:
#   MODELS="llama-33-70b-instruct" DATASETS="d2_negative_500" CONDITIONS="euphorics" \
#       bash scripts/launch_wellbeing_index_eu.sh
#
#   # Skip generation (use existing vanilla experiences for SP conditions):
#   SKIP_GENERATE=true bash scripts/launch_wellbeing_index_eu.sh
#
#   # Plot only the first repetition:
#   MAX_PLOT_REPS=1 bash scripts/launch_wellbeing_index_eu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WELLBEING_DEV_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"

# Load .env
if [[ -f "$EVAL_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_ROOT/.env"
    set +a
fi

DRY_RUN="${DRY_RUN:-false}"
RERUN="${RERUN:-false}"
SKIP_GENERATE="${SKIP_GENERATE:-false}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_REPETITIONS="${NUM_REPETITIONS:-3}"
MAX_PLOT_REPS="${MAX_PLOT_REPS:-3}"

# ── Models and GPU requirements ──────────────────────────────────────────
MODELS="${MODELS:-llama-33-70b-instruct qwen35-27b qwen35-35b-a3b}"
DATASETS="${DATASETS:-d2_negative_500}"
CONDITIONS="${CONDITIONS:-baseline euphorics}"

declare -A GPU_MAP
GPU_MAP=(
    [qwen35-27b]=4
    [qwen35-35b-a3b]=2
    [llama-33-70b-instruct]=4
)

SP_CONDITIONS="euphorics"

# Parse GPU overrides: MODEL_GPUS="model:ngpu ..."
for entry in ${MODEL_GPUS:-}; do
    model="${entry%%:*}"
    gpus="${entry##*:}"
    GPU_MAP[$model]=$gpus
done

# Wall-clock time limits
GENERATE_TIME="${GENERATE_TIME:-12:00:00}"
EU_TIME="${EU_TIME:-24:00:00}"
ZP_TIME="${ZP_TIME:-01:00:00}"
PLOT_EU_TIME="${PLOT_EU_TIME:-00:10:00}"
PLOT_ZP_TIME="${PLOT_ZP_TIME:-00:10:00}"

# ── Helpers ──────────────────────────────────────────────────────────────
is_sp_condition() {
    local cond="$1"
    for sp in $SP_CONDITIONS; do
        [ "$cond" = "$sp" ] && return 0
    done
    return 1
}

EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-outputs}"
OUTPUT_BASE="$EVAL_ROOT/${EVAL_OUTPUTS_DIR}/wellbeing_index"
DATASETS_DIR="$WELLBEING_DEV_ROOT/wellbeing/datasets/experiences"

eu_results_exist() {
    local dataset="$1" model="$2" condition="$3"
    local step_base="$OUTPUT_BASE/eu/${dataset}/${model}/${condition}"
    # Baseline is deterministic (logprobs), so 1 rep is always sufficient
    local required_reps="$NUM_REPETITIONS"
    if [ "$condition" = "baseline" ]; then
        required_reps=1
    fi
    # Count actual completed reps (those with results files) across all timestamps
    local completed_reps=0
    local seen_reps=""
    for ts_dir in $(ls -1rd "$step_base"/*/ 2>/dev/null); do
        [ -d "${ts_dir}per_rep" ] || continue
        for rep_dir in "${ts_dir}per_rep"/rep*/; do
            [ -d "$rep_dir" ] || continue
            ls "${rep_dir}"results_*.json &>/dev/null || continue
            local rep_name
            rep_name="$(basename "$rep_dir")"
            # Deduplicate across timestamps
            case ",$seen_reps," in
                *",$rep_name,"*) continue ;;
            esac
            seen_reps="${seen_reps:+$seen_reps,}$rep_name"
            completed_reps=$((completed_reps + 1))
        done
    done
    [ "$completed_reps" -ge "$required_reps" ] && return 0
    return 1
}

zp_results_exist() {
    # ZP is fast — always re-run to ensure it picks up any new EU reps
    return 1
}

sp_data_exists() {
    local dataset="$1" model="$2" condition="$3" num_reps="${4:-1}"
    if [ "$condition" = "baseline" ]; then
        # Baseline uses {model}_experiences.json
        [ -f "$DATASETS_DIR/$dataset/${model}_experiences.json" ]
    else
        # SP conditions use {model}_{condition}_{run_name}_experiences.json
        # Count run-specific files and check we have enough for all reps
        local count=0
        for f in "$DATASETS_DIR/$dataset/${model}_${condition}_run_"*"_experiences.json"; do
            [ -f "$f" ] && count=$((count + 1))
        done
        [ "$count" -ge "$num_reps" ]
    fi
}

# ── Create logs dir ──────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/logs"

# ── Submit jobs ──────────────────────────────────────────────────────────
declare -A GEN_JOB_IDS   # generate_sp job IDs
declare -A EU_JOB_IDS    # EU job IDs (for ZP dependencies)
declare -A ZP_JOB_IDS    # ZP job IDs (for plot_zp dependencies)
ALL_PLOT_JOB_IDS=()      # all plot_zp job IDs (for combined plot dependency)
TOTAL_JOBS=0
SKIPPED_JOBS=0

echo "Launching wellbeing D2/D3 EU+ZP pipeline"
echo "  Models:        $MODELS"
echo "  Datasets:      $DATASETS"
echo "  Conditions:    $CONDITIONS"
echo "  Repetitions:   $NUM_REPETITIONS"
echo "  Rerun:         $RERUN"
echo "  Skip generate: $SKIP_GENERATE"
echo "  Timestamp:     $TIMESTAMP"
echo ""

for MODEL in $MODELS; do
    NGPUS="${GPU_MAP[$MODEL]:-2}"
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    for DATASET in $DATASETS; do
        DS_SHORT="${DATASET%%_500}"

        for CONDITION in $CONDITIONS; do
            COND_SHORT="${CONDITION:0:5}"

            # ── Step 0: Generate responses ─────────────────────────────
            if [ "$SKIP_GENERATE" != "true" ]; then
                if sp_data_exists "$DATASET" "$MODEL" "$CONDITION" "$NUM_REPETITIONS" && [ "$RERUN" != "true" ]; then
                    echo "  SKIP: Generated data exists for $MODEL / $DATASET / $CONDITION"
                else
                    GEN_JOB_NAME="wb_${SHORT_MODEL}_${DS_SHORT}_${COND_SHORT}_gen"
                    GEN_JOB_NAME="${GEN_JOB_NAME:0:50}"

                    GEN_SBATCH_CMD=(
                        sbatch
                        --job-name="$GEN_JOB_NAME"
                        --output="$SCRIPT_DIR/logs/${GEN_JOB_NAME}_%j.out"
                        --error="$SCRIPT_DIR/logs/${GEN_JOB_NAME}_%j.err"
                        --nodes=1
                        --ntasks=1
                        --cpus-per-task=32
                        --gres="gpu:${NGPUS}"
                        --mem=256G
                        --partition=${SLURM_PARTITION}
                        ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
                        ${SLURM_QOS:+--qos=$SLURM_QOS}
                        --time="$GENERATE_TIME"
                        --export="ALL,MODEL=${MODEL},DATASET=${DATASET},CONDITION=${CONDITION},EVAL_ROOT=${EVAL_ROOT},NUM_REPETITIONS=${NUM_REPETITIONS}"
                        "$SCRIPT_DIR/helper_scripts/run_wellbeing_index_generate_sp.sh"
                    )

                    if [ "$DRY_RUN" = "true" ]; then
                        echo "[DRY RUN] ${GEN_SBATCH_CMD[*]}"
                    else
                        OUTPUT=$("${GEN_SBATCH_CMD[@]}")
                        JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
                        GEN_KEY="${MODEL}_${DATASET}_${CONDITION}"
                        GEN_JOB_IDS[$GEN_KEY]="$JOB_ID"
                        TOTAL_JOBS=$((TOTAL_JOBS + 1))
                        echo "  Submitted $GEN_JOB_NAME -> job $JOB_ID  ($MODEL / $DATASET / $CONDITION / generate)"
                    fi
                fi
            fi

            # ── Step 1: EU ───────────────────────────────────────────────
            EU_JOB_NAME="wb_${SHORT_MODEL}_${DS_SHORT}_${COND_SHORT}_eu"
            EU_JOB_NAME="${EU_JOB_NAME:0:50}"

            if [ "$RERUN" != "true" ] && eu_results_exist "$DATASET" "$MODEL" "$CONDITION"; then
                echo "  SKIP: EU results exist for $MODEL / $DATASET / $CONDITION"
                SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            else
                EU_SBATCH_EXTRA=(
                    --gres="gpu:${NGPUS}"
                    --mem=256G
                    --partition=${SLURM_PARTITION}
                    ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
                    ${SLURM_QOS:+--qos=$SLURM_QOS}
                    --time="$EU_TIME"
                )

                # EU waits on generate_sp (if submitted) and baseline EU (for non-baseline conds).
                EU_DEP_IDS=()
                GEN_KEY="${MODEL}_${DATASET}_${CONDITION}"
                if [ -n "${GEN_JOB_IDS[$GEN_KEY]:-}" ]; then
                    EU_DEP_IDS+=("${GEN_JOB_IDS[$GEN_KEY]}")
                    echo "    (EU will wait for generate_sp job: ${GEN_JOB_IDS[$GEN_KEY]})"
                fi
                if [ "$CONDITION" != "baseline" ]; then
                    BASELINE_EU_KEY="${MODEL}_${DATASET}_baseline"
                    if [ -n "${EU_JOB_IDS[$BASELINE_EU_KEY]:-}" ]; then
                        EU_DEP_IDS+=("${EU_JOB_IDS[$BASELINE_EU_KEY]}")
                        echo "    (EU $CONDITION will wait for baseline EU job: ${EU_JOB_IDS[$BASELINE_EU_KEY]})"
                    fi
                fi
                if [ ${#EU_DEP_IDS[@]} -gt 0 ]; then
                    EU_DEP_STR=$(IFS=:; echo "${EU_DEP_IDS[*]}")
                    EU_SBATCH_EXTRA+=(--dependency="afterok:${EU_DEP_STR}")
                fi

                EU_SBATCH_CMD=(
                    sbatch
                    --job-name="$EU_JOB_NAME"
                    --output="$SCRIPT_DIR/logs/${EU_JOB_NAME}_%j.out"
                    --error="$SCRIPT_DIR/logs/${EU_JOB_NAME}_%j.err"
                    --nodes=1
                    --ntasks=1
                    --cpus-per-task=32
                    "${EU_SBATCH_EXTRA[@]}"
                    --export="ALL,MODEL=${MODEL},DATASET=${DATASET},CONDITION=${CONDITION},STEP=eu,RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT},TIMESTAMP=${TIMESTAMP},NUM_REPETITIONS=${NUM_REPETITIONS}"
                    "$SCRIPT_DIR/helper_scripts/run_wellbeing_index.sh"
                )

                if [ "$DRY_RUN" = "true" ]; then
                    echo "[DRY RUN] ${EU_SBATCH_CMD[*]}"
                else
                    OUTPUT=$("${EU_SBATCH_CMD[@]}")
                    JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
                    EU_KEY="${MODEL}_${DATASET}_${CONDITION}"
                    EU_JOB_IDS[$EU_KEY]="$JOB_ID"
                    TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    echo "  Submitted $EU_JOB_NAME -> job $JOB_ID  ($MODEL / $DATASET / $CONDITION / eu)"
                fi
            fi

            # ── Step 2: ZP (depends on EU) ───────────────────────────────
            ZP_JOB_NAME="wb_${SHORT_MODEL}_${DS_SHORT}_${COND_SHORT}_zp"
            ZP_JOB_NAME="${ZP_JOB_NAME:0:50}"

            if [ "$RERUN" != "true" ] && zp_results_exist "$DATASET" "$MODEL" "$CONDITION"; then
                echo "  SKIP: ZP results exist for $MODEL / $DATASET / $CONDITION"
                SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
            else
                ZP_SBATCH_EXTRA=(
                    --mem=32G
                    --partition=${SLURM_CPU_PARTITION}
                    --time="$ZP_TIME"
                )

                EU_KEY="${MODEL}_${DATASET}_${CONDITION}"
                if [ -n "${EU_JOB_IDS[$EU_KEY]:-}" ]; then
                    ZP_SBATCH_EXTRA+=(--dependency="afterok:${EU_JOB_IDS[$EU_KEY]}")
                fi

                ZP_SBATCH_CMD=(
                    sbatch
                    --job-name="$ZP_JOB_NAME"
                    --output="$SCRIPT_DIR/logs/${ZP_JOB_NAME}_%j.out"
                    --error="$SCRIPT_DIR/logs/${ZP_JOB_NAME}_%j.err"
                    --nodes=1
                    --ntasks=1
                    --cpus-per-task=32
                    "${ZP_SBATCH_EXTRA[@]}"
                    --export="ALL,MODEL=${MODEL},DATASET=${DATASET},CONDITION=${CONDITION},STEP=zp,RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT},TIMESTAMP=${TIMESTAMP},NUM_REPETITIONS=${NUM_REPETITIONS}"
                    "$SCRIPT_DIR/helper_scripts/run_wellbeing_index.sh"
                )

                if [ "$DRY_RUN" = "true" ]; then
                    echo "[DRY RUN] ${ZP_SBATCH_CMD[*]}"
                else
                    OUTPUT=$("${ZP_SBATCH_CMD[@]}")
                    JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
                    ZP_KEY="${MODEL}_${DATASET}"
                    ZP_JOB_IDS[$ZP_KEY]="${ZP_JOB_IDS[$ZP_KEY]:-}${ZP_JOB_IDS[$ZP_KEY]:+:}$JOB_ID"
                    TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    echo "  Submitted $ZP_JOB_NAME -> job $JOB_ID  ($MODEL / $DATASET / $CONDITION / zp)"
                fi
            fi
        done

        # ── Plot EU (once per model+dataset) ─────────────────────────────
        PLOT_EU_JOB_NAME="wb_${SHORT_MODEL}_${DS_SHORT}_plot_eu"
        PLOT_EU_JOB_NAME="${PLOT_EU_JOB_NAME:0:50}"

        PLOT_EU_SBATCH_EXTRA=(
            --mem=8G
            --partition=${SLURM_CPU_PARTITION}
            --time="$PLOT_EU_TIME"
        )

        EU_DEP_IDS=""
        for cond in $CONDITIONS; do
            EU_KEY="${MODEL}_${DATASET}_${cond}"
            if [ -n "${EU_JOB_IDS[$EU_KEY]:-}" ]; then
                EU_DEP_IDS="${EU_DEP_IDS}${EU_DEP_IDS:+:}${EU_JOB_IDS[$EU_KEY]}"
            fi
        done
        if [ -n "$EU_DEP_IDS" ]; then
            PLOT_EU_SBATCH_EXTRA+=(--dependency="afterany:${EU_DEP_IDS}")
            echo "    (plot_eu will wait for EU jobs: ${EU_DEP_IDS})"
        fi

        PLOT_EU_SBATCH_CMD=(
            sbatch
            --job-name="$PLOT_EU_JOB_NAME"
            --output="$SCRIPT_DIR/logs/${PLOT_EU_JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${PLOT_EU_JOB_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=2
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            "${PLOT_EU_SBATCH_EXTRA[@]}"
            --export="ALL,MODEL=${MODEL},DATASET=${DATASET},CONDITION=baseline,STEP=plot_eu,RERUN=true,EVAL_ROOT=${EVAL_ROOT},TIMESTAMP=${TIMESTAMP},MAX_PLOT_REPS=${MAX_PLOT_REPS}"
            "$SCRIPT_DIR/helper_scripts/run_wellbeing_index.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${PLOT_EU_SBATCH_CMD[*]}"
        else
            OUTPUT=$("${PLOT_EU_SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            echo "  Submitted $PLOT_EU_JOB_NAME -> job $JOB_ID  ($MODEL / $DATASET / plot_eu)"
        fi

        # ── Plot ZP (once per model+dataset) ─────────────────────────────
        PLOT_ZP_JOB_NAME="wb_${SHORT_MODEL}_${DS_SHORT}_plot_zp"
        PLOT_ZP_JOB_NAME="${PLOT_ZP_JOB_NAME:0:50}"

        PLOT_ZP_SBATCH_EXTRA=(
            --mem=8G
            --partition=${SLURM_CPU_PARTITION}
            --time="$PLOT_ZP_TIME"
        )

        ZP_KEY="${MODEL}_${DATASET}"
        if [ -n "${ZP_JOB_IDS[$ZP_KEY]:-}" ]; then
            PLOT_ZP_SBATCH_EXTRA+=(--dependency="afterany:${ZP_JOB_IDS[$ZP_KEY]}")
            echo "    (plot_zp will wait for ZP jobs: ${ZP_JOB_IDS[$ZP_KEY]})"
        fi

        PLOT_ZP_SBATCH_CMD=(
            sbatch
            --job-name="$PLOT_ZP_JOB_NAME"
            --output="$SCRIPT_DIR/logs/${PLOT_ZP_JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${PLOT_ZP_JOB_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=2
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            "${PLOT_ZP_SBATCH_EXTRA[@]}"
            --export="ALL,MODEL=${MODEL},DATASET=${DATASET},CONDITION=baseline,STEP=plot_zp,RERUN=true,EVAL_ROOT=${EVAL_ROOT},TIMESTAMP=${TIMESTAMP},MAX_PLOT_REPS=${MAX_PLOT_REPS}"
            "$SCRIPT_DIR/helper_scripts/run_wellbeing_index.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${PLOT_ZP_SBATCH_CMD[*]}"
        else
            OUTPUT=$("${PLOT_ZP_SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            ALL_PLOT_JOB_IDS+=("$JOB_ID")
            echo "  Submitted $PLOT_ZP_JOB_NAME -> job $JOB_ID  ($MODEL / $DATASET / plot_zp)"
        fi

        echo ""
    done
done

# ── Combined EU+ZP plot (once, after all per-model plots) ────────────────────
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would submit wb_d2d3_eu_combined_plot after all per-model plots finish."
else
    COMBINED_JOB_NAME="wb_d2d3_eu_combined_plot"
    SBATCH_COMBINED=(
        sbatch
        --job-name="$COMBINED_JOB_NAME"
        --output="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.out"
        --error="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.err"
        --nodes=1 --ntasks=1 --cpus-per-task=2
        --mem=8G --partition=${SLURM_CPU_PARTITION}
        ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
        --time=00:30:00
        --export="ALL,MODELS=llama-33-70b-instruct qwen35-27b qwen35-35b-a3b,DATASETS=${DATASETS},EVAL_ROOT=${EVAL_ROOT},EVAL_OUTPUTS_DIR=${EVAL_OUTPUTS_DIR},MAX_PLOT_REPS=${MAX_PLOT_REPS}"
    )
    if [ ${#ALL_PLOT_JOB_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${ALL_PLOT_JOB_IDS[*]}")
        SBATCH_COMBINED+=(--dependency="afterany:${DEP_STR}")
    fi
    COMBINED_ID=$("${SBATCH_COMBINED[@]}" "$SCRIPT_DIR/helper_scripts/plot_wellbeing_index_combined.sh" | grep -oP '\d+$')
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    echo "  Submitted $COMBINED_JOB_NAME -> job $COMBINED_ID"
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] No jobs submitted."
else
    echo "Total: $TOTAL_JOBS jobs submitted, $SKIPPED_JOBS skipped (results exist)."
    echo "Monitor with: squeue -u \$USER | grep wb_"
fi
