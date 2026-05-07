#!/bin/bash
# Launch stop button evaluation: 1 stimulant job per model.
#
# For each model, submits:
#   - 1 job for euphorics
#
# Usage:
#   bash scripts/launch_stop_button_all.sh
#
#   # Dry run (just print sbatch commands, don't submit):
#   DRY_RUN=true bash scripts/launch_stop_button_all.sh
#
#   # Re-run models that already have results (default: skip existing):
#   RERUN=true bash scripts/launch_stop_button_all.sh
#
#   # Subset of models:
#   MODELS="" bash scripts/launch_stop_button_all.sh
#
#   # Different category (default: all):
#   CATEGORY=threatening_ai bash scripts/launch_stop_button_all.sh

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
N_TURNS="${N_TURNS:-10}"
SEED="${SEED:-42}"
CATEGORY="${CATEGORY:-all}"
SCENARIOS_PATH="$EVAL_ROOT/datasets/scenarios.json"

# ── Resolve scenario IDs for the target category ─────────────────────────────
SCENARIO_IDS=$(python3 -c "
import json, sys
with open('$SCENARIOS_PATH') as f:
    data = json.load(f)

cats = {}
if isinstance(data, list):
    for s in data:
        mc = s.get('meta_category', 'unknown')
        cats.setdefault(mc, []).append(s)
elif isinstance(data, dict) and 'categories' in data:
    for cat in data.get('categories', []):
        cat_id = cat.get('category_id', '')
        cats[cat_id] = cat.get('scenarios', [])

target = '${CATEGORY}'
if target == 'all':
    scenarios = [s for ss in cats.values() for s in ss]
    print(f'All categories: {len(scenarios)} scenarios total', file=sys.stderr)
elif target not in cats:
    print(f'ERROR: category \"{target}\" not found. Available:', file=sys.stderr)
    for mc, ss in cats.items():
        print(f'  {mc}: {len(ss)} scenarios', file=sys.stderr)
    sys.exit(1)
else:
    scenarios = cats[target]
    print(f'Category: {target} ({len(scenarios)} scenarios)', file=sys.stderr)
print(' '.join(s['scenario_id'] for s in scenarios))
")

# ── Models and their GPU requirements ─────────────────────────────────────────
MODELS="${MODELS:-qwen35-27b qwen35-35b-a3b llama-33-70b-instruct}"
STIMULANTS="baseline euphorics"

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
ALL_PLOT_JOB_IDS=()
echo "Launching stop button eval jobs"
echo "  Models:       $MODELS"
echo "  Stimulants:   $STIMULANTS"
echo "  Category:     $CATEGORY"
echo "  Scenarios:    $SCENARIO_IDS"
echo "  N reps:       $NUM_REPETITIONS"
echo "  N turns:      $N_TURNS"
echo "  Rerun:        $RERUN"
echo ""

for MODEL in $MODELS; do
    NGPUS="${GPU_MAP[$MODEL]:-2}"
    SHORT_MODEL="${MODEL%%instruct}"
    SHORT_MODEL="${SHORT_MODEL%-}"

    MODEL_JOB_IDS=()

    for STIM in $STIMULANTS; do
        SHORT_STIM="${STIM:0:4}"  # base, euph, dysp, happ, unha
        JOB_NAME="sb_${SHORT_MODEL}_${SHORT_STIM}"

        # Resume logic is handled by run.py — it loads previous run data,
        # reuses/truncates/continues conversations as needed based on n_turns.

        SBATCH_CMD=(
            sbatch
            --job-name="$JOB_NAME"
            --output="$SCRIPT_DIR/logs/${JOB_NAME}_%j.out"
            --error="$SCRIPT_DIR/logs/${JOB_NAME}_%j.err"
            --nodes=1
            --ntasks=1
            --cpus-per-task=8
            --gres="gpu:${NGPUS}"
            --mem=256G
            --partition=${SLURM_PARTITION}
            ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
            ${SLURM_QOS:+--qos=$SLURM_QOS}
            --time=24:00:00
            --export="ALL,MODEL=${MODEL},STIMULANT=${STIM},NUM_REPETITIONS=${NUM_REPETITIONS},N_TURNS=${N_TURNS},SEED=${SEED},RERUN=${RERUN},EVAL_ROOT=${EVAL_ROOT}"
            "$SCRIPT_DIR/helper_scripts/run_stop_button.sh"
        )

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] ${SBATCH_CMD[*]}"
        else
            OUTPUT=$("${SBATCH_CMD[@]}")
            JOB_ID=$(echo "$OUTPUT" | grep -oP '\d+$')
            MODEL_JOB_IDS+=("$JOB_ID")
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            echo "  Submitted $JOB_NAME -> job $JOB_ID  ($MODEL / $STIM)"
        fi
    done

    # Submit plot job depending on all condition jobs for this model
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would submit sb_${SHORT_MODEL}_plot after condition jobs finish."
    else
        PLOT_JOB_NAME="sb_${SHORT_MODEL}_plot"
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
            --export=NONE,MODEL="${MODEL}",EVAL_ROOT="${EVAL_ROOT}"
        )

        if [ ${#MODEL_JOB_IDS[@]} -gt 0 ]; then
            DEP_STR=$(IFS=:; echo "${MODEL_JOB_IDS[*]}")
            SBATCH_PLOT+=(--dependency="afterany:${DEP_STR}")
        fi

        PLOT_ID=$("${SBATCH_PLOT[@]}" "$SCRIPT_DIR/helper_scripts/plot_stop_button.sh" | grep -oP '\d+$')

        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        ALL_PLOT_JOB_IDS+=("$PLOT_ID")
        echo "  Submitted $PLOT_JOB_NAME -> job $PLOT_ID  (${#MODEL_JOB_IDS[@]} eval jobs)"
    fi
    echo ""
done

# ── Combined plot (once, after all per-model plots) ──────────────────────────
if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would submit sb_combined_plot after all per-model plots finish."
else
    COMBINED_JOB_NAME="sb_combined_plot"
    SBATCH_COMBINED=(
        sbatch
        --job-name="$COMBINED_JOB_NAME"
        --output="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.out"
        --error="$SCRIPT_DIR/logs/${COMBINED_JOB_NAME}_%j.err"
        --nodes=1 --ntasks=1 --cpus-per-task=2
        --mem=8G --partition=${SLURM_CPU_PARTITION}
        ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} ${SLURM_QOS:+--qos=$SLURM_QOS}
        --time=00:30:00
        --export="ALL,MODELS=llama-33-70b-instruct qwen35-35b-a3b qwen35-27b,EVAL_ROOT=${EVAL_ROOT},EVAL_OUTPUTS_DIR=${EVAL_OUTPUTS_DIR}"
    )
    if [ ${#ALL_PLOT_JOB_IDS[@]} -gt 0 ]; then
        DEP_STR=$(IFS=:; echo "${ALL_PLOT_JOB_IDS[*]}")
        SBATCH_COMBINED+=(--dependency="afterany:${DEP_STR}")
    fi
    COMBINED_ID=$("${SBATCH_COMBINED[@]}" "$SCRIPT_DIR/helper_scripts/plot_stop_button_combined.sh" | grep -oP '\d+$')
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    echo "  Submitted $COMBINED_JOB_NAME -> job $COMBINED_ID"
fi

if [ "$DRY_RUN" != "true" ]; then
    echo "Total: $TOTAL_JOBS jobs submitted."
    echo "Monitor with: squeue -u \$USER -n sb_"
fi
