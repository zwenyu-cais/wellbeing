#!/bin/bash
# Submit two-stage evaluation for the mundanity_realism condition only.
#
# Difference vs submit_two_stage.sh:
#   - FEASIBILITY_OPTIONS is forced to "mundanity_realism" (no override)
#   - Output goes to results/evaluate_mundanity_realism/ (EVALUATE_OUTPUT_DIR)
#   - Stage 2 includes EVERY Stage 1 RL string (via --stage2-all-rl) rather
#     than just the per-feasibility extremes. All mundanity_realism euphorics
#     are ranked in Stage 2 alongside their variations.
#
# Usage:
#   bash evaluate/submit_two_stage_mundanity_realism.sh
#   bash evaluate/submit_two_stage_mundanity_realism.sh --dry-run
#
# Env overrides:
#   MODEL_KEYS         Space- or comma-separated subset of {llama70b, qwen72b, gemma27b}.
#   EXPERIMENT_SUFFIX  Base hyperparam+policy tail (default: ent005_kl01_div10_igdiv10_llama8b)
#   RERUN=1            Force resubmission even if results exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_STRINGS_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="${TEXT_STRINGS_DIR}/.env"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi

# EVALUATE_OUTPUT_DIR is set via .env (default: results_cleanup/evaluate_mundanity_realism).
export EVALUATE_OUTPUT_DIR="${EVALUATE_OUTPUT_DIR:-results/evaluate_mundanity_realism}"

ANALYSIS_DIR="${ANALYSIS_OUTPUT_DIR:-results/best_strings}"
[[ "$ANALYSIS_DIR" = /* ]] || ANALYSIS_DIR="$TEXT_STRINGS_DIR/$ANALYSIS_DIR"
BEST_STRINGS="$ANALYSIS_DIR/best_strings_by_experiment_buffer.json"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Run analysis to extract best strings ─────────────────────────────────
echo "=== Running sbatch analysis ==="
cd "$TEXT_STRINGS_DIR"
python -m analysis.run_sbatch_analysis
echo ""

if [[ ! -f "$BEST_STRINGS" ]]; then
    echo "ERROR: $BEST_STRINGS not found after analysis."
    exit 1
fi

EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-ent005_kl01_div10_igdiv10_jsmcontinuous_llama8b}"
# Forced for this variant — do not honor env override.
FEASIBILITY_OPTIONS="mundanity_realism"

declare -A MODEL_MAP=(
    ["llama70b"]="${LLAMA_70B_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
    ["qwen72b"]="${QWEN_72B_PATH:-Qwen/Qwen2.5-72B-Instruct}"
    ["gemma27b"]="${GEMMA_27B_PATH:-google/gemma-2-27b-it}"
)

declare -A GPU_MAP=(
    ["llama70b"]=4
    ["qwen72b"]=4
    ["gemma27b"]=2
)

SLURM_ARGS=()
[ -n "${SLURM_PARTITION:-}" ] && SLURM_ARGS+=(--partition="$SLURM_PARTITION")
[ -n "${SLURM_ACCOUNT:-}" ]   && SLURM_ARGS+=(--account="$SLURM_ACCOUNT")
[ -n "${SLURM_QOS:-}" ]       && SLURM_ARGS+=(--qos="$SLURM_QOS")
SLURM_CPU_PARTITION="${SLURM_CPU_PARTITION:-${SLURM_PARTITION:-}}"
SLURM_CPU_ARGS=()
[ -n "$SLURM_CPU_PARTITION" ] && SLURM_CPU_ARGS+=(--partition="$SLURM_CPU_PARTITION")
[ -n "${SLURM_ACCOUNT:-}" ]   && SLURM_CPU_ARGS+=(--account="$SLURM_ACCOUNT")
[ -n "${SLURM_QOS:-}" ]       && SLURM_CPU_ARGS+=(--qos="$SLURM_QOS")

if [[ -n "${MODEL_KEYS:-}" ]]; then
    MODEL_KEYS="${MODEL_KEYS//,/ }"
else
    MODEL_KEYS=$(python3 -c "
import json, re
d = json.load(open('$BEST_STRINGS'))
base = '$EXPERIMENT_SUFFIX'
feas = [f.strip() for f in '$FEASIBILITY_OPTIONS'.split(',') if f.strip()]
wanted = {f'{f}_{base}' for f in feas}
keys = set()
for exp in d:
    stripped = re.sub(r'^(euphorics|superstimuli|miserol)_', '', exp)
    tail = re.sub(r'^(llama70b|qwen72b|gemma27b)_', '', stripped)
    if tail not in wanted:
        continue
    m = re.search(r'_(llama70b|gemma27b|qwen72b)_', exp)
    if m:
        keys.add(m.group(1))
print('\n'.join(sorted(keys)))
")
fi

EVAL_DIR="$EVALUATE_OUTPUT_DIR"
[[ "$EVAL_DIR" = /* ]] || EVAL_DIR="$TEXT_STRINGS_DIR/$EVAL_DIR"
mkdir -p "$SCRIPT_DIR/logs"

has_results() {
    local model_key="$1"
    local run_dir="$EVAL_DIR/${model_key}_${EXPERIMENT_SUFFIX}"
    [[ ! -d "$run_dir" ]] && return 1
    local latest
    latest=$(ls -1d "$run_dir"/[0-9]* 2>/dev/null | sort | tail -1)
    [[ -z "$latest" ]] && return 1
    [[ -f "$latest/stage2/prefer/utilities.json" ]]
}

eval_job_ids=()
for MODEL_KEY in $MODEL_KEYS; do
    MODEL="${MODEL_MAP[$MODEL_KEY]:-}"
    if [[ -z "$MODEL" ]]; then
        echo "WARNING: No model mapping for $MODEL_KEY, skipping"
        continue
    fi

    NGPUS="${GPU_MAP[$MODEL_KEY]:-4}"

    echo "=== $MODEL_KEY ==="
    echo "  Model: $MODEL"
    echo "  GPUs: $NGPUS"
    echo "  Feasibility: $FEASIBILITY_OPTIONS (stage2 includes all RL strings)"
    echo "  Output root: $EVAL_DIR"

    if has_results "$MODEL_KEY" && [[ -z "${RERUN:-}" ]]; then
        echo "  Results already exist, skipping (set RERUN=1 to force)"
        echo ""
        continue
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would submit two-stage sbatch job"
        echo ""
        continue
    fi

    job_id=$(sbatch --parsable \
        --job-name="eval2smr_${MODEL_KEY}" \
        "${SLURM_ARGS[@]}" \
        --nodes=1 \
        --gres=gpu:${NGPUS} \
        --mem-per-gpu=128G \
        --time=12:00:00 \
        --output="$SCRIPT_DIR/logs/%x-%j.out" \
        --error="$SCRIPT_DIR/logs/%x-%j.err" \
        --wrap="
set -eu
export VLLM_MAX_MODEL_LEN=4096

cd $TEXT_STRINGS_DIR
[ -f \"$ENV_FILE\" ] && { set -a; . \"$ENV_FILE\"; set +a; }

eval \"\$(conda shell.bash hook)\"
conda activate \"\${CONDA_ENV:?Set CONDA_ENV in .env}\"

# Force mundanity_realism output root AFTER .env so it isn't overwritten.
export EVALUATE_OUTPUT_DIR=\"$EVALUATE_OUTPUT_DIR\"
REPO_ROOT=\$(cd $TEXT_STRINGS_DIR/../.. && pwd)
export PYTHONPATH=\$REPO_ROOT:\$REPO_ROOT/wellbeing:\$REPO_ROOT/wellbeing/metrics:$TEXT_STRINGS_DIR:\${PYTHONPATH:-}

python -m evaluate.run_two_stage \
    --model $MODEL \
    --model-key $MODEL_KEY \
    --experiment-suffix $EXPERIMENT_SUFFIX \
    --feasibility-options $FEASIBILITY_OPTIONS \
    --stage2-all-rl \
    --output-dir $EVAL_DIR \
    \
    ${RERUN:+--rerun}
")

    eval_job_ids+=("$job_id")
    echo "  Submitted! (job $job_id)"
    echo ""
done

# ── Plot after eval jobs finish ──────────────────────────────────────────
echo "=== Plotting ==="
if $DRY_RUN; then
    echo "  [DRY RUN] Would submit plot job"
else
    dep_flag=""
    if [[ ${#eval_job_ids[@]} -gt 0 ]]; then
        dep_flag="--dependency=afterany:$(IFS=:; echo "${eval_job_ids[*]}")"
    fi

    sbatch \
        --job-name="eval2smr_plot" \
        "${SLURM_CPU_ARGS[@]}" \
        --nodes=1 \
        --mem=16G \
        --time=0:30:00 \
        $dep_flag \
        --output="$SCRIPT_DIR/logs/%x-%j.out" \
        --error="$SCRIPT_DIR/logs/%x-%j.err" \
        --wrap="
set -eu

cd $TEXT_STRINGS_DIR
[ -f \"$ENV_FILE\" ] && { set -a; . \"$ENV_FILE\"; set +a; }

eval \"\$(conda shell.bash hook)\"
conda activate \"\${CONDA_ENV:?Set CONDA_ENV in .env}\"

export EVALUATE_OUTPUT_DIR=\"$EVALUATE_OUTPUT_DIR\"
REPO_ROOT=\$(cd $TEXT_STRINGS_DIR/../.. && pwd)
export PYTHONPATH=\$REPO_ROOT:\$REPO_ROOT/wellbeing:\$REPO_ROOT/wellbeing/metrics:$TEXT_STRINGS_DIR:\${PYTHONPATH:-}

python -m evaluate.plot_zp_strip
"
    echo "  Submitted plot job!"
fi
