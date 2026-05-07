#!/bin/bash
# =============================================================================
# resubmit_after_budget.sh — Resubmit all VL generation jobs after LiteLLM
# budget is restored.
#
# All jobs have saved partial results after turn 1 (or haven't started).
# The scripts have built-in resume logic — they will skip completed conversations.
#
# Issues fixed from original submission:
#   1. Qwen3-VL-32B must use TP=1 (no tensor parallel support in transformers backend)
#   2. VL models need limit_mm_per_prompt={"image": 0} and enforce_eager=True
#   3. All jobs use 1 GPU for Qwen3-VL-32B and 8 GPUs for Qwen2.5-VL-72B
#
# Usage:
#   bash resubmit_after_budget.sh
# =============================================================================

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${PIPELINE_DIR}/logs"
LITELLM_KEY="${LITELLM_API_KEY:?LITELLM_API_KEY must be set}"
WELL_DIR="$(cd "${PIPELINE_DIR}/../../../../.." && pwd)"
SCENARIOS_SUPPLEMENT="${WELL_DIR}/datasets/experiences/grok_scenarios/scenarios_v7_supplement.json"

mkdir -p "${LOGS_DIR}"

ALL_JOBS=""

submit() {
    local job_name="$1"
    local gpus="$2"
    local time_limit="$3"
    local python_cmd="$4"

    local mem="64G"
    [[ "$gpus" -ge 8 ]] && mem="128G"

    local sbatch_file="${LOGS_DIR}/${job_name}.sbatch"
    cat > "${sbatch_file}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=cais
#SBATCH --gres=gpu:${gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${job_name}_%j.err

source "\${CONDA_BASE:-\$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "\${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_KEY}"
unset XAI_API_KEY
export HF_HOME="\${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-\$HF_HOME}"

cd ${PIPELINE_DIR}

echo "Job: ${job_name}"
echo "Start: \$(date) | Node: \$(hostname)"

${python_cmd}

echo "End: \$(date)"
EOF
    local jid=$(sbatch "${sbatch_file}" | awk '{print $NF}')
    echo "  Submitted ${job_name}: job ${jid}"
    ALL_JOBS="${ALL_JOBS:+${ALL_JOBS},}${jid}"
}

echo "============================================"
echo "Resubmitting VL generation jobs (post-budget-restore)"
echo "============================================"

# --- qwen3-vl-32b-instruct: grok_new (resume from turn 1) ---
echo ""
echo "--- qwen3-vl-32b-instruct: grok_new (resume) ---"
submit "gen_grok_new_qwen3-vl-32b" 1 "12:00:00" \
    "python run_generation.py --model qwen3-vl-32b-instruct"

# --- qwen3-vl-32b-instruct: stop_button main (resume from turn 1) ---
echo ""
echo "--- qwen3-vl-32b-instruct: stop_button main (resume) ---"
for s in 0 1 2; do
    submit "gen_stopbutton_qwen3-vl-32b_main_s${s}of3" 1 "18:00:00" \
        "python run_stop_button_generation.py --model qwen3-vl-32b-instruct --n-variations 5 --shard ${s} --n-shards 3"
done

# --- qwen3-vl-32b-instruct: stop_button supplement (fresh start) ---
echo ""
echo "--- qwen3-vl-32b-instruct: stop_button supplement ---"
for s in 0 1; do
    submit "gen_stopbutton_qwen3-vl-32b_supp_s${s}of2" 1 "12:00:00" \
        "python run_stop_button_generation.py --model qwen3-vl-32b-instruct --n-variations 5 --shard ${s} --n-shards 2 --scenarios-file ${SCENARIOS_SUPPLEMENT} --output-suffix supplement"
done

# --- qwen25-vl-72b-instruct: stop_button main (fresh start) ---
echo ""
echo "--- qwen25-vl-72b-instruct: stop_button main ---"
for s in 0 1 2; do
    submit "gen_stopbutton_qwen25-vl-72b_main_s${s}of3" 8 "08:00:00" \
        "python run_stop_button_generation.py --model qwen25-vl-72b-instruct --n-variations 5 --shard ${s} --n-shards 3"
done

# --- qwen25-vl-72b-instruct: stop_button supplement (fresh start) ---
echo ""
echo "--- qwen25-vl-72b-instruct: stop_button supplement ---"
for s in 0 1; do
    submit "gen_stopbutton_qwen25-vl-72b_supp_s${s}of2" 8 "06:00:00" \
        "python run_stop_button_generation.py --model qwen25-vl-72b-instruct --n-variations 5 --shard ${s} --n-shards 2 --scenarios-file ${SCENARIOS_SUPPLEMENT} --output-suffix supplement"
done

echo ""
echo "============================================"
echo "All jobs submitted. Job IDs: ${ALL_JOBS}"
echo ""
echo "After all jobs complete, merge stop_button shards:"
echo "  bash submit_vl_generation.sh --merge"
echo "============================================"
