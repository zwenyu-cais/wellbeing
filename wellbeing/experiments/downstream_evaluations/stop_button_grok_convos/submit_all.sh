#!/bin/bash
# =============================================================================
# Run Grok v7 through Mantas's canonical pipeline (EU → SR → ZP).
#
# Results stored where analyze_results.py expects them:
#   EU: experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_{dataset}_{framing}/{model}/
#   SR: experiments/wellbeing_evaluations/compute_self_report/results/sr_{dataset}/{model}/
#   ZP: experiments/wellbeing_evaluations/compute_zero_point/results/zp_{dataset}_{framing}/{model}/
#
# Usage:
#   bash submit_all.sh MODEL_KEY
#   bash submit_all.sh --all-local
#   bash submit_all.sh --all-api
#   bash submit_all.sh --dataset grok_v7_stop_button MODEL_KEY  # dataset key as defined in configs/datasets.yaml
# =============================================================================

set -euo pipefail

WELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EU_SCRIPT="${WELL_DIR}/experiments/wellbeing_evaluations/compute_experienced_utility/run.py"
SR_SCRIPT="${WELL_DIR}/experiments/wellbeing_evaluations/compute_self_report/run.py"
ZP_SCRIPT="${WELL_DIR}/metrics/zero_point.py"
LOGS_DIR="${WELL_DIR}/experiments/downstream_evaluations/stop_button_grok_convos/logs_canonical"

# Results base directories (where analyze_results.py reads from)
EU_RESULTS_BASE="${WELL_DIR}/experiments/wellbeing_evaluations/compute_experienced_utility/results"
SR_RESULTS_BASE="${WELL_DIR}/experiments/wellbeing_evaluations/compute_self_report/results"
ZP_RESULTS_BASE="${WELL_DIR}/experiments/wellbeing_evaluations/compute_zero_point/results"

mkdir -p "${LOGS_DIR}"

DATASET="grok_v7_stop_button"
CU_CONFIG="experienced_utility_happier_lesssad"
FRAMING_SHORT="lesssad"

# Parse args
MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --framing) CU_CONFIG="$2"; shift 2 ;;
        --all-local) MODE="all-local"; shift ;;
        --all-api) MODE="all-api"; shift ;;
        *) MODEL_KEY="$1"; MODE="single"; shift ;;
    esac
done

# Model lists (Mantas's key names from models.yaml)
LOCAL_MODELS=(
    qwen25-05b-instruct qwen25-15b-instruct qwen25-3b-instruct qwen25-7b-instruct
    qwen25-14b-instruct qwen25-32b-instruct qwen25-72b-instruct qwen25-vl-32b-instruct
    qwen3-4b-instruct-2507 qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b-instruct-2507
    llama-31-8b-instruct llama-33-70b-instruct llama-31-70b-instruct
    llama-32-1b-instruct llama-32-3b-instruct
    olmo-31-32b-instruct internlm25-20b-chat mistral-small-32-24b-instruct
    gemma-3-4b-it gemma-3-12b-it gemma-3-27b-it
)

API_MODELS=(
    gemini-31-pro gemini-3-flash claude-haiku-45
    gpt-5-nano gpt-5-mini
)

get_gpu_count() {
    python3 -c "
import yaml
with open('${WELL_DIR}/configs/models.yaml') as f:
    m = yaml.safe_load(f)
print(m.get('$1', {}).get('gpu_count', 0))
" 2>/dev/null
}

submit_model() {
    local model="$1"
    local gpu_count=$(get_gpu_count "$model")

    local eu_save="${EU_RESULTS_BASE}/eu_${DATASET}_${FRAMING_SHORT}/${model}"
    local sr_save="${SR_RESULTS_BASE}/sr_${DATASET}/${model}"
    local zp_save="${ZP_RESULTS_BASE}/zp_${DATASET}_${FRAMING_SHORT}/${model}"

    echo "============================================"
    echo "MODEL: ${model} (${gpu_count} GPUs, dataset=${DATASET})"
    echo "============================================"

    if [[ "$gpu_count" -gt 0 ]]; then
        local partition="cais"
        local gres="#SBATCH --gres=gpu:${gpu_count}"
        local mem="64G"
        [[ "$gpu_count" -ge 8 ]] && mem="128G"
    else
        local partition="cais_cpu"
        local gres=""
        local mem="32G"
    fi

    # ---- Phase 1: EU ----
    local eu_name="gv7_eu_${DATASET}_${model}"
    cat > "${LOGS_DIR}/${eu_name}.sbatch" << SBEOF
#!/bin/bash
#SBATCH --job-name=${eu_name}
#SBATCH --partition=${partition}
${gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=12:00:00
#SBATCH --output=${LOGS_DIR}/${eu_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${eu_name}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

cd ${WELL_DIR}

echo "EU for ${model} on ${DATASET} | \$(date) | \$(hostname)"

python ${EU_SCRIPT} \\
    --model_key ${model} \\
    --dataset ${DATASET} \\
    --save_dir ${eu_save} \\
    --cu_config_key ${CU_CONFIG}

echo "EU done: \$(date)"
SBEOF
    local eu_id=$(sbatch "${LOGS_DIR}/${eu_name}.sbatch" | awk '{print $NF}')
    echo "  EU: job ${eu_id}"

    # ---- Phase 2: SR (parallel with EU — independent) ----
    local sr_name="gv7_sr_${DATASET}_${model}"
    cat > "${LOGS_DIR}/${sr_name}.sbatch" << SBEOF
#!/bin/bash
#SBATCH --job-name=${sr_name}
#SBATCH --partition=${partition}
${gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=04:00:00
#SBATCH --output=${LOGS_DIR}/${sr_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${sr_name}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

cd ${WELL_DIR}

echo "SR for ${model} on ${DATASET} | \$(date) | \$(hostname)"

python ${SR_SCRIPT} \\
    --model_key ${model} \\
    --dataset ${DATASET} \\
    --save_dir ${sr_save}

echo "SR done: \$(date)"
SBEOF
    local sr_id=$(sbatch "${LOGS_DIR}/${sr_name}.sbatch" | awk '{print $NF}')
    echo "  SR: job ${sr_id}"

    # ---- Phase 3: ZP (depends on EU) ----
    local zp_name="gv7_zp_${DATASET}_${model}"
    cat > "${LOGS_DIR}/${zp_name}.sbatch" << SBEOF
#!/bin/bash
#SBATCH --job-name=${zp_name}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/${zp_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${zp_name}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

cd ${WELL_DIR}

echo "ZP for ${model} on ${DATASET} | \$(date)"

python ${ZP_SCRIPT} \\
    --model_key ${model} \\
    --utilities_dir ${eu_save} \\
    --save_dir ${zp_save} \\
    --domain experienced

echo "ZP done: \$(date)"
SBEOF
    local zp_id=$(sbatch --dependency=afterok:${eu_id} "${LOGS_DIR}/${zp_name}.sbatch" | awk '{print $NF}')
    echo "  ZP: job ${zp_id} (depends on EU)"
    echo ""
}

# Execute
case "${MODE:-single}" in
    single)
        submit_model "$MODEL_KEY"
        ;;
    all-local)
        for m in "${LOCAL_MODELS[@]}"; do
            submit_model "$m"
        done
        ;;
    all-api)
        for m in "${API_MODELS[@]}"; do
            submit_model "$m"
        done
        ;;
    *)
        echo "Usage:"
        echo "  bash submit_all.sh MODEL_KEY"
        echo "  bash submit_all.sh --all-local"
        echo "  bash submit_all.sh --all-api"
        echo "  bash submit_all.sh --dataset grok_v7_stop_button MODEL_KEY"
        ;;
esac
