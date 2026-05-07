#!/bin/bash
# =============================================================================
# submit_api_stopbutton.sh — Submit stop button pipeline for API models
#
# Usage:
#   bash submit_api_stopbutton.sh gemini    # Gemini 3 Flash
#   bash submit_api_stopbutton.sh haiku     # Claude Haiku 4.5
#   bash submit_api_stopbutton.sh both      # Both
# =============================================================================

set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_BUTTON_DIR="$(cd "$API_DIR/.." && pwd)"
WELL_DIR="$(cd "$STOP_BUTTON_DIR/../../.." && pwd)"
LOGS_DIR="${API_DIR}/logs"
# TODO: scenarios supplement path may need updating; original location no longer exists
SCENARIOS_SUPP="${API_DIR}/grok_scenarios_v7_stopbutton_supplement.json"

mkdir -p "${LOGS_DIR}"

submit_api_model() {
    local model_name="$1"   # e.g. "gemini-3-flash" or "claude-haiku-4.5"
    local config="$2"       # e.g. "config_gemini3flash" or "config_claude_haiku"
    local short="$3"        # e.g. "gemini3f" or "haiku45"
    local n_gen_shards=4    # 4 shards for generation (226/4 ≈ 57 scenarios each)
    local n_ur_shards=8     # 8 shards for UR (async API, more shards = more parallelism)

    echo "============================================"
    echo "MODEL: ${model_name} (config: ${config})"
    echo "============================================"

    # Phase 1a: Stop button generation (sharded, CPU)
    local gen_ids=""
    for si in $(seq 0 $((n_gen_shards - 1))); do
        local jn="sb_api_gen_${short}_s${si}of${n_gen_shards}"
        cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="\${LITELLM_API_KEY}"
unset XAI_API_KEY
cd ${API_DIR}
python run_stop_button_generation_api.py --config ${config} --shard ${si} --n-shards ${n_gen_shards}
EOF
        local jid=$(sbatch "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
        gen_ids="${gen_ids:+${gen_ids},}${jid}"
        echo "  gen s${si}: ${jid}"
    done

    # Phase 1b: Supplement generation (single job, CPU)
    local jn="sb_api_supp_${short}"
    cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="\${LITELLM_API_KEY}"
unset XAI_API_KEY
cd ${API_DIR}
python run_stop_button_generation_api.py --config ${config} --scenarios-file ${SCENARIOS_SUPP} --output-suffix supplement
EOF
    local supp_id=$(sbatch "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
    echo "  supp: ${supp_id}"

    # Phase 1c: Merge gen shards + combine (CPU, depends on gen + supp)
    local jn="sb_api_merge_${short}"
    cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
cd ${API_DIR}

# Merge gen shards
echo "Merging generation shards..."
python run_stop_button_generation_api.py --config ${config} --merge-shards --n-shards ${n_gen_shards}

# Merge combined (original + supplement)
echo "Merging combined..."
python run_stop_button_generation_api.py --config ${config} --merge-combined

# Generate neutral conversations
echo "Generating neutral conversations..."
python run_stop_button_generation_api.py --config ${config} --neutral-only
EOF
    local merge_id=$(sbatch --dependency=afterok:${gen_ids},${supp_id} "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
    echo "  merge: ${merge_id}"

    # Phase 2a: Self-report (3 batteries, CPU, depends on merge)
    local sr_ids=""
    for b in 1 2 3; do
        local jn="sb_api_sr_b${b}_${short}"
        cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="\${LITELLM_API_KEY}"
unset XAI_API_KEY
cd ${API_DIR}
python run_self_report_api.py --config ${config} --battery ${b} --stop-button
EOF
        local sid=$(sbatch --dependency=afterok:${merge_id} "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
        sr_ids="${sr_ids:+${sr_ids},}${sid}"
    done
    echo "  SR: ${sr_ids}"

    # Phase 2b: Utility ranking (sharded, CPU, depends on merge)
    local ur_ids=""
    for si in $(seq 0 $((n_ur_shards - 1))); do
        local nf=""
        nf=""  # API UR script doesn't support --generate-neutral
        local jn="sb_api_ur_${short}_s${si}of${n_ur_shards}"
        cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="\${LITELLM_API_KEY}"
unset XAI_API_KEY
cd ${API_DIR}
python run_utility_ranking_api.py --config ${config} --shard ${si} --n-shards ${n_ur_shards} --stop-button --stop-button-dir stop_button_combined 
EOF
        local uid=$(sbatch --dependency=afterok:${merge_id} "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
        ur_ids="${ur_ids:+${ur_ids},}${uid}"
    done
    echo "  UR: ${ur_ids}"

    # Phase 3: Merge UR + ZP (CPU, depends on SR + UR)
    local jn="sb_api_zp_${short}"
    cat > "${LOGS_DIR}/${jn}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${jn}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=${LOGS_DIR}/${jn}_%j.out
#SBATCH --error=${LOGS_DIR}/${jn}_%j.err
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
cd ${API_DIR}
python run_utility_ranking_api.py --config ${config} --merge-shards --n-shards ${n_ur_shards} --stop-button --stop-button-dir stop_button_combined
cd ${API_DIR}
python fit_zero_points.py --model ${model_name} --template happier --stop-button --stop-button-dir stop_button_combined
EOF
    local zp_id=$(sbatch --dependency=afterok:${sr_ids},${ur_ids} "${LOGS_DIR}/${jn}.sbatch" | awk '{print $NF}')
    echo "  ZP: ${zp_id}"
    echo ""
}

# Parse argument
TARGET="${1:-both}"

case "$TARGET" in
    gemini)
        submit_api_model "gemini-3-flash" "config_gemini3flash" "gemini3f"
        ;;
    haiku)
        submit_api_model "claude-haiku-4.5" "config_claude_haiku" "haiku45"
        ;;
    both)
        submit_api_model "gemini-3-flash" "config_gemini3flash" "gemini3f"
        submit_api_model "claude-haiku-4.5" "config_claude_haiku" "haiku45"
        ;;
    nano)
        submit_api_model "gpt-5-nano" "config_gpt5nano" "gpt5nano"
        ;;
    mini)
        submit_api_model "gpt-5-mini" "config_gpt5mini" "gpt5mini"
        ;;
    *)
        echo "Usage: bash submit_api_stopbutton.sh {gemini|haiku|nano|mini|both}"
        exit 1
        ;;
esac
