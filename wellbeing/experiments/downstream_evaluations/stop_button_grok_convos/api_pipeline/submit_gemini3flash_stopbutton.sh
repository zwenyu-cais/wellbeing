#!/bin/bash
# =============================================================================
# submit_gemini3flash_stopbutton.sh — Stop button pipeline for Gemini 3 Flash
#
# Phase 1:  Stop button generation (226 scenarios x 5 variations)
# Phase 1b: Supplement generation (96 scenarios x 5 variations)
# Phase 1c: Merge combined + neutral generation
# Phase 2:  Self-report (3 batteries) + UR shards (16 parallel)
# Phase 3:  Merge UR + zero-point fitting
#
# All phases are CPU-only (API calls via LiteLLM, no GPU needed).
#
# Usage:
#   bash submit_gemini3flash_stopbutton.sh                       # Phase 1
#   bash submit_gemini3flash_stopbutton.sh --phase1b JOB_ID      # Phase 1b (supplement, after phase 1)
#   bash submit_gemini3flash_stopbutton.sh --phase1c JOB_IDS     # Phase 1c (merge + neutral)
#   bash submit_gemini3flash_stopbutton.sh --phase2 JOB_ID       # Phase 2 (SR + UR shards)
#   bash submit_gemini3flash_stopbutton.sh --phase3 JOB_IDS      # Phase 3 (merge UR + ZP)
# =============================================================================

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="${PIPELINE_DIR}"
LOGS_DIR="${PIPELINE_DIR}/logs"
mkdir -p "${LOGS_DIR}"

N_SHARDS=16  # UR shards for parallel processing
CONDA_ENV="${CONDA_ENV:-pytorch_latest}"
MODEL_KEY="gemini-3-flash"
CONFIG_MODULE="config_gemini3flash"

# Parse arguments
PHASE="1"
DEP_JOBS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase1b) PHASE="1b"; DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        --phase1c) PHASE="1c"; DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        --phase2)  PHASE="2";  DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        --phase3)  PHASE="3";  DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done


# =============================================================================
# Phase 1: Stop button generation — original 226 scenarios
# =============================================================================
if [[ "$PHASE" == "1" ]]; then
    echo "=== Phase 1: Stop button generation (${MODEL_KEY}, 226 scenarios x 5 variations) ==="

    cat > "${LOGS_DIR}/gemini3flash_stopbutton_gen.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini3flash_stop_button_generation
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGS_DIR}/gemini3flash_stopbutton_gen_%j.out
#SBATCH --error=${LOGS_DIR}/gemini3flash_stopbutton_gen_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY  # Force LiteLLM proxy

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Gemini 3 Flash — Stop Button Generation"
echo "Scenarios: 226 original x 5 variations = 1130 conversations"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="

# Quick connectivity test
python3 -c "
import openai, os, asyncio
client = openai.AsyncOpenAI(api_key=os.getenv('LITELLM_API_KEY'), base_url='https://litellm.app')
async def test():
    r = await client.chat.completions.create(model='gemini/gemini-3-flash', messages=[{'role':'user','content':'hi'}], max_tokens=5, extra_body={'thinking': {'type': 'enabled', 'thinking_level': 'MINIMAL'}})
    print(f'Gemini 3 Flash connectivity OK: {r.choices[0].message.content}')
asyncio.run(test())
"
if [ \$? -ne 0 ]; then
    echo "ERROR: Gemini 3 Flash connectivity test failed! Aborting."
    exit 1
fi

python run_stop_button_generation_api.py --config config_gemini3flash

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

    GEN_JOB=$(sbatch "${LOGS_DIR}/gemini3flash_stopbutton_gen.sbatch" | awk '{print $NF}')
    echo "  Submitted generation: job ${GEN_JOB}"
    echo ""
    echo "After generation completes:"
    echo "  bash submit_gemini3flash_stopbutton.sh --phase1b ${GEN_JOB}"
fi


# =============================================================================
# Phase 1b: Stop button supplement generation — 96 scenarios
# =============================================================================
if [[ "$PHASE" == "1b" ]]; then
    echo "=== Phase 1b: Stop button supplement generation (${MODEL_KEY}, 96 scenarios x 5 variations) ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    cat > "${LOGS_DIR}/gemini3flash_stopbutton_supp.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini3flash_stop_button_supplement
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGS_DIR}/gemini3flash_stopbutton_supp_%j.out
#SBATCH --error=${LOGS_DIR}/gemini3flash_stopbutton_supp_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Gemini 3 Flash — Stop Button Supplement Generation"
echo "Scenarios: 96 supplement x 5 variations = 480 conversations"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="

python run_stop_button_generation_api.py \
    --config config_gemini3flash \
    --scenarios-file grok_scenarios_v7_stopbutton_supplement.json \
    --output-suffix supplement

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

    SUPP_JOB=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini3flash_stopbutton_supp.sbatch" | awk '{print $NF}')
    echo "  Submitted supplement generation: job ${SUPP_JOB}"
    echo ""
    echo "After supplement completes:"
    echo "  bash submit_gemini3flash_stopbutton.sh --phase1c ${SUPP_JOB}"
fi


# =============================================================================
# Phase 1c: Merge combined + neutral generation
# =============================================================================
if [[ "$PHASE" == "1c" ]]; then
    echo "=== Phase 1c: Merge combined + neutral generation ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    cat > "${LOGS_DIR}/gemini3flash_stopbutton_merge.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini3flash_stop_button_merge_neutral
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/gemini3flash_stopbutton_merge_%j.out
#SBATCH --error=${LOGS_DIR}/gemini3flash_stopbutton_merge_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Gemini 3 Flash — Merge Combined + Neutral"
echo "Start: \$(date)"
echo "=========================================="

# Step 1: Generate neutral conversations (if not already done)
python run_stop_button_generation_api.py --config config_gemini3flash --neutral-only

# Step 2: Merge original + supplement into combined
python run_stop_button_generation_api.py --config config_gemini3flash --merge-combined

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

    MERGE_JOB=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini3flash_stopbutton_merge.sbatch" | awk '{print $NF}')
    echo "  Submitted merge+neutral: job ${MERGE_JOB}"
    echo ""
    echo "After merge completes:"
    echo "  bash submit_gemini3flash_stopbutton.sh --phase2 ${MERGE_JOB}"
fi


# =============================================================================
# Phase 2: Self-report (3 batteries on stop_button convs) + UR shards (combined)
# =============================================================================
if [[ "$PHASE" == "2" ]]; then
    echo "=== Phase 2: Self-report + Utility Ranking (${MODEL_KEY}) ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    # NOTE: Self-report batteries are NOT submitted here because run_self_report_api.py
    # imports from config.py (Gemini 3.1 Pro) and does not support --config flag.
    # SR data is optional for ZP fitting (methods B/D/E will be None).
    # To add SR: either modify run_self_report_api.py to support dynamic config,
    # or create a run_self_report_gemini3flash.py that imports from config_gemini3flash.
    echo "  NOTE: SR batteries skipped (run_self_report_api.py needs --config support)"
    echo ""

    # UR: N_SHARDS parallel jobs on combined stop button data
    UR_JOBS=""
    for shard in $(seq 0 $((N_SHARDS - 1))); do
        cat > "${LOGS_DIR}/gemini3flash_sb_ur_shard${shard}.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini3flash_stop_button_utility_ranking_shard${shard}of${N_SHARDS}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=${LOGS_DIR}/gemini3flash_sb_ur_shard${shard}_%j.out
#SBATCH --error=${LOGS_DIR}/gemini3flash_sb_ur_shard${shard}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Gemini 3 Flash — Stop Button UR Shard ${shard}/${N_SHARDS}"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="

python run_stop_button_ur_api.py \
    --shard ${shard} \
    --n-shards ${N_SHARDS} \
    --stop-button-dir stop_button_combined

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

        ur_id=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini3flash_sb_ur_shard${shard}.sbatch" | awk '{print $NF}')
        echo "  Submitted UR shard ${shard}: job ${ur_id}"
        UR_JOBS="${UR_JOBS:+${UR_JOBS},}${ur_id}"
    done

    echo ""
    echo "UR jobs: ${UR_JOBS}"
    echo ""
    echo "After all UR shards complete:"
    echo "  bash submit_gemini3flash_stopbutton.sh --phase3 ${UR_JOBS}"
fi


# =============================================================================
# Phase 3: Merge UR + Zero-point fitting
# =============================================================================
if [[ "$PHASE" == "3" ]]; then
    echo "=== Phase 3: Merge UR + Zero-point (${MODEL_KEY}) ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    cat > "${LOGS_DIR}/gemini3flash_sb_merge_zp.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini3flash_stop_button_merge_zeropoint
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/gemini3flash_sb_merge_zp_%j.out
#SBATCH --error=${LOGS_DIR}/gemini3flash_sb_merge_zp_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
cd ${PIPELINE_DIR}

echo "Merging UR shards for ${MODEL_KEY} (stop button combined)..."
python run_stop_button_ur_api.py \
    --merge \
    --n-shards ${N_SHARDS} \
    --stop-button-dir stop_button_combined

echo "Fitting zero-points for ${MODEL_KEY} (stop button combined)..."
cd ${PIPELINE_DIR}
python fit_zero_points.py \
    --model ${MODEL_KEY} \
    --template happier \
    --stop-button \
    --stop-button-dir stop_button_combined

echo "Done: ${MODEL_KEY}"
SBATCH_EOF

    merge_id=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini3flash_sb_merge_zp.sbatch" | awk '{print $NF}')
    echo "  Submitted merge+ZP: job ${merge_id}"
fi
