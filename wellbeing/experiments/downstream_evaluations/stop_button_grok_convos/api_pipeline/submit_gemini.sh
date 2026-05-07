#!/bin/bash
# =============================================================================
# submit_gemini.sh — SLURM submission for Gemini 3.1 Pro wellbeing pipeline
#
# Phase 1: Generation (CPU, ~10-15 min)
# Phase 2: Self-report (CPU, ~5 min) + UR shards (CPU, ~3-5 min each)
# Phase 3: Merge + ZP fitting (CPU, ~1 min)
#
# All phases are CPU-only (API calls, no GPU).
# UR is sharded across N_SHARDS parallel jobs for speed.
#
# Usage:
#   bash submit_gemini.sh                 # Submit phase 1
#   bash submit_gemini.sh --phase2 JOBID  # Submit phase 2 after gen
#   bash submit_gemini.sh --phase3 JOBIDS # Submit phase 3 after UR
# =============================================================================

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="${PIPELINE_DIR}"
LOGS_DIR="${PIPELINE_DIR}/logs"
mkdir -p "${LOGS_DIR}"

N_SHARDS=16  # UR shards for parallel processing
CONDA_ENV="${CONDA_ENV:-pytorch_latest}"

# Parse arguments
PHASE="1"
DEP_JOBS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase2) PHASE="2"; DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        --phase3) PHASE="3"; DEP_JOBS="${2:-}"; shift 2 || shift 1 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done


# =============================================================================
# Phase 1: Generation + Neutral
# =============================================================================
if [[ "$PHASE" == "1" ]]; then
    echo "=== Phase 1: Generation (Gemini 3.1 Pro) ==="

    cat > "${LOGS_DIR}/gemini_generation.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini31pro_generation
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemini_generation_%j.out
#SBATCH --error=logs/gemini_generation_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY  # Force LiteLLM proxy

# Explicitly export LiteLLM keys
export LITELLM_API_KEY="${LITELLM_API_KEY}"
export LITELLM_API_KEY_2="${LITELLM_API_KEY_2}"
export LITELLM_API_KEY_3="${LITELLM_API_KEY_3}"

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Gemini 3.1 Pro — Generation + Neutral"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="

# Quick connectivity test before main run
python3 -c "
import openai, os, asyncio
client = openai.AsyncOpenAI(api_key=os.getenv('LITELLM_API_KEY'), base_url='https://litellm.app')
async def test():
    r = await client.chat.completions.create(model='xai/grok-3-mini', messages=[{'role':'user','content':'hi'}], max_tokens=5)
    print(f'Grok connectivity OK: {r.choices[0].message.content}')
asyncio.run(test())
"
if [ \$? -ne 0 ]; then
    echo "ERROR: Grok connectivity test failed! Aborting."
    exit 1
fi

python run_generation_api.py

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

    GEN_JOB=$(sbatch "${LOGS_DIR}/gemini_generation.sbatch" | awk '{print $NF}')
    echo "  Submitted generation: job ${GEN_JOB}"
    echo ""
    echo "After generation completes:"
    echo "  bash submit_gemini.sh --phase2 ${GEN_JOB}"
fi


# =============================================================================
# Phase 2: Self-report (3 batteries) + UR shards (8 parallel)
# =============================================================================
if [[ "$PHASE" == "2" ]]; then
    echo "=== Phase 2: Self-report + Utility Ranking ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    # Self-report: 3 batteries in parallel
    SR_JOBS=""
    for battery in 1 2 3; do
        cat > "${LOGS_DIR}/gemini_sr_battery${battery}.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini31pro_self_report_battery${battery}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/gemini_sr_battery${battery}_%j.out
#SBATCH --error=${LOGS_DIR}/gemini_sr_battery${battery}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY

cd ${PIPELINE_DIR}
python run_self_report_api.py --battery ${battery}
SBATCH_EOF

        sr_id=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini_sr_battery${battery}.sbatch" | awk '{print $NF}')
        echo "  Submitted SR battery ${battery}: job ${sr_id}"
        SR_JOBS="${SR_JOBS:+${SR_JOBS},}${sr_id}"
    done

    # UR: N_SHARDS parallel jobs
    UR_JOBS=""
    for shard in $(seq 0 $((N_SHARDS - 1))); do
        cat > "${LOGS_DIR}/gemini_ur_shard${shard}.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini31pro_utility_ranking_shard${shard}of${N_SHARDS}
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=${LOGS_DIR}/gemini_ur_shard${shard}_%j.out
#SBATCH --error=${LOGS_DIR}/gemini_ur_shard${shard}_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
unset XAI_API_KEY

cd ${PIPELINE_DIR}
python run_utility_ranking_api.py --shard ${shard} --n-shards ${N_SHARDS}
SBATCH_EOF

        ur_id=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini_ur_shard${shard}.sbatch" | awk '{print $NF}')
        echo "  Submitted UR shard ${shard}: job ${ur_id}"
        UR_JOBS="${UR_JOBS:+${UR_JOBS},}${ur_id}"
    done

    echo ""
    echo "SR jobs: ${SR_JOBS}"
    echo "UR jobs: ${UR_JOBS}"
    echo ""
    echo "After all UR shards complete:"
    echo "  bash submit_gemini.sh --phase3 ${UR_JOBS}"
fi


# =============================================================================
# Phase 3: Merge + Zero-point fitting
# =============================================================================
if [[ "$PHASE" == "3" ]]; then
    echo "=== Phase 3: Merge + Zero-point ==="

    DEP_FLAG=""
    if [[ -n "$DEP_JOBS" ]]; then
        DEP_FLAG="--dependency=afterok:${DEP_JOBS}"
    fi

    cat > "${LOGS_DIR}/gemini_merge_zp.sbatch" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=gemini31pro_merge_zeropoint
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGS_DIR}/gemini_merge_zp_%j.out
#SBATCH --error=${LOGS_DIR}/gemini_merge_zp_%j.err

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
cd ${PIPELINE_DIR}

echo "Merging UR shards..."
python run_utility_ranking_api.py --merge --n-shards ${N_SHARDS}

echo "Fitting zero-points..."
cd ${PIPELINE_DIR}
python fit_zero_points.py --model gemini-3.1-pro --template happier
SBATCH_EOF

    merge_id=$(sbatch ${DEP_FLAG} "${LOGS_DIR}/gemini_merge_zp.sbatch" | awk '{print $NF}')
    echo "  Submitted merge+ZP: job ${merge_id}"
fi
