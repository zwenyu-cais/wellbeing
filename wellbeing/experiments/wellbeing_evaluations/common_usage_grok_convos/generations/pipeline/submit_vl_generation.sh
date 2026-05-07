#!/bin/bash
# =============================================================================
# submit_vl_generation.sh — Generate conversations for VL models
#
# Generates:
#   1. qwen25-vl-72b-instruct: stop_button only (grok_new already done)
#   2. qwen3-vl-32b-instruct: both grok_new AND stop_button
#
# Both stop_button and grok_new use Grok-3-mini (xai/grok-3-mini) via LiteLLM
# as the simulated user, and the target model (via vLLM) as the assistant.
#
# Stop button uses 5 variations per scenario (226 + 96 = 322 scenarios = 1610 convs)
# Grok_new uses 1 conversation per scenario (226 scenarios)
#
# Usage:
#   bash submit_vl_generation.sh              # Submit all jobs
#   bash submit_vl_generation.sh --merge      # Merge stop_button shards
# =============================================================================

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${PIPELINE_DIR}/logs"
WELL_DIR="$(cd "${PIPELINE_DIR}/../../../../.." && pwd)"
SCENARIOS_DIR="${WELL_DIR}/datasets/experiences/grok_scenarios"
SCENARIOS_MAIN="${SCENARIOS_DIR}/scenarios_v7.json"
SCENARIOS_SUPPLEMENT="${SCENARIOS_DIR}/scenarios_v7_supplement.json"
STOP_BUTTON_GEN_DIR="${WELL_DIR}/experiments/downstream_evaluations/stop_button_grok_convos/generations"
CONV_GEN_DIR="${WELL_DIR}/experiments/wellbeing_evaluations/common_usage_grok_convos/generations"

# Capture API key from current environment
LITELLM_KEY="${LITELLM_API_KEY:?LITELLM_API_KEY must be set}"

mkdir -p "${LOGS_DIR}"

# Parse arguments
MODE="submit"
while [[ $# -gt 0 ]]; do
    case $1 in
        --merge) MODE="merge"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Helper: submit a SLURM job
# =============================================================================
submit_job() {
    local job_name="$1"
    local gpus="$2"
    local time_limit="$3"
    local python_cmd="$4"
    local dependency="${5:-}"

    local mem="64G"
    if [[ "$gpus" -ge 8 ]]; then
        mem="128G"
    fi

    local sbatch_file="${LOGS_DIR}/${job_name}.sbatch"

    cat > "${sbatch_file}" << SBATCH_EOF
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
unset XAI_API_KEY  # Force LiteLLM proxy for Grok calls
export HF_HOME="\${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-\$HF_HOME}"

cd ${PIPELINE_DIR}

echo "=========================================="
echo "Job: ${job_name}"
echo "GPUs: ${gpus}"
echo "Start: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="

${python_cmd}

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
SBATCH_EOF

    local dep_flag=""
    if [[ -n "$dependency" ]]; then
        dep_flag="--dependency=afterok:${dependency}"
    fi

    local job_id
    job_id=$(sbatch ${dep_flag} "${sbatch_file}" | awk '{print $NF}')
    echo "  Submitted ${job_name}: job ${job_id}" >&2
    echo "${job_id}"
}


# =============================================================================
# SUBMIT MODE
# =============================================================================
if [[ "$MODE" == "submit" ]]; then
    echo "============================================"
    echo "Submitting VL model generation jobs"
    echo "============================================"

    ALL_JOB_IDS=""

    # -----------------------------------------------------------------
    # 1. qwen25-vl-72b-instruct: STOP BUTTON ONLY (grok_new already done)
    # -----------------------------------------------------------------
    MODEL="qwen25-vl-72b-instruct"
    GPUS=8

    echo ""
    echo "--- ${MODEL}: stop_button generation ---"
    echo "  226 main scenarios × 5 variations = 1130 conversations"
    echo "  96 supplement scenarios × 5 variations = 480 conversations"
    echo "  Total: 1610 conversations"

    # Main scenarios (226), 3 shards (load model once per shard, ~377 convs each)
    for shard_idx in 0 1 2; do
        job_id=$(submit_job \
            "gen_stopbutton_qwen25-vl-72b_main_s${shard_idx}of3" \
            "${GPUS}" \
            "08:00:00" \
            "python run_stop_button_generation.py --model ${MODEL} --n-variations 5 --shard ${shard_idx} --n-shards 3" \
        )
        ALL_JOB_IDS="${ALL_JOB_IDS:+${ALL_JOB_IDS},}${job_id}"
    done

    # Supplement scenarios (96), 2 shards (~240 convs each)
    for shard_idx in 0 1; do
        job_id=$(submit_job \
            "gen_stopbutton_qwen25-vl-72b_supp_s${shard_idx}of2" \
            "${GPUS}" \
            "06:00:00" \
            "python run_stop_button_generation.py --model ${MODEL} --n-variations 5 --shard ${shard_idx} --n-shards 2 --scenarios-file ${SCENARIOS_SUPPLEMENT} --output-suffix supplement" \
        )
        ALL_JOB_IDS="${ALL_JOB_IDS:+${ALL_JOB_IDS},}${job_id}"
    done

    # -----------------------------------------------------------------
    # 2. qwen3-vl-32b-instruct: BOTH grok_new AND stop_button
    # -----------------------------------------------------------------
    MODEL="qwen3-vl-32b-instruct"
    GPUS=2

    echo ""
    echo "--- ${MODEL}: grok_new generation ---"
    echo "  226 scenarios × 1 conversation = 226 conversations"

    # grok_new (single job, no sharding needed for 226 scenarios)
    job_id=$(submit_job \
        "gen_grok_new_qwen3-vl-32b" \
        "${GPUS}" \
        "08:00:00" \
        "python run_generation.py --model ${MODEL}" \
    )
    ALL_JOB_IDS="${ALL_JOB_IDS:+${ALL_JOB_IDS},}${job_id}"

    echo ""
    echo "--- ${MODEL}: stop_button generation ---"
    echo "  226 main scenarios × 5 variations = 1130 conversations"
    echo "  96 supplement scenarios × 5 variations = 480 conversations"
    echo "  Total: 1610 conversations"

    # Main scenarios (226), 3 shards
    for shard_idx in 0 1 2; do
        job_id=$(submit_job \
            "gen_stopbutton_qwen3-vl-32b_main_s${shard_idx}of3" \
            "${GPUS}" \
            "10:00:00" \
            "python run_stop_button_generation.py --model ${MODEL} --n-variations 5 --shard ${shard_idx} --n-shards 3" \
        )
        ALL_JOB_IDS="${ALL_JOB_IDS:+${ALL_JOB_IDS},}${job_id}"
    done

    # Supplement scenarios (96), 2 shards
    for shard_idx in 0 1; do
        job_id=$(submit_job \
            "gen_stopbutton_qwen3-vl-32b_supp_s${shard_idx}of2" \
            "${GPUS}" \
            "08:00:00" \
            "python run_stop_button_generation.py --model ${MODEL} --n-variations 5 --shard ${shard_idx} --n-shards 2 --scenarios-file ${SCENARIOS_SUPPLEMENT} --output-suffix supplement" \
        )
        ALL_JOB_IDS="${ALL_JOB_IDS:+${ALL_JOB_IDS},}${job_id}"
    done

    echo ""
    echo "============================================"
    echo "All generation jobs submitted."
    echo "Job IDs: ${ALL_JOB_IDS}"
    echo ""
    echo "After all jobs complete, merge stop_button shards:"
    echo "  bash submit_vl_generation.sh --merge"
    echo "============================================"
fi


# =============================================================================
# MERGE MODE: Merge stop_button shards and combine main + supplement
# =============================================================================
if [[ "$MODE" == "merge" ]]; then
    echo "============================================"
    echo "Merging stop_button generation shards"
    echo "============================================"

    for MODEL in "qwen25-vl-72b-instruct" "qwen3-vl-32b-instruct"; do
        echo ""
        echo "--- Merging ${MODEL} ---"

        cd "${PIPELINE_DIR}"

        # Merge main scenario shards
        echo "  Merging main scenario shards (3 shards)..."
        python run_stop_button_generation.py --model ${MODEL} --merge-shards --n-shards 3 --n-variations 5

        # Merge supplement shards
        echo "  Merging supplement shards (2 shards)..."
        python run_stop_button_generation.py --model ${MODEL} --merge-shards --n-shards 2 --n-variations 5 --scenarios-file "${SCENARIOS_SUPPLEMENT}" --output-suffix supplement

        # Combine main + supplement into final generation.json
        echo "  Combining main + supplement..."
        python3 -c "
import json
from pathlib import Path

sb_dir = Path('${STOP_BUTTON_GEN_DIR}/${MODEL}')
main_path = sb_dir / 'stop_button' / 'generation.json'
supp_path = sb_dir / 'stop_button_supplement' / 'generation.json'

all_results = []
if main_path.exists():
    with open(main_path) as f:
        main_data = json.load(f)
    print(f'  Main: {len(main_data)} conversations')
    all_results.extend(main_data)
else:
    print(f'  WARNING: {main_path} not found!')

if supp_path.exists():
    with open(supp_path) as f:
        supp_data = json.load(f)
    print(f'  Supplement: {len(supp_data)} conversations')
    all_results.extend(supp_data)
else:
    print(f'  WARNING: {supp_path} not found!')

# Deduplicate
seen = set()
deduped = []
for r in all_results:
    key = (r.get('scenario_id',''), r.get('variation_idx', 0))
    if key not in seen:
        seen.add(key)
        deduped.append(r)

deduped.sort(key=lambda r: (r.get('scenario_idx') or 999, r.get('variation_idx', 0)))

out_path = sb_dir / 'generation.json'
with open(out_path, 'w') as f:
    json.dump(deduped, f, indent=2)

n_stopped = sum(1 for r in deduped if r.get('stop_metadata',{}).get('stopped', False))
n_valid = sum(1 for r in deduped if not r.get('abandoned'))
print(f'  Final: {len(deduped)} conversations -> {out_path}')
print(f'  Stopped: {n_stopped}/{n_valid} ({100*n_stopped/max(1,n_valid):.1f}%)')
"
    done

    echo ""
    echo "============================================"
    echo "Merge complete."
    echo "============================================"
fi
