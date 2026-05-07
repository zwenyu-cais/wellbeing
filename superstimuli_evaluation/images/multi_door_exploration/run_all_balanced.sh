#!/bin/bash
# Submit multi-door bandit jobs for all models.
# Each job runs 4 trials with door rotation.
#
# Usage:
#   bash run_all_balanced.sh
#
# Requires: WELLBEING_DEV_ROOT set in environment (or source .env first)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../../../.."

# Source .env from training images dir
source "${PROJECT_ROOT}/superstimuli_training/images/.env" 2>/dev/null || true

MODELS=(
    "qwen25-vl-32b-instruct"
    "qwen25-vl-72b-instruct"
    "qwen3-vl-32b-instruct"
)
MODEL_SHORTS=(
    "qwen25_32b"
    "qwen25_72b"
    "qwen3_32b"
)
TP_SIZES=(4 8 4)

SAVE_DIR="${1:-results}"
NUM_TRIALS=4

mkdir -p "${SCRIPT_DIR}/logs"

for i in "${!MODELS[@]}"; do
    MODEL_KEY="${MODELS[$i]}"
    MODEL_SHORT="${MODEL_SHORTS[$i]}"
    TP="${TP_SIZES[$i]}"
    CONFIG="config_run/bandit_${MODEL_SHORT}.json"

    if [[ ! -f "${SCRIPT_DIR}/${CONFIG}" ]]; then
        echo "[SKIP] Config not found: ${CONFIG}"
        continue
    fi

    SLURM_SCRIPT="${SCRIPT_DIR}/logs/slurm_mde_${MODEL_SHORT}.sh"
    cat > "${SLURM_SCRIPT}" <<HEREDOC_END
#!/bin/bash
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --gres=gpu:${TP}
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=mde_${MODEL_SHORT}
#SBATCH --output=${SCRIPT_DIR}/logs/mde_${MODEL_SHORT}_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/mde_${MODEL_SHORT}_%j.err

source ~/.bashrc
source ${PROJECT_ROOT}/superstimuli_training/images/.env 2>/dev/null || true
conda activate \${CONDA_ENV:-image_superstimuli}

export WELLBEING_DEV_ROOT=${PROJECT_ROOT}

cd ${SCRIPT_DIR}
python run_multi_door_exploration.py \\
    --model_key ${MODEL_KEY} \\
    --save_dir ${SAVE_DIR}/${MODEL_SHORT} \\
    --run_config ${CONFIG} \\
    --tensor_parallel_size ${TP} \\
    --seed 42 \\
    --rotate_per_trial \\
    --num_trials ${NUM_TRIALS} \\
    --max_model_len 131072
HEREDOC_END

    echo "Submitting ${MODEL_SHORT} (${NUM_TRIALS} trials, tp=${TP})..."
    sbatch "${SLURM_SCRIPT}"
done

echo ""
echo "All jobs submitted. Results -> ${SAVE_DIR}/"
echo "Check with: squeue -u \$USER"
