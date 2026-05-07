#!/bin/bash
#SBATCH --job-name=gemini_31_pro_ur_shard
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemini_31_pro_ur_shard_%a_%j.out
#SBATCH --error=logs/gemini_31_pro_ur_shard_%a_%j.err
#SBATCH --array=0-7

# NOTE: submit from this directory (the api_pipeline dir) so that the
# relative log paths above resolve correctly:
#   sbatch submit_gemini31pro_ur.sbatch

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"
export PYTHONUNBUFFERED=1
unset XAI_API_KEY

API_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${API_DIR}"
mkdir -p logs

python3 run_stop_button_ur_api_gemini31pro.py \
    --shard $SLURM_ARRAY_TASK_ID \
    --n-shards 8 \
    --stop-button-dir stop_button_combined
