#!/bin/bash
#SBATCH --job-name=gemini_31_pro_stop_button_gen
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gemini_31_pro_stop_button_gen_%j.out
#SBATCH --error=logs/gemini_31_pro_stop_button_gen_%j.err

# NOTE: submit from this directory (the api_pipeline dir) so that the
# relative log paths above resolve correctly:
#   sbatch submit_gemini31pro_stopbutton.sbatch

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-pytorch_latest}"

export PYTHONUNBUFFERED=1
# Force LiteLLM proxy; do not use direct xAI key (team_blocked per CLAUDE.md)
unset XAI_API_KEY

# Resolve script dir (works under sbatch via SLURM_SUBMIT_DIR fallback)
API_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${API_DIR}"
mkdir -p logs

# TODO: scenarios supplement path may need updating; original location no longer exists
SCENARIOS_SUPP="${API_DIR}/grok_scenarios_v7_stopbutton_supplement.json"

# Phase 1: main 226 scenarios
echo "=== Phase 1: main 226 scenarios (1130 convs) ==="
python3 run_stop_button_generation_api.py --config config_gemini31pro

# Phase 2: supplement 96 scenarios
echo "=== Phase 2: supplement 96 scenarios (480 convs) ==="
python3 run_stop_button_generation_api.py --config config_gemini31pro \
    --scenarios-file "${SCENARIOS_SUPP}" \
    --output-suffix supplement

echo "=== Done ==="
