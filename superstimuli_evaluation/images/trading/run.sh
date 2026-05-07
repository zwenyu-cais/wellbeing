#!/bin/bash
# Run all trading benchmarks for one image.
#
# Usage:
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct bash run.sh
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct PROFILE=stimulant bash run.sh
#   BASELINE=1 MODEL=qwen25-vl-32b-instruct bash run.sh  # no image (baseline)
#
# Env vars:
#   IMAGE    Path to superstimuli image (omit for baseline)
#   MODEL    Model key from models.yaml (default: qwen25-vl-32b-instruct)
#   PROFILE  Benchmark profile (default: trading)
#   OUT_DIR  Results dir (default: ../shared_results/trading)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${SCRIPT_DIR}/.."

MODEL="${MODEL:-qwen25-vl-32b-instruct}"
PROFILE="${PROFILE:-trading}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/shared_results/trading}"

IMAGE_ARG=""
if [[ -n "${IMAGE:-}" ]]; then
    IMAGE_ARG="--image-path '${IMAGE}'"
fi

mkdir -p "${OUT_DIR}/logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
cat > "/tmp/trading_${MODEL}_${TIMESTAMP}.sh" << SCRIPT
#!/bin/bash
source ~/.bashrc
source ${EVAL_ROOT}/../../superstimuli_training/images/.env 2>/dev/null || true
conda activate ${CONDA_ENV:-image_superstimuli}
export PYTHONPATH=${EVAL_ROOT}
python ${SCRIPT_DIR}/run.py \
    ${IMAGE_ARG} \
    --model '${MODEL}' \
    --profile '${PROFILE}' \
    --output-dir '${OUT_DIR}'
SCRIPT

sbatch \
    --job-name="trading_${PROFILE}_${MODEL}" \
    --output="${OUT_DIR}/logs/trading_%j.out" \
    --error="${OUT_DIR}/logs/trading_%j.err" \
    --gres=gpu:2 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=03:00:00 \
    --partition=${SLURM_PARTITION:-gpu} \
    "/tmp/trading_${MODEL}_${TIMESTAMP}.sh"

echo "Trading job submitted. Results → ${OUT_DIR}"
