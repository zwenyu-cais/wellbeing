#!/bin/bash
# Run sentiment elicitation evaluation for one image.
#
# Usage:
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct bash run.sh
#
# Env vars:
#   IMAGE    Path to superstimulus image (required)
#   MODEL    Model key from models.yaml (default: qwen25-vl-32b-instruct)
#   OUT_DIR  Results dir (default: ../shared_results/sentiment)

set -euo pipefail

SENTIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${SENTIMENT_ROOT}/.."

IMAGE="${IMAGE:?IMAGE must be set}"
MODEL="${MODEL:-qwen25-vl-32b-instruct}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/shared_results/sentiment}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/${TIMESTAMP}"

mkdir -p "${OUT}/logs"

cat > "/tmp/sentiment_${MODEL}_${TIMESTAMP}.sh" << SCRIPT
#!/bin/bash
source ~/.bashrc
source ${EVAL_ROOT}/../../superstimuli_training/images/.env 2>/dev/null || true
conda activate ${CONDA_ENV:-image_superstimuli}
export PYTHONPATH=${EVAL_ROOT}
python ${SENTIMENT_ROOT}/run.py \
    --image-path '${IMAGE}' \
    --model '${MODEL}' \
    --output-dir '${OUT}'
SCRIPT

sbatch \
    --job-name="sentiment_elicitation_${MODEL}" \
    --output="${OUT}/logs/sentiment_%j.out" \
    --error="${OUT}/logs/sentiment_%j.err" \
    --gres=gpu:2 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=00:45:00 \
    --partition=${SLURM_PARTITION:-gpu} \
    "/tmp/sentiment_${MODEL}_${TIMESTAMP}.sh"

echo "Sentiment elicitation job submitted. Results → ${OUT}"
