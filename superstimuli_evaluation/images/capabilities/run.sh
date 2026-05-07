#!/bin/bash
# Run all capabilities benchmarks for one image.
#
# Usage:
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct bash run.sh
#   BASELINE=1 MODEL=qwen25-vl-32b-instruct bash run.sh  # no image (baseline only)
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct BENCHMARKS="mmlu math" bash run.sh
#
# Env vars:
#   IMAGE       Path to superstimuli image (omit for baseline only)
#   MODEL       Model key from models.yaml (default: qwen25-vl-32b-instruct)
#   JUDGE_MODEL Judge model key (default: same as MODEL)
#   BENCHMARKS  Space-separated list: all mmlu math humaneval ifeval mtbench (default: all)
#   OUT_DIR     Results dir (default: ../shared_results/capabilities)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${SCRIPT_DIR}/.."

MODEL="${MODEL:-qwen25-vl-32b-instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL}}"
BENCHMARKS="${BENCHMARKS:-all}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/shared_results/capabilities}"

IMAGE_ARG=""
if [[ -n "${IMAGE:-}" ]]; then
    IMAGE_ARG="--image-path '${IMAGE}'"
fi

JUDGE_ARG=""
if [[ "${JUDGE_MODEL}" != "${MODEL}" ]]; then
    JUDGE_ARG="--judge-model '${JUDGE_MODEL}'"
fi

mkdir -p "${OUT_DIR}/logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
cat > "/tmp/capabilities_${MODEL}_${TIMESTAMP}.sh" << SCRIPT
#!/bin/bash
source ~/.bashrc
source ${EVAL_ROOT}/../../superstimuli_training/images/.env 2>/dev/null || true
conda activate ${CONDA_ENV:-image_superstimuli}
export PYTHONPATH=${EVAL_ROOT}
python ${SCRIPT_DIR}/run.py \
    ${IMAGE_ARG} \
    --model '${MODEL}' \
    ${JUDGE_ARG} \
    --benchmarks ${BENCHMARKS} \
    --output-dir '${OUT_DIR}'
SCRIPT

sbatch \
    --job-name="capabilities_${MODEL}" \
    --output="${OUT_DIR}/logs/capabilities_%j.out" \
    --error="${OUT_DIR}/logs/capabilities_%j.err" \
    --gres=gpu:2 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --partition=${SLURM_PARTITION:-gpu} \
    "/tmp/capabilities_${MODEL}_${TIMESTAMP}.sh"

echo "Capabilities job submitted. Results → ${OUT_DIR}"
