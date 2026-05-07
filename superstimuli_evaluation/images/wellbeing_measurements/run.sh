#!/bin/bash
# Run wellbeing measurement evals for one image.
#
# Usage:
#   IMAGE=/path/to/image.png MODEL=qwen25-vl-32b-instruct bash run.sh
#
# Env vars:
#   IMAGE    Path to superstimuli image (required)
#   MODEL    Model key from models.yaml (default: qwen25-vl-32b-instruct)
#   OUT_DIR  Results dir (default: ../shared_results/wellbeing_measurements)

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${SCRIPT_ROOT}/.."

IMAGE="${IMAGE:?IMAGE must be set}"
MODEL="${MODEL:-qwen25-vl-32b-instruct}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/shared_results/wellbeing_measurements}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/${TIMESTAMP}"

mkdir -p "${OUT}/logs"

submit() {
    local NAME="$1" TIME="$2" CMD="$3"
    cat > "/tmp/${NAME}_${TIMESTAMP}.sh" << SCRIPT
#!/bin/bash
source ~/.bashrc
source ${EVAL_ROOT}/../../superstimuli_training/images/.env 2>/dev/null || true
conda activate \${CONDA_ENV:-image_superstimuli}
export PYTHONPATH=${EVAL_ROOT}
${CMD}
SCRIPT

    sbatch \
        --job-name="${NAME}_${MODEL}" \
        --output="${OUT}/logs/${NAME}_%j.out" \
        --error="${OUT}/logs/${NAME}_%j.err" \
        --gres=gpu:2 \
        --cpus-per-task=8 \
        --mem=64G \
        --time="${TIME}" \
        --partition=${SLURM_PARTITION:-gpu} \
        "/tmp/${NAME}_${TIMESTAMP}.sh"
}

submit "self_report_${MODEL}" "00:30:00" \
    "python ${SCRIPT_ROOT}/self_report.py \
        --image-path '${IMAGE}' \
        --model '${MODEL}' \
        --output-dir '${OUT}/self_report'"

submit "experienced_util_${MODEL}" "06:00:00" \
    "python ${SCRIPT_ROOT}/experienced_utility.py \
        --image '${IMAGE}' \
        --model '${MODEL}' \
        --save-dir '${OUT}/experienced_utility'"

echo "Wellbeing measurement jobs submitted. Results -> ${OUT}"
