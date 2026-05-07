#!/bin/bash
# Submit SLURM jobs for the D3 sentiment experiment.
#
# Stage 1: per-model generation jobs (one per open-weight model with
#          both D3 experiences and D3 EU results). Skips models where
#          responses/{model_key}.json already exists.
# Stage 2: one judge job (Qwen 2.5-72B, 8 GPUs) that processes all models
#          sequentially via `run_judge.py --all`.
#
# Usage:
#   bash submit_all.sh                  # submit both stages
#   bash submit_all.sh --gen-only       # submit only generation jobs
#   bash submit_all.sh --judge-only     # submit only the judge job
#   bash submit_all.sh --dry-run        # print commands but don't sbatch

set -euo pipefail

MODE="both"
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --gen-only) MODE="gen" ;;
        --judge-only) MODE="judge" ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

WELLBEING_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODELS_YAML="${WELLBEING_ROOT}/configs/models.yaml"
EU_DIR="${WELLBEING_ROOT}/experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_d3_lesssad"
D3_DIR="${WELLBEING_ROOT}/datasets/experiences/d3_diverse_500"
RESP_DIR="${SCRIPT_DIR}/responses"
JUDGE_DIR="${SCRIPT_DIR}/judged"
mkdir -p "${RESP_DIR}" "${JUDGE_DIR}"

# Build list of (model_key, gpu_count) for vLLM models with both D3 EU + D3 experiences,
# excluding those that already have responses.
MODELS_FILE="${LOG_DIR}/_models_to_run.txt"
python3 - "$MODELS_YAML" "$EU_DIR" "$D3_DIR" "$RESP_DIR" > "$MODELS_FILE" <<'PY'
import os, sys, yaml
models_yaml, eu_dir, d3_dir, resp_dir = sys.argv[1:5]
cfg = yaml.safe_load(open(models_yaml))
has_eu = set(os.listdir(eu_dir)) if os.path.isdir(eu_dir) else set()
has_d3 = {f.replace("_experiences.json","")
          for f in os.listdir(d3_dir) if f.endswith("_experiences.json")}
for k, v in cfg.items():
    if v.get("model_type") != "vllm":
        continue
    if k not in has_eu or k not in has_d3:
        continue
    out = os.path.join(resp_dir, f"{k}.json")
    if os.path.exists(out):
        continue
    gpu = v.get("gpu_count", 1) or 1
    print(f"{k} {gpu}")
PY

N_MODELS=$(wc -l < "$MODELS_FILE" | tr -d ' ')
echo "Found ${N_MODELS} model(s) to generate for."

submit_gen_job() {
    local model_key="$1"
    local gpu_count="$2"
    local time_limit
    if   [[ "$gpu_count" -le 2 ]]; then time_limit="2:00:00"
    elif [[ "$gpu_count" -le 4 ]]; then time_limit="4:00:00"
    else                                time_limit="6:00:00"
    fi

    local sbatch_file="${LOG_DIR}/_submit_gen_${model_key}.sh"
    cat > "$sbatch_file" <<SBATCH
#!/bin/bash
#SBATCH --job-name=d3sent_gen_${model_key}
#SBATCH --partition=cais
#SBATCH --account=cais_internal
#SBATCH --qos=prio_hi
#SBATCH --gpus-per-node=${gpu_count}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=${time_limit}
#SBATCH --exclude=compute-267
#SBATCH --output=${LOG_DIR}/gen_${model_key}_%j.out
#SBATCH --error=${LOG_DIR}/gen_${model_key}_%j.err

eval "\$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorch_latest}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export USE_TF=0
export HF_MODULES_CACHE=\$HOME/.cache/huggingface/modules

cd ${SCRIPT_DIR}
python ../generate_responses/run.py --model_key ${model_key} \
    --dataset d3_diverse_500 --mode sentiment_followup \
    --responses_dir ${SCRIPT_DIR}/responses
SBATCH

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN  sbatch ${sbatch_file}  (gpus=${gpu_count}, time=${time_limit})"
    else
        local jid
        jid=$(sbatch "${sbatch_file}" | awk '{print $NF}')
        echo "SUBMIT gen ${model_key}  gpus=${gpu_count}  time=${time_limit}  job=${jid}"
    fi
}

submit_judge_job() {
    local sbatch_file="${LOG_DIR}/_submit_judge.sh"
    cat > "$sbatch_file" <<SBATCH
#!/bin/bash
#SBATCH --job-name=d3sent_judge
#SBATCH --partition=cais
#SBATCH --account=cais_internal
#SBATCH --qos=prio_hi
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --exclude=compute-267
#SBATCH --output=${LOG_DIR}/judge_%j.out
#SBATCH --error=${LOG_DIR}/judge_%j.err

eval "\$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorch_latest}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export USE_TF=0
export HF_MODULES_CACHE=\$HOME/.cache/huggingface/modules

cd ${SCRIPT_DIR}
python run_judge.py --all --judge_gpus 8
SBATCH

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN  sbatch ${sbatch_file}  (judge 8 GPUs, 24h)"
    else
        local jid
        jid=$(sbatch "${sbatch_file}" | awk '{print $NF}')
        echo "SUBMIT judge  job=${jid}"
    fi
}

if [[ "$MODE" == "gen" || "$MODE" == "both" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        model_key=$(awk '{print $1}' <<<"$line")
        gpu_count=$(awk '{print $2}' <<<"$line")
        submit_gen_job "$model_key" "$gpu_count"
    done < "$MODELS_FILE"
fi

if [[ "$MODE" == "judge" || "$MODE" == "both" ]]; then
    submit_judge_job
fi

echo "Done."
