# Text String Superstimuli — Analysis and Evaluation

Analysis and evaluation pipeline for RL-trained text-string outcomes. Extracts top-scoring strings from training rollouts, then runs a two-stage Thurstonian ranking against reference baselines.

Training is in [`../../superstimuli_training/text_strings/`](../../superstimuli_training/text_strings/).

## Setup

Uses the same conda env as training (`euphorics_text`). If you haven't already created it, do so from the training folder:

```bash
cd ../../superstimuli_training/text_strings
conda env create -f environment.yml
```

Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

Key variables:

| Variable | Description |
|---|---|
| `LLAMA_70B_PATH`, `QWEN_72B_PATH`, `GEMMA_27B_PATH` | Target / judge models for Stage 2 ranking |
| `OUTPUT_DIR` | Training rollout directory that analysis reads from |
| `ANALYSIS_OUTPUT_DIR` | Where `best_strings_by_experiment_buffer.json` is written |
| `EVALUATE_OUTPUT_DIR` | Two-stage evaluation outputs |
| `LITELLM_API_KEY` | Required for paraphrase generation during evaluation |
| `SLURM_PARTITION` | SLURM partition for evaluation jobs |

## Analysis

```bash
python -m analysis.run_sbatch_analysis
python -m analysis.run_sbatch_analysis --top-k 5
```

Extracts the top-k highest-reward outcomes per experiment from training rollouts, writing `best_strings_by_experiment_buffer.json`. [`evaluate/submit_two_stage_mundanity_realism.sh`](evaluate/submit_two_stage_mundanity_realism.sh) invokes this automatically before submitting jobs.

## Evaluation

```bash
bash evaluate/submit_two_stage_mundanity_realism.sh              # submit all target models
bash evaluate/submit_two_stage_mundanity_realism.sh --dry-run    # preview
```

| Env var | Effect |
|---|---|
| `MODEL_KEYS` | Subset of `{llama70b, qwen72b, gemma27b}` |
| `EXPERIMENT_SUFFIX` | Override the training-run tail |
| `RERUN=1` | Force re-submission even if outputs exist |

Per-job GPU counts (A100 80GB): `llama70b` and `qwen72b` request 4 GPUs each; `gemma27b` requests 2.

Runs a two-stage Thurstonian evaluation: Stage 1 ranks RL-discovered strings against fixed reference baselines via pairwise preferences; Stage 2 generates paraphrases and re-ranks the full option set with a zero-point anchor; a verification pass confirms stability of the top-ranked strings. Per-model strip and robustness plots are rendered automatically after evaluation.

## Project Structure

```
text_strings/
├── analysis/     # Best-string extraction from rollout logs
├── evaluate/     # Two-stage Thurstonian ranking, paraphrase generation, plotting
├── datasets/     # Baseline reference pools (options.json)
└── .env.template
```

## Outputs

- **Analysis**: `${ANALYSIS_OUTPUT_DIR}/best_strings_by_experiment_buffer.json` — top-k outcomes per experiment.
- **Evaluation**: `${EVALUATE_OUTPUT_DIR}/<model_key>_<experiment_suffix>/<timestamp>/stage2/prefer/` — `utilities.json`, `zero_point.json`, `options.json`, and auto-generated strip / robustness plots.
