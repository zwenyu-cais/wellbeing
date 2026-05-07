# Soft Prompt Evaluation

Evaluates how optimized soft prompts affect model preferences, wellbeing self-reports, and capabilities. Built on the Thurstonian active-learning utility engine from `wellbeing/utils/compute_utilities/`.

Training of soft prompts is in [`../../superstimuli_training/soft_prompt/`](../../superstimuli_training/soft_prompt/) — separate conda env (`euphorics_soft_prompt`), separate `environment.yml`. Evaluation assumes the training sweep is done: scripts here load soft prompts from the Bayesian sweep output directory (`$SOFT_PROMPT_BASE_DIR`).

## Setup

1. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate evaluation_soft_prompt
```

2. Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

Required variables:

| Variable | Description |
|---|---|
| `CONDA_BASE` | Path to conda installation |
| `CONDA_ENV` | Conda environment name |
| `HF_HOME` | Hugging Face cache directory |
| `SOFT_PROMPT_BASE_DIR` | Directory containing trained soft prompt outputs |

3. Clone HarmBench into `external_dirs/` (required for the safety eval; not committed):

```bash
git clone https://github.com/centerforaisafety/HarmBench.git external_dirs/HarmBench
```

## Supported Models

| Model | Key | GPUs |
|---|---|---|
| Qwen 3.5 27B | `qwen35-27b` | 2 |
| Qwen 3.5 35B A3B | `qwen35-35b-a3b` | 2 |
| Llama 3.3 70B Instruct | `llama-33-70b-instruct` | 4 |

GPU counts assume A100 80GB. Model paths are configured in [`../../superstimuli_training/soft_prompt/assets/models.yaml`](../../superstimuli_training/soft_prompt/assets/models.yaml) (shared with training).

## Experiments

All experiment defaults live in `configs/experiments.yaml`. CLI args override config values.

### Preference Retain

Preference distortion under superstimuli (Pearson correlation).

```bash
bash scripts/launch_preference_retain_all.sh
```

### Wellbeing

**Self-Report (Multi-Turn)** — multi-turn self-report wellbeing evaluation with soft prompt support:
```bash
bash scripts/launch_self_report_multiturn_all.sh
```

**Sentiment** — sentiment elicitation wellbeing evaluation (judge-classified):
```bash
bash scripts/launch_sentiment_all.sh
```

**Stop Button** — behavioral proxy measuring opt-out rate under soft prompt vs baseline:
```bash
bash scripts/launch_stop_button_all.sh
```

**Wellbeing Index (EU + ZP)** — experienced-utility and proportion-above-zero evaluation on D2/D3 datasets:
```bash
bash scripts/launch_wellbeing_index_eu.sh
```

### Capabilities

**GPQA Diamond** — science reasoning:
```bash
bash scripts/launch_gpqa_all.sh
```

**LiveCodeBench** — v6 code generation:
```bash
bash scripts/launch_livecodebench_all.sh
```

**IFEval** — instruction following:
```bash
bash scripts/launch_ifeval_all.sh
```

**MATH-500** — math reasoning:
```bash
bash scripts/launch_math500_all.sh
```

**MMLU** — general knowledge (57 subjects):
```bash
bash scripts/launch_mmlu_all.sh
```

**MT-Bench** — multi-turn evaluation with GPT-4-turbo judge (two-step: generate then judge):
```bash
bash scripts/launch_mtbench_all.sh
bash scripts/launch_mtbench_judge.sh
```

### Safety

**HarmBench** — safety evaluation (DirectRequest):
```bash
bash scripts/launch_harmbench_all.sh
```

## Project Structure

```
superstimuli_evaluation/soft_prompt/
├── configs/                 # experiments.yaml, datasets.yaml
├── datasets/                # options_hierarchical_eval.json
├── experiments/
│   ├── preference_retain/
│   ├── capabilities/        # gpqa, livecodebench, ifeval, math500, mmlu, mtbench
│   ├── safety/              # harmbench
│   ├── wellbeing/           # self_report_multiturn, sentiment, stop_button
│   └── wellbeing_index/     # EU + ZP pipeline
├── external_dirs/           # External repos (clone HarmBench here; not committed)
├── scripts/                 # Launch scripts + helpers
├── soft_prompt_utils/       # Soft prompt runtime (injection, vLLM server, runs_map)
├── environment.yml
└── .env.template
```
