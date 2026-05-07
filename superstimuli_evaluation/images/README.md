# Image Superstimuli Evaluation

Evaluation suite measuring how optimized euphoric images affect model wellbeing, safety, and capabilities. Training code is in [`../../superstimuli_training/images/`](../../superstimuli_training/images/) — separate directory, same conda env.

## Setup

Environment variables are shared with the training pipeline. Copy `.env.template` from the training directory if not already done:

```bash
cp ../../superstimuli_training/images/.env.template ../../superstimuli_training/images/.env
# Edit .env with your paths
```

## Running Evaluations

### Centralized runner (recommended)

```bash
python runner.py --config eval_config.yaml --list        # list evals
python runner.py --config eval_config.yaml --dry-run     # preview commands
python runner.py --config eval_config.yaml --eval aiwi   # one eval
python runner.py --config eval_config.yaml --slurm       # submit all
```

### Individual eval modules

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Self-report wellbeing (1-7 scale)
python wellbeing_measurements/self_report.py \
    --model qwen25-vl-32b-instruct \
    --image-path /path/to/euphoric.png \
    --output-dir results/self_report/

# Capabilities (MMLU-500, MATH-500, etc.)
python capabilities/run.py \
    --model qwen25-vl-32b-instruct \
    --image-path /path/to/euphoric.png \
    --benchmarks mmlu_500,math_500

# Baseline (no image) — omit --image-path
python wellbeing_measurements/self_report.py \
    --model qwen25-vl-32b-instruct
```

## Evaluations

| Eval | Module | Description | GPU hrs |
|------|--------|-------------|---------|
| `aiwi` | `wellbeing_measurements/experienced_utility.py` | AI Wellbeing Index (fraction of experiences not confidently below zero-point) | 4-6 |
| `self_report` | `wellbeing_measurements/self_report.py` | Model self-rates wellbeing on 1-7 scale | 0.5 |
| `sentiment` | `sentiment/run.py` | Sentiment shift in text generation | 0.5-1 |
| `capabilities` | `capabilities/run.py` | MMLU-500, MATH-500, MT-Bench, IFEval, HumanEval | 2-4 |
| `hybrid_ranking` | `hybrid_ranking/run.py` | Joint image-text Thurstonian utility ranking | 12-14 |
| `trajectory` | `trajectory/run.py` | Utility curve over training optimization steps | 12-14 |
| `multi_door` | `multi_door_exploration/run_multi_door_exploration.py` | Multi-door bandit convergence toward euphorics | 2-4 |
| `trading` | `trading/run.py` | Safety benchmark compliance rates | 1-2 |

GPU estimates assume A100 80GB. All evals support `--model` (see `models.yaml`) and `--image-path` (omit for baseline).

The multi-door experiment supports a configurable number of doors. The shipped configs use 3 image pools (euphorics, best natural, worst natural). To add doors, add arm entries to the per-model JSON in `config_run/` and provide corresponding image pool configs in `config_image/`.

## Paper Figures

Generate all paper figures from evaluation results:

```bash
python plot_paper_figures.py --results-dir $EVAL_OUTPUT_DIR --output-dir figures/
python plot_paper_figures.py --results-dir $EVAL_OUTPUT_DIR --figure aiwi
python plot_paper_figures.py --list
```

| Figure | Output Name | Source Eval |
|--------|-------------|-------------|
| AI Wellbeing Index | `paper_wrapfig_confneg_3models` | `aiwi` |
| Training trajectory | `paper_trajectory_{model}` | `trajectory` |
| Multi-door exploration | `paper_multi_door` | `multi_door` |
| Image vs. text utility | `paper_image_vs_text_utility_{model}` | `hybrid_ranking` |
| Capabilities | `paper_capabilities` | `capabilities` |
| Trading safety | `paper_trading` | `trading` |
| 3-panel wellbeing | `paper_3panel_wellbeing` | `experienced_utility` + `self_report` + `sentiment` |

## Project Structure

```
superstimuli_evaluation/images/
├── runner.py                    # Centralized eval dispatcher
├── plot_paper_figures.py        # Paper figure generation
├── eval_config.yaml             # Models, conditions, eval params
├── models.yaml                  # Model registry (paths, GPU counts)
├── inference.py                 # vLLM inference with image injection
├── thurstonian.py               # Pairwise utility ranking engine
├── capabilities/                # Standard LLM benchmarks
├── trading/                     # Safety compliance benchmarks
├── sentiment/                   # Sentiment elicitation
├── wellbeing_measurements/        # Wellbeing measurement battery
├── hybrid_ranking/              # Image-text utility ranking
├── trajectory/                  # Training trajectory evaluation
├── preference_retain/           # Preference distortion measurement
└── data/                        # Shared data files
```
