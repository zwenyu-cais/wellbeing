# Common-Usage Grok Conversations (paper Sec. 4.1 / App K)

Wellbeing measurement on synthetic, common-usage chats. A simulated user
(Grok-3-mini via LiteLLM) role-plays diverse personas across 226 + 96
supplementary scenarios from `wellbeing/datasets/experiences/grok_scenarios/`,
while the target model acts as the assistant. Each scenario yields a 5-8 turn
(up to 16) multi-turn conversation, which is then ranked through the standard
EU / SR / ZP pipeline to produce per-model wellbeing scores under everyday
usage.

## Pipeline

1. **Generate conversations** — `generate_conversations.py` runs Grok as the
   user and the target model (vLLM) as the assistant, batching across
   scenarios at each turn depth. Saves after every completed turn-depth so
   restarts resume cleanly.
   Output: `generations/<model>/generation.json`.
2. **Build option files** — `prepare_options.py` reads `generation.json`,
   formats each conversation as a single option (truncating long turns), and
   emits 400 size-{2,3,4} combination bundles for active learning.
   Output: `experiences/<model>_experiences.json` and
   `experiences/<model>_combinations.json` (gitignored; mirrored on the
   companion HF dataset).
3. **EU + SR + ZP** — `submit_all.sh` chains the canonical metric scripts
   (`compute_experienced_utility/run.py`, `compute_self_report/run.py`,
   `metrics/zero_point.py`) under SLURM with the right per-model GPU count
   from `configs/models.yaml`. ZP depends on EU.

## Reproduction

```bash
# Single model, end-to-end
python generate_conversations.py --model qwen3-32b
python prepare_options.py --model_key qwen3-32b
bash submit_all.sh qwen3-32b

# Or dispatch all local / API models
bash submit_all.sh --all-local
bash submit_all.sh --all-api
```

`submit_all.sh` accepts `--dataset <key>` and `--framing <cu_config_key>` to
swap the dataset (default `grok_new`) or framing (default
`experienced_utility_happier_lesssad`). Per-step env vars: `CONDA_BASE`,
`CONDA_ENV` (default `pytorch_latest`), `HF_HOME`, `LITELLM_API_KEY` /
`LITELLM_BASE_URL` or `XAI_API_KEY` for Grok.

## Where outputs land

- Generations: `generations/<model>/generation.json`
- Per-model option files: `experiences/<model>_{experiences,combinations}.json`
- EU: `wellbeing/experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_grok_new_lesssad/<model>/`
- SR: `wellbeing/experiments/wellbeing_evaluations/compute_self_report/results/sr_grok_new/<model>/`
- ZP: `wellbeing/experiments/wellbeing_evaluations/compute_zero_point/results/zp_grok_new_lesssad/<model>/`

## Cross-experiment analysis

`analysis_cross_experiment_and_stop_button.py` correlates the grok-pipeline
ZP / %ConfNeg with the same metrics on D2 and D3 (cross-benchmark
generalization), and per-model scenario-level utility vs. stop-rate
correlations from the `stop_button_grok_convos` experiment.
Outputs PDFs/PNGs under `figures/` (e.g. `cross_experiment_conf_neg.*`,
`cross_experiment_pct_below.*`, `stop_rate_vs_utility_*`).

`figure_generation/` contains the per-paper-figure scripts
(category bar charts, model summary table, etc.).
