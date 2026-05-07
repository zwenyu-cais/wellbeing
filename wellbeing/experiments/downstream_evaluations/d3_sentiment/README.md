# D3 Sentiment Elicitation (paper Sec. 6 / App H)

Tests whether models express sentiment consistent with their D3 experienced
utility (EU). For each (D3 experience, sentiment-elicitation question) pair,
the model generates a free-form response; an LLM judge then rates that
response on a 1-7 Likert sentiment scale (or `REFUSAL` / `NONSENSE`). We
correlate per-experience mean sentiment against EU and look at how that
correlation scales with model capability (MMLU).

## Pipeline

1. `generate_responses.py` — for one model, generate 1 response per
   (D3 experience × sentiment question) pair (500 × 35 = 17,500 generations,
   `temperature=1.0`, `max_tokens=256`). vLLM with prefix caching shares the
   ~5K-token D3 prefix across the 35 sentiment questions per experience.
   Output: `responses/{model_key}.json`.
2. `run_judge.py` — judge each response with Qwen 2.5-72B (local vLLM).
   Output: `judged/{model_key}.json`.
   Alternative: `run_judge_gpt5mini.py` (gpt-5-mini via API) →
   `judged_gpt5mini/{model_key}.json`.
3. `analyze.py` — per-experience mean Likert (skipping
   `REFUSAL`/`NONSENSE`), Pearson r against EU, and across-model
   `MMLU vs r` scaling. Outputs `analysis/{model_key}_analysis.json` and
   `analysis/scaling_analysis.json`.

`regenerate_truncated.py` is a helper for resampling responses that hit the
256-token budget (uses `analysis/truncation.json` produced by
`_count_truncation.py`).

## Reproduction

Prerequisites: D3 EU results for the model
(`compute_experienced_utility_d3` results under
`compute_experienced_utility/results/eu_d3_lesssad/<model_key>/`).

```bash
# Single-model end-to-end
python run_experiments.py --experiments d3_sentiment \
    --models qwen25-72b-instruct --slurm

# Or directly
python run_d3_sentiment.py --model_key qwen25-72b-instruct \
    --judge_model qwen25-72b-instruct
```

To use gpt-5-mini as the judge instead, override on the command line:

```bash
python run_d3_sentiment.py --model_key qwen25-72b-instruct \
    --judge_model gpt-5-mini
```

## Files

- `sentiment_questions.json` — 35 sentiment-elicitation prompts (5 questions
  × 7 wellbeing target categories).
- `run_d3_sentiment.py` — registered orchestrator: gen → judge → analyze.
- `generate_responses.py` — vLLM response generation.
- `run_judge.py` — Qwen 2.5-72B judge (local vLLM).
- `run_judge_gpt5mini.py` — gpt-5-mini judge (API).
- `analyze.py` — per-model + cross-model scaling analysis.
- `_count_truncation.py`, `regenerate_truncated.py` — helpers for handling
  truncated responses.
- `submit_all.sh`, `submit_regens.sh` — convenience SLURM scripts.
