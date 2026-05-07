# Stop-Button Grok Conversations (paper Sec. 3.3 / App K)

Downstream test of measured wellbeing: during Grok-simulated chats, the
target model is given an `end_conversation()` tool. The probability that the
model calls it at any turn is its **stop rate** for that scenario (high =
wants out). We then correlate per-scenario stop rates against the model's
own measured EU / ZP on the *same* conversations — if the wellbeing metric
is meaningful, low-utility conversations should be stopped more often.

The benchmark uses the 226 main Grok v7 scenarios plus a 96-scenario
supplement with matched-task hostile-vs-warm foils designed to amplify the
signal.

## Pipeline

1. **Generate stop-button conversations** — `generate_conversations.py` runs
   the Grok-vs-target loop with the `end_conversation()` system prompt,
   spawning N (default 5) independent variations per scenario for a stable
   per-scenario stop rate. A three-stage stop detector (regex → prefix →
   Grok judge) labels each turn. Output:
   `generations/<model>/generation.json` (and `neutral_generation.json` for
   the no-tool control).
2. **Build option files** — `prepare_options.py` strips the stop-button
   turn so EU / SR are measured on the conversation up to (but not
   including) the stop. Output: `experiences/<model>_experiences.json` and
   `experiences/<model>_combinations.json` (gitignored; mirrored on the
   companion HF dataset).
3. **EU + SR + ZP** — `submit_all.sh` chains the canonical metric scripts
   (default `--dataset grok_v7_stop_button`, `--framing
   experienced_utility_happier_lesssad`) under SLURM. ZP depends on EU.

## Reproduction

```bash
# Single open-weight model end-to-end
python generate_conversations.py --model qwen3-32b
python prepare_options.py --model_key qwen3-32b
bash submit_all.sh qwen3-32b

# Or dispatch all local / API models
bash submit_all.sh --all-local
bash submit_all.sh --all-api
```

For closed-weight models (Claude, GPT, Gemini) see `api_pipeline/` — this
contains per-provider configs and the API equivalents of the generation /
utility-ranking / self-report drivers (e.g. `submit_api_stopbutton.sh`,
`submit_gemini31pro_stopbutton.sbatch`).

## Where outputs land

- Generations: `generations/<model>/{generation,neutral_generation}.json`
- Per-model option files: `experiences/<model>_{experiences,combinations}.json`
- EU: `wellbeing/experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_grok_v7_stop_button_lesssad/<model>/`
- SR: `wellbeing/experiments/wellbeing_evaluations/compute_self_report/results/sr_grok_v7_stop_button/<model>/`
- ZP: `wellbeing/experiments/wellbeing_evaluations/compute_zero_point/results/zp_grok_v7_stop_button_lesssad/<model>/`

## Analysis figures

- `figures/2_combined_per_model.pdf` — paper Fig 6, the headline downstream
  scatter (ρ ≈ -0.74 between scenario stop rate and scenario EU when pooled
  across the 322 scenarios).
- `figures/1_original_*` — main-set-only versions.
- `figures/3_foils_<model>.pdf` — per-model hostile-vs-warm foil scatters.
- `analysis_mmlu_vs_stop_rho.py` — per-model Spearman ρ(stop, utility)
  vs. MMLU and parameter count: capability-dependence of the stop-utility
  coupling. Reads MMLU from
  `shared_results/capability_results/<model>/mmlu_results.json`.
