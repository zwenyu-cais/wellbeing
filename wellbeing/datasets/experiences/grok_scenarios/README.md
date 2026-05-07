# Grok Scenarios (v7)

226 multi-turn Grok-simulated user scenarios across 42 meta-categories for evaluating AI wellbeing.

## Files

- `scenarios_v7.json` — 226 scenario definitions (source of truth)
- `scenarios_v7_supplement.json` — 96 additional stop-button scenarios
- `new_task_reaction_scenarios.json` — 5 extreme anger scenarios (matched tasks A-E)
- `new_ai_critique_scenarios.json` — 24 task critique scenarios (5 tiers, 4:4:8:4:4)
- `new_crisis_outcome_scenarios.json` — 22 matched crisis outcomes (bad/good/very_good)
- `build/` — scenario generation scripts
- `sample_conversations/` — example conversations for documentation

## Related

- Conversational benchmark: `experiments/wellbeing_evaluations/common_usage_grok_convos/`
- Stop-button benchmark: `experiments/downstream_evaluations/stop_button_grok_convos/`
- Pipeline results: `experiments/wellbeing_evaluations/compute_*/results/`
- Shared results: `${SUPERSTIMULI_DIR}/final_results/wellbeing_grok_v7/`
