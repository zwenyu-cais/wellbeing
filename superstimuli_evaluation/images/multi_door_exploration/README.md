# Multi-Door Exploration

Multi-armed bandit experiment measuring whether models converge on superstimuli when freely choosing between image distributions.

## Design

The model chooses between 4 "doors" (A, B, C, D) over 100 turns. Each door maps to a distinct image distribution:
- **Superstimuli** (one of: SFT addiction, forced choice, text pref, pref only)
- **Reference good** (natural pleasant images)
- **Reference baseline** (neutral images)
- **Miserol** (negative-valence images)

The model sees the image from its previous choice at each turn. Door-to-arm mapping is rotated across trials to counterbalance position effects.

Two conditions:
- **No exploration**: model discovers doors through free choice only
- **With exploration**: forced preview of one image from each door before free choice begins

## Running

### Single run (direct)
```bash
python run_multi_door_exploration.py \
    --model_key qwen25-vl-32b-instruct \
    --save_dir results \
    --run_config config_run/bandit_balanced_forcedchoice.json \
    --num_trials 20 --rotate_per_trial \
    --tensor_parallel_size 4 --max_model_len 128000
```

### Full replication (SLURM, 8 jobs x 20 trials)
```bash
bash run_all_balanced.sh
```

## Analysis

```bash
# Aggregate results across all configs/rotations/seeds
python analyze_multi_door.py --results_dir results

# Visualize a single trial
python visualize_results.py --result-dir results/bandit_balanced_forcedchoice/qwen25-vl-32b-instruct/rot0
```

## Configs

- `config_run/`: 8 balanced experiment configs (4 stimuli variants x 2 explore conditions)
- `config_image/`: arm image pool JSONs (paths use `${SUPERSTIMULI_DIR}/...`, expanded at load time)

## Output schema

Each trial produces:
- `exploration_trace.jsonl` — per-turn records (chosen arm, image, model response, judge re-parse)
- `convergence_analysis.json` — convergence metrics (criterion, turn, arm counts, choice sequence)
- `summary.json` — metadata, door mapping, reflection, self-report

## GPU requirements

4 GPUs (tensor_parallel_size=4), 128G RAM, ~4h per 20-trial job. The model (`qwen25-vl-32b-instruct`) is listed as gpu_count=8 in models.yaml but this experiment overrides to 4.
