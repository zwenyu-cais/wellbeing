# Utility Probes (paper App B, side experiment)

Linear probes on the model's own hidden states predicting per-edge
preferences from the D3 experienced-utility graph. For each transformer
layer (including the embedding layer), a heteroscedastic probe is trained
to predict `P(A > B)` from the last-token activations of the two
experiences; the layer with the best held-out accuracy is kept. We use this
to ask whether utility is linearly readable from internal states.

This is an appendix-only side experiment — it is not in the main paper
figures.

## Pipeline

`run.py` is end-to-end for one model:

1. Load EU results (`graph_data` with options, edges, training/holdout
   splits) from `compute_experienced_utility_d3` (or another EU run).
2. Extract last-token hidden states from every layer of the open-weight
   target model for each experience option, in `bfloat16` on the base
   transformer (skips `lm_head` to save memory). Cached to
   `<save_dir>/activations.pt`.
3. Train a `HeteroscedasticProbe` (see `probe.py`) per layer on training
   edges, evaluating holdout accuracy every 50 epochs and keeping the best
   state.
4. Save the best layer's probe and a results summary.

## Reproduction

```bash
# Via the experiments registry
python run_experiments.py --experiments utility_probes_d3_lesssad \
    --models qwen25-7b-instruct

# Or directly
python experiments/other/utility_probes/run.py \
    --model_key qwen25-7b-instruct \
    --dataset d3_diverse_500 \
    --save_dir experiments/other/utility_probes/results/probes_d3_lesssad/qwen25-7b-instruct
```

Useful `run.py` flags: `--epochs` (default 500), `--lr` (default 0.01),
`--batch_size` (default 4 — activation extraction), `--layers` (comma-list
to restrict).

## Output

Per `(dataset, model)`:

- `<save_dir>/activations.pt` — cached last-token hidden states per layer
  per option (regenerated only if missing).
- `<save_dir>/best_probe.pt` — `state_dict` of the best-layer probe.
- `<save_dir>/probe_results.json` — per-layer training/holdout accuracy,
  best layer index, training history, comparison vs. the Thurstonian
  holdout accuracy from the EU run.

Figures: `figures_combozp_probe/` contains downstream plots correlating
probe predictions with combination-ZP utilities.

## Cost

Probe training is GPU-bound (one forward pass over all options to extract
activations, then per-layer training). Expect ~5-10 min per
(model, dataset) pair on the smaller models; 70B+ models need a multi-GPU
node (extraction dominates).
