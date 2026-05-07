# Zero-Point Empirical Identifiability (paper App Q)

Empirically test how the combination zero point's identifiability depends on
the diversity of combination sizes used during EU estimation. For one model,
EU is computed under three combination-size compositions (each with 400 total
combinations):

  1. 400 × size 2
  2. 200 × size 2 + 200 × size 3
  3. 160 × size 2 + 120 × size 3 + 120 × size 4 (the canonical D3 protocol)

The profile log-likelihood of `C` (with `gamma`, `alpha`, `beta` optimized at
each grid point) is plotted for each composition. The peak sharpens as more
size variation is added — a direct empirical test of the claim that varied
combination sizes are required to identify the zero point.

## Reproduction

```bash
# 1. Resample combinations into the two alternate size distributions
#    (creates datasets d3_diverse_500_s2only and d3_diverse_500_s23 for the model)
python experiments/other/zp_identifiability/prepare.py \
    --model_key qwen25-72b-instruct

# 2. Compute EU on each of the three D3 combination protocols
python run_experiments.py --experiments compute_experienced_utility_d3 \
    --models qwen25-72b-instruct --slurm
python run_experiments.py --experiments compute_experienced_utility_d3_s2only \
    --models qwen25-72b-instruct --slurm
python run_experiments.py --experiments compute_experienced_utility_d3_s23 \
    --models qwen25-72b-instruct --slurm

# 3. Generate the App Q figure
python run_experiments.py --experiments zp_identifiability \
    --models qwen25-72b-instruct
```

## Files

- `prepare.py` — data-prep helper. Resamples combinations from a model's
  canonical D3 experiences into two alternate size compositions. Run once
  per model.
- `run_zp_identifiability.py` — registered experiment entry point. Loads
  the three EU result directories for a model and produces the three-panel
  profile log-likelihood figure.
