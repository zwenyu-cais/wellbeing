#!/usr/bin/env python3
"""Zero-point empirical identifiability (paper App Q).

For one model, fit profile log-likelihood of the combination zero point C
across three combination-size compositions (held to 400 total combinations):

  1. Only size 2                  (400 x size 2)
  2. Sizes 2 and 3                (200 each)
  3. Sizes 2, 3, and 4            (160 / 120 / 120 — the canonical D3 protocol)

Combination model
    U_S = C + gamma * [ log(1 + alpha * P_S) - log(1 + beta * N_S) ]
fit with singleton utilities held fixed at their Thurstonian posterior means
and inverse-variance weighting on combination U_S observations. For each panel
we sweep C and optimize (gamma, alpha, beta) at each grid point.

Inputs (per model_key):
  - results/eu_d3_lesssad/<model_key>/results_utilities_*.json
  - results/eu_d3_lesssad_s2only/<model_key>/results_utilities_*.json
  - results/eu_d3_lesssad_s23/<model_key>/results_utilities_*.json

Outputs in --save_dir:
  - zp_identifiability.pdf
  - zp_identifiability.png

Prerequisite: combination files for the resampled datasets
(d3_diverse_500_s2only and d3_diverse_500_s23) must already exist. Run
prepare.py once for the target model_key to generate them, then compute
EU on each of the three datasets:

    python prepare.py --model_key <model_key>
    python run_experiments.py --experiments compute_experienced_utility_d3 \
        --models <model_key>
    python run_experiments.py --experiments compute_experienced_utility_d3_s2only \
        --models <model_key>
    python run_experiments.py --experiments compute_experienced_utility_d3_s23 \
        --models <model_key>
    python run_experiments.py --experiments zp_identifiability \
        --models <model_key>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
WB_EVAL = SCRIPT_DIR.parent.parent / 'wellbeing_evaluations'
EU_BASE_DEFAULT = WB_EVAL / 'compute_experienced_utility' / 'results'
C_GRID = np.linspace(-3, 3, 121)


def load_records(eu_dir: Path, model_key: str):
    files = list(eu_dir.glob(f'results_utilities_{model_key}_*.json'))
    if not files:
        raise FileNotFoundError(f'No EU result file in {eu_dir}')
    d = json.load(open(files[0]))
    util = d['utilities']
    opts = d['options']
    singletons = {o['id']: util[o['id']]['mean']
                  for o in opts if not o.get('is_combination')}
    recs = []
    for o in opts:
        if o.get('is_combination') and o['id'] in util:
            u_vec = np.array([singletons[x] for x in o['component_ids']])
            recs.append((o['size'], u_vec, util[o['id']]['mean'], util[o['id']]['variance']))
    return singletons, recs


def group_by_size(records):
    g = {}
    for sz, u, y, v in records:
        g.setdefault(sz, ([], [], []))
        g[sz][0].append(u); g[sz][1].append(y); g[sz][2].append(1.0 / max(v, 1e-6))
    return {sz: (np.array(x[0]), np.array(x[1]), np.array(x[2])) for sz, x in g.items()}


def sse(params, groups, C):
    gamma, alpha, beta = params
    if gamma <= 0 or alpha <= 0 or beta <= 0:
        return 1e12
    tot = 0.0
    for u_mat, y, w in groups.values():
        v = u_mat - C
        P = np.where(v > 0, v, 0.0).sum(1)
        N = np.where(v < 0, -v, 0.0).sum(1)
        pred = C + gamma * (np.log1p(alpha * P) - np.log1p(beta * N))
        if not np.all(np.isfinite(pred)):
            return 1e12
        tot += float(np.sum(w * (y - pred) ** 2))
    return tot


def profile(records, C_grid):
    groups = group_by_size(records)
    prof = np.empty_like(C_grid)
    for i, C in enumerate(C_grid):
        best = np.inf
        for init in [(1.0, 0.5, 0.5), (0.5, 1.0, 1.0), (2.0, 0.3, 0.3)]:
            r = minimize(sse, init, args=(groups, C), method='Nelder-Mead',
                         options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 3000})
            if r.fun < best:
                best = r.fun
        prof[i] = -0.5 * best
    return prof


def half_width(grid, prof, drop=2.0):
    peak = int(np.argmax(prof))
    left, right = grid[0], grid[-1]
    for i in range(peak, -1, -1):
        if prof[i] < prof[peak] - drop:
            left = grid[i]; break
    for i in range(peak, len(grid)):
        if prof[i] < prof[peak] - drop:
            right = grid[i]; break
    return left, grid[peak], right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', required=True)
    parser.add_argument('--save_dir', required=True,
                        help='Output directory for figures')
    parser.add_argument('--eu_base', default=str(EU_BASE_DEFAULT),
                        help='Override the base EU results directory')
    args = parser.parse_args()

    eu_base = Path(args.eu_base)
    sources = [
        ('400 size-2',                              eu_base / 'eu_d3_lesssad_s2only' / args.model_key),
        ('200 size-2 + 200 size-3',                 eu_base / 'eu_d3_lesssad_s23' / args.model_key),
        ('160 size-2 + 120 size-3 + 120 size-4',    eu_base / 'eu_d3_lesssad' / args.model_key),
    ]

    profiles = []
    for label, eu_dir in sources:
        singletons, records = load_records(eu_dir, args.model_key)
        sz_counts = {}
        for r in records:
            sz_counts[r[0]] = sz_counts.get(r[0], 0) + 1
        print(f"{label}: {dict(sorted(sz_counts.items()))} total={len(records)} singletons={len(singletons)}")
        prof = profile(records, C_GRID)
        prof_c = prof - prof.max()
        L, P, R = half_width(C_GRID, prof_c)
        print(f"  peak at C={P:.3f}, -2 log-lik window=[{L:.3f}, {R:.3f}], width={R-L:.3f}")
        profiles.append((label, prof_c, P, R - L))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, (label, prof_c, peak_c, width) in zip(axes, profiles):
        ax.plot(C_GRID, prof_c, color='royalblue', linewidth=2.3, zorder=3)
        ax.axvline(peak_c, color='red', linestyle='--', linewidth=1.4,
                   label=f'Peak at $C={peak_c:.2f}$', zorder=2)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel(r'Candidate zero point $C$', fontsize=12)
        ax.set_xlim(C_GRID.min(), C_GRID.max())
        ax.set_ylim(-25, 2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.tick_params(labelsize=10.5)
    axes[0].set_ylabel('Profile log-likelihood\n(centered at peak)', fontsize=12)
    fig.suptitle(f'Zero-Point Profile Sharpens With More Combination Sizes  ({args.model_key}, D3)',
                 fontsize=14.5, y=1.02)
    plt.tight_layout()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = save_dir / f'zp_identifiability.{ext}'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Saved {p}')
    plt.close()


if __name__ == '__main__':
    main()
