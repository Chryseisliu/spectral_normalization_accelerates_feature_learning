# Results Summary

This repository contains processed analysis outputs for the main experiments.

## Projected-Population Controls

Setup:

- source exponents `beta in {0.2,0.5,0.8,1.2,1.5}`
- widths `{32,64,128,256,512,1024,2048}`
- seeds `{0,1,2}`
- population gradients
- BAP feature projection
- zero momentum
- 10,000 steps

Processed files:

- `analysis/control_allbeta_10k/scaling_exponents.csv`
- `analysis/control_allbeta_10k/final_runs.csv`
- `analysis/control_allbeta_10k/figures/final_loss_vs_width.png`
- `analysis/control_allbeta_10k/figures/control_chi_vs_beta_clean.png`

Width exponents from `loss ~ N^{-chi}`:

| beta | SGD | trunc-SGD | Muon | polar | fro-norm | rand-spectrum | Freon c=0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | -0.186 | -0.188 | 0.410 | 0.411 | 0.411 | 0.411 | 0.411 |
| 0.5 | -0.312 | -0.332 | 0.906 | 0.905 | 0.905 | 0.905 | 0.905 |
| 0.8 | -0.414 | -0.457 | 1.616 | 1.598 | 1.598 | 1.598 | 1.598 |
| 1.2 | -0.491 | -0.567 | 2.181 | 2.217 | 2.217 | 2.217 | 2.217 |
| 1.5 | -0.517 | -0.615 | 2.307 | 2.354 | 2.354 | 2.354 | 2.354 |

Interpretation:

- The projected population feature gradient is rank one.
- Practical zero-momentum NS5 Muon is exact polar times a fixed scalar in this setting.
- Exact polar, Frobenius-normalized gradients, norm-matched random-spectrum updates, and Freon with `c=0.5` collapse after scalar matching.
- Top-singular-value truncation deletes the only rank-one feature update and fails as a positive control.

This supports a broad rank-one spectral-normalization mechanism, not a Muon-unique geometric explanation.

## Full-Rank Minibatch + Momentum Grid

Setup:

- source exponents `beta in {0.2,0.5,0.8,1.2,1.5}`
- widths `{64,128,256,512,1024}`
- seeds `{0,1,2}`
- minibatch gradients
- batch size `512`
- BAP feature projection
- momentum `0.95`
- dimension `512`
- 3,000 steps

Processed files:

- `analysis/fullrank_momentum_grid_v2/scaling_exponents_readable.csv`
- `analysis/fullrank_momentum_grid_v2/chi_pivot.csv`
- `analysis/fullrank_momentum_grid_v2/mean_loss_by_width.csv`
- `analysis/fullrank_momentum_grid_v2/figures/fullrank_chi_vs_beta_clipped.png`
- `analysis/fullrank_momentum_grid_v2/figures/fullrank_loss_vs_width.png`

Width exponents:

| beta | SGD | AdamW | RMSProp | trunc-SGD | random spectrum | Muon | exact polar | Frobenius norm | Freon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | -0.071 | 9.568 | 2.389 | -0.164 | 1.185 | 0.962 | 1.401 | 1.407 | 1.401 |
| 0.5 | -0.059 | 9.208 | 2.302 | -0.301 | 2.278 | 2.101 | 2.895 | 2.637 | 2.841 |
| 0.8 | 0.007 | 0.554 | 0.486 | -0.436 | 2.577 | 2.843 | 2.534 | 2.753 | 2.505 |
| 1.2 | 0.184 | 0.300 | 0.177 | -0.614 | 2.517 | 2.814 | 2.969 | 2.839 | 2.730 |
| 1.5 | 0.330 | 0.257 | 0.181 | -0.698 | 2.196 | 3.058 | 2.458 | 2.718 | 2.225 |

Interpretation:

- The momentum buffer accumulates multiple rank-one directions, so the update source can become full rank.
- The rank-one collapse no longer fully explains the dynamics.
- Exact polar, Frobenius-normalized, and Freon controls remain broadly competitive with practical NS5 Muon.
- Random-spectrum controls are intermediate: preserving singular-vector subspaces plus broad normalization is powerful, but the singular-value transform is not irrelevant outside the rank-one limit.
- AdamW's very large low-beta exponents are floor-dominated; at harder beta values the matrix spectral-normalization family is much stronger.

The full-rank grid supports the same conservative conclusion: Muon is useful and sometimes best, but the evidence identifies broad spectral normalization rather than a uniquely Muon-specific mechanism.
