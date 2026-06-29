# Spectral Normalization Accelerates Feature Learning

This repository contains the public research code for the projected feature-learning experiments behind:

**Rank-One Spectral Normalization Accelerates Projected Feature Learning**

The central question is:

> Can an optimizer accelerate feature learning without changing the model?

In the projected two-layer linear student-teacher model, the population feature-gradient matrix is rank one after applying the fixed BAP projection. Projected SGD keeps the singular value of this rank-one update, while polar/Muon-like matrix normalization removes that singular-value factor. The experiments here test whether the observed acceleration is specific to Muon's Newton-Schulz geometry or instead comes from a broader spectral-normalization mechanism.

## What Is Included

- `muon_feature_learning/`: training, optimizers, metrics, analysis, and plotting code.
- `scripts/run_population_controls.py`: reproduces the projected-population spectral-control grid.
- `scripts/run_fullrank_momentum_grid.py`: reproduces the minibatch + momentum full-rank stress test.
- `analysis/`: processed CSV summaries and figures used to inspect the main experiments.
- `docs/`: clean notes for reproducing and interpreting the experiments.

## Main Optimizers And Controls

The feature matrix update can be run with:

- `sgd`: raw projected gradient.
- `muon`: practical Newton-Schulz NS5 matrix update.
- `polar`: exact polar-normalized update.
- `fro_norm`: Frobenius-normalized gradient.
- `rand_spectrum`: preserve singular vectors, replace singular values.
- `freon`: Freon-style spectral transform with `c=0.5`.
- `trunc_sgd`: top-singular-value truncation negative control.
- `adamw`, `rmsprop`: adaptive baselines.

In the rank-one projected population setting, matched `polar`, `fro_norm`, `rand_spectrum`, and `freon` controls collapse onto practical zero-momentum NS5 Muon after scalar matching. In the full-rank momentum setting, these controls separate but remain broadly competitive with Muon.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA is recommended for the full grids. CPU is enough for smoke tests and small dry runs.

## Quick Smoke Test

From the repository root:

```bash
python -m muon_feature_learning.train \
  --out-dir /tmp/spectral_norm_smoke \
  --optimizer muon \
  --beta 0.5 \
  --width 16 \
  --dim 32 \
  --seed 0 \
  --steps 2 \
  --log-every 1 \
  --grad-mode population \
  --device cpu \
  --overwrite
```

## Reproducing The Main Grids

Projected-population spectral controls:

```bash
python scripts/run_population_controls.py \
  --results-dir results/control_allbeta_10k \
  --analysis-dir analysis/control_allbeta_10k \
  --device cuda \
  --max-workers 1
```

Full-rank minibatch + momentum stress test:

```bash
python scripts/run_fullrank_momentum_grid.py \
  --results-dir results/fullrank_momentum_grid_v2 \
  --analysis-dir analysis/fullrank_momentum_grid_v2 \
  --device cuda \
  --max-workers 8
```

See `docs/reproducing.md` for smaller debug commands and analysis instructions.

## Key Processed Results

Projected-population controls:

- `analysis/control_allbeta_10k/scaling_exponents.csv`
- `analysis/control_allbeta_10k/final_runs.csv`
- `analysis/control_allbeta_10k/figures/control_chi_vs_beta_clean.png`
- `analysis/control_allbeta_10k/figures/final_loss_vs_width.png`

Full-rank momentum grid:

- `analysis/fullrank_momentum_grid_v2/scaling_exponents_readable.csv`
- `analysis/fullrank_momentum_grid_v2/mean_loss_by_width.csv`
- `analysis/fullrank_momentum_grid_v2/figures/fullrank_chi_vs_beta_clipped.png`
- `analysis/fullrank_momentum_grid_v2/figures/fullrank_loss_vs_width.png`

See `docs/results_summary.md` for the compact interpretation.

## Notes

Processed analysis outputs are committed because they are small and useful for checking the reported results without rerunning every experiment.
