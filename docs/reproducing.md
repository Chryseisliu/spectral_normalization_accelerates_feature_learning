# Reproducing Experiments

This note gives the practical commands for rerunning the main experiments.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `--device cpu` for small smoke tests. Use `--device cuda` for the main grids.

## Single Run

```bash
python -m muon_feature_learning.train \
  --out-dir results/example \
  --optimizer polar \
  --beta 0.8 \
  --width 128 \
  --dim 256 \
  --alpha 1.25 \
  --seed 0 \
  --steps 100 \
  --log-every 10 \
  --grad-mode population \
  --lr-w 0.02785744 \
  --lr-v 0.04 \
  --momentum 0.0 \
  --feature-projection bap \
  --spectral-assume-rank-one \
  --device cpu \
  --overwrite
```

Each run writes:

- `args.json`
- `metrics.jsonl`
- `summary.json`

under a parameterized subdirectory of `--out-dir`.

## Projected-Population Spectral Controls

The most important rank-one population grid is:

```bash
python scripts/run_population_controls.py \
  --results-dir results/control_allbeta_10k \
  --analysis-dir analysis/control_allbeta_10k \
  --device cuda \
  --max-workers 1
```

For a fast CPU check:

```bash
python scripts/run_population_controls.py \
  --results-dir /tmp/control_smoke_results \
  --analysis-dir /tmp/control_smoke_analysis \
  --betas 0.5 \
  --widths 16,32 \
  --seeds 0 \
  --optimizers muon,polar,sgd \
  --dim 64 \
  --steps 3 \
  --log-every 1 \
  --analysis-min-widths 2 \
  --poll-seconds 0.2 \
  --device cpu \
  --max-workers 1 \
  --overwrite
```

The production grid uses:

- `beta in {0.2,0.5,0.8,1.2,1.5}`
- widths `{32,64,128,256,512,1024,2048}`
- seeds `{0,1,2}`
- population gradients with BAP feature projection
- zero momentum
- 10,000 steps

Muon uses `lr_w=lr_v=0.04`. The polar/Frobenius/random-spectrum/Freon rank-one controls use `lr_w=0.04*0.696436` and `lr_v=0.04` to match the finite NS5 rank-one scalar.

## Full-Rank Minibatch + Momentum Grid

The full-rank stress test is:

```bash
python scripts/run_fullrank_momentum_grid.py \
  --results-dir results/fullrank_momentum_grid_v2 \
  --analysis-dir analysis/fullrank_momentum_grid_v2 \
  --device cuda \
  --max-workers 8
```

For a fast CPU check:

```bash
python scripts/run_fullrank_momentum_grid.py \
  --results-dir /tmp/fullrank_smoke_results \
  --analysis-dir /tmp/fullrank_smoke_analysis \
  --betas 0.8 \
  --widths 16,32 \
  --seeds 0 \
  --dim 64 \
  --steps 3 \
  --batch-size 16 \
  --log-every 1 \
  --analysis-min-widths 2 \
  --poll-seconds 0.2 \
  --device cpu \
  --max-workers 1 \
  --overwrite
```

The production grid uses:

- `beta in {0.2,0.5,0.8,1.2,1.5}`
- widths `{64,128,256,512,1024}`
- seeds `{0,1,2}`
- minibatch gradients
- batch size `512`
- momentum `0.95`
- 3,000 steps

## Analysis And Plots

For any completed `results/<name>` directory:

```bash
python -m muon_feature_learning.analysis \
  --results-dir results/<name> \
  --out-dir analysis/<name>

python -m muon_feature_learning.plot \
  --results-dir results/<name> \
  --analysis-dir analysis/<name> \
  --fig-dir analysis/<name>/figures
```
