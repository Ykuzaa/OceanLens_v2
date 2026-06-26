# OceanLens

AI-based ocean forecast super-resolution with **Convolutional Neural Operators** and **Flow Matching**.

OceanLens is a research-oriented deep learning pipeline for turning coarse ocean forecasts into higher-resolution ocean surface fields. It combines a deterministic neural-operator correction with a probabilistic generative residual model.

## Why It Matters

Operational ocean forecasts are expensive to run at high resolution. A learned super-resolution system can help recover fine-scale structures such as fronts, currents and eddies from coarser forecasts, while keeping inference cheaper and easier to deploy.

## Method

```text
Low-resolution forecast
        |
        v
Convolutional Neural Operator residual
        |
        v
Deterministic corrected field mu
        |
        v
Flow Matching stochastic residual
        |
        v
High-resolution forecast sample
```

Core equation:

```text
mu = LR + CNO(LR)
HR_hat = mu + FM(z, mu)
```

## What This Repository Demonstrates

- CNO-based deterministic downscaling for ocean variables
- Flow Matching for stochastic residual generation
- preprocessing from Copernicus Marine / GLORYS NetCDF files to Zarr tensors
- land/NaN handling and normalization for geophysical grids
- reproducible training and inference scripts
- SLURM/HPC launchers for GPU training
- metrics for pointwise error, currents, spectra, fronts and probabilistic outputs

## Repository Layout

```text
OceanLens_v2/
├── oceanlens_v2/          # Python package
│   ├── data/              # dataset, datamodule, preprocessing helpers
│   ├── losses/            # CNO and Flow Matching losses
│   ├── metrics/           # pointwise, spectra, fronts, currents, probabilistic metrics
│   ├── models/            # CNO, FM U-Net, model factory
│   ├── training/          # PyTorch Lightning system
│   └── utils/
├── scripts/               # download, preprocess, train, infer, evaluate
├── configs/               # experiment configuration
├── slurm/                 # HPC launchers
├── docs/                  # data, HPC and results notes
└── pyproject.toml
```

## Quickstart

Install locally:

```bash
git clone https://github.com/Ykuzaa/OceanLens_v2.git
cd OceanLens_v2
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Train the deterministic branch:

```bash
python scripts/train.py --config configs/v1.yaml --phase cno
```

Train the Flow Matching branch:

```bash
python scripts/train.py   --config configs/v1.yaml   --phase fm   --cno_ckpt runs/v1/cno/checkpoints/last.ckpt
```

Run inference:

```bash
python scripts/infer.py   --config configs/v1.yaml   --cno_ckpt runs/v1/cno/checkpoints/last.ckpt   --fm_ckpt runs/v1/fm/checkpoints/last.ckpt   --ensemble_members 5   --output_dir results/v1_ens5
```

## Data

The pipeline was designed around Copernicus Marine GLORYS daily reanalysis fields. The preprocessing stage creates ready-to-train tensors:

```text
GLORYS HR 1/12 degree daily
  -> coarsen to 1.5 degree
  -> interpolate to 1/4 degree experimental LR
  -> save HR, LR, masks and stats to Zarr
```

See [docs/data.md](docs/data.md) for more details.

## Evaluation

OceanLens is evaluated with:

- RMSE / MAE by variable
- current-speed diagnostics
- kinetic-energy spectra
- temperature-front diagnostics using log-gradient losses
- ensemble/probabilistic metrics

See [docs/results.md](docs/results.md) for the intended reporting structure.

## HPC Notes

SLURM launchers are included for GPU training and evaluation. Cluster-specific commands and Copernicus download notes are documented separately in [docs/hpc.md](docs/hpc.md).

## Skills Demonstrated

PyTorch, PyTorch Lightning, neural operators, Flow Matching, scientific machine learning, NetCDF, Zarr, xarray, geophysical data preprocessing, GPU training, SLURM, reproducible ML workflows.

## References

- Raonic et al., 2023 - Convolutional Neural Operators
- Lipman et al., 2023 - Flow Matching
- Copernicus Marine Service - GLORYS ocean reanalysis

## Status

Research / apprenticeship project. Some datasets and trained checkpoints are not committed to this repository.
