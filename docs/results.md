# Results

This repository is structured to report ocean super-resolution results with both statistical and physical diagnostics.

## Metrics

- RMSE / MAE by variable
- current-speed RMSE
- spatial correlation
- kinetic-energy spectra
- temperature-front log-gradient diagnostics
- ensemble spread / probabilistic diagnostics

## Recommended Figures

For each experiment:

1. LR input
2. HR target
3. deterministic CNO field
4. final CNO + Flow Matching prediction
5. error before/after residual generation
6. kinetic-energy spectrum comparison

## Reporting Template

```text
Experiment:
Data split:
Variables:
Model variant:
Checkpoint:
Metrics:
Main visual result:
Failure modes:
Next decision:
```

Large generated results and `.npz` inference arrays should stay out of Git.
