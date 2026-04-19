# OceanLens_v2

Clean restart of OceanLens.

`v1` is the full architecture from the beginning:

```text
LR -> residual CNO -> mu -> Flow Matching residual -> HR prediction
```

with

```text
mu = LR + CNO(LR)
HR_hat = mu + FM(z, mu)
```

The goal is to avoid the prototype issues from `OceanLens_git`:

- no NetCDF opening inside every training batch;
- no ambiguous nearest/bilinear training input;
- clean land/NaN handling before normalization;
- CNO resampling isolated in explicit code;
- metrics available from day one;
- `ensemble_members` supported in inference.

## Data Pipeline

The preprocessing stage creates ready-to-train tensors:

```text
GLORYS HR 1/12 daily
  -> coarsen to 1.5 degree
  -> interpolate to 1/4 degree experimental LR
  -> save HR, LR, masks and stats to Zarr
```

The `1/4` LR tensor is an experimental LR representation: it contains information degraded to `1.5 deg`, represented on a `1/4 deg` grid.

## v1 Model

- CNO branch learns a deterministic residual:

```text
delta = HR - LR
mu = LR + delta
```

- Flow Matching branch learns the remaining stochastic residual:

```text
r = HR - mu
```

- FM is conditioned on `mu`.
- Temperature-front loss is available through:

```text
log(|grad(thetao)| + eps)
```

## Typical Commands

Preprocess:

```bash
python scripts/preprocess_glorys.py --config configs/v1.yaml
```

Train CNO:

```bash
python scripts/train.py --config configs/v1.yaml --phase cno
```

Train FM:

```bash
python scripts/train.py --config configs/v1.yaml --phase fm --cno_ckpt runs/v1/cno/checkpoints/last.ckpt
```

Inference:

```bash
python scripts/infer.py \
  --config configs/v1.yaml \
  --cno_ckpt runs/v1/cno/checkpoints/last.ckpt \
  --fm_ckpt runs/v1/fm/checkpoints/last.ckpt \
  --ensemble_members 5 \
  --output_dir results/v1_ens5
```

Evaluate:

```bash
python scripts/evaluate.py --result_dir results/v1_ens5
```

