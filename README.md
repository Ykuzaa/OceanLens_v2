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

### 1. Pull the latest code on LIR/HPC

```bash
ssh <your_login>@<lir_hpc>
cd /scratch/emboulaalam/OceanLens_v2
git pull --ff-only origin main
```

### 2. Install and login to Copernicus Marine

Run this once in the environment used on the cluster:

```bash
conda activate oceanlens
python -m pip install --upgrade copernicusmarine
copernicusmarine --version
copernicusmarine login
```

### 3. Download GLORYS daily files

The default dataset is the Copernicus Marine GLORYS12V1 daily reanalysis:

```text
cmems_mod_glo_phy_my_0.083deg_P1D-m
```

Set the longitude/latitude box before running the commands. The values below
are placeholders for a North Atlantic-like domain; adapt them to the scientific
domain before launching a long download.

Download train/validation years:

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens

OUT_DIR=/scratch/emboulaalam/data/glorys/raw_daily \
START_DATE=1994-01-01 \
END_DATE=2004-12-31 \
LON_MIN=-20 LON_MAX=20 \
LAT_MIN=30 LAT_MAX=60 \
bash scripts/download_glorys_daily.sh
```

Download the test year:

```bash
OUT_DIR=/scratch/emboulaalam/data/glorys/raw_daily \
START_DATE=2019-01-01 \
END_DATE=2019-12-31 \
LON_MIN=-20 LON_MAX=20 \
LAT_MIN=30 LAT_MAX=60 \
bash scripts/download_glorys_daily.sh
```

The script writes files as:

```text
/scratch/emboulaalam/data/glorys/raw_daily/YYYY/glorys_YYYY-MM-DD.nc
```

### 4. Preprocess to Zarr

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens
python scripts/preprocess_glorys.py --config configs/v1.yaml
```

Or submit the existing SLURM job:

```bash
sbatch slurm/preprocess.sbatch
```

### 5. Train CNO

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens
python scripts/train.py --config configs/v1.yaml --phase cno
```

Or with SLURM:

```bash
sbatch slurm/train_v1_cno.sbatch
```

### 6. Train FM

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens
python scripts/train.py --config configs/v1.yaml --phase fm --cno_ckpt runs/v1/cno/checkpoints/last.ckpt
```

Or with SLURM:

```bash
sbatch slurm/train_v1_fm.sbatch
```

### 7. Inference

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens
python scripts/infer.py \
  --config configs/v1.yaml \
  --cno_ckpt runs/v1/cno/checkpoints/last.ckpt \
  --fm_ckpt runs/v1/fm/checkpoints/last.ckpt \
  --ensemble_members 5 \
  --output_dir results/v1_ens5
```

### 8. Evaluate

```bash
cd /scratch/emboulaalam/OceanLens_v2
conda activate oceanlens
python scripts/evaluate.py --result_dir results/v1_ens5
```
