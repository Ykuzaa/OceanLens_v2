# HPC Notes

This repository includes SLURM launchers for preprocessing, training and evaluation.

## Typical Workflow

```bash
cd /scratch/<user>/OceanLens_v2
git pull --ff-only origin main
conda activate oceanlens
```

Preprocess:

```bash
python scripts/preprocess_glorys.py --config configs/v1.yaml
# or
sbatch slurm/preprocess.sbatch
```

Train CNO:

```bash
python scripts/train.py --config configs/v1.yaml --phase cno
# or
sbatch slurm/train_v1_cno.sbatch
```

Train Flow Matching:

```bash
python scripts/train.py   --config configs/v1.yaml   --phase fm   --cno_ckpt runs/v1/cno/checkpoints/last.ckpt
# or
sbatch slurm/train_v1_fm.sbatch
```

Evaluate:

```bash
python scripts/evaluate.py --result_dir results/v1_ens5
# or
sbatch slurm/eval_v1.sbatch
```

## Copernicus Downloads

Some clusters allow internet access only from login/frontal nodes. In that case, run Copernicus Marine downloads from the appropriate node with `tmux`, `screen` or `nohup`, then launch preprocessing/training jobs with SLURM.

Example:

```bash
mkdir -p logs
nohup bash scripts/download_glorys_global.sh > logs/download_glorys_global.out 2>&1 &
echo $! > logs/download_glorys_global.pid
```
