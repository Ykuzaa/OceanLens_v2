#!/bin/bash
#SBATCH --job-name=ol2-v1-fm
#SBATCH --partition=training-large-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_fm_%j.out
#SBATCH --error=logs/train_fm_%j.err

set -euo pipefail
cd /scratch/emboulaalam/OceanLens_v2
mkdir -p logs
conda activate oceanlens
python scripts/train.py \
  --config configs/v1.yaml \
  --phase fm \
  --gpu 0 \
  --cno_ckpt runs/v1/cno/checkpoints/last.ckpt

